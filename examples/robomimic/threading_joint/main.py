"""Evaluate an OpenPI joint-position policy from fresh Threading environment resets."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np
import tqdm
import tyro

try:
    from .convert_robomimic_data_to_lerobot import ACTION_DIM
    from .convert_robomimic_data_to_lerobot import DEFAULT_DATA_PATH
    from .convert_robomimic_data_to_lerobot import DEFAULT_TASK_PROMPT
    from .convert_robomimic_data_to_lerobot import IMAGE_HEIGHT
    from .convert_robomimic_data_to_lerobot import IMAGE_SHAPE
    from .convert_robomimic_data_to_lerobot import IMAGE_WIDTH
    from .convert_robomimic_data_to_lerobot import STATE_DIM
    from .convert_robomimic_data_to_lerobot import panda_gripper_qpos_to_closure
    from .convert_robomimic_data_to_lerobot import validate_environment_metadata
except ImportError:  # Allows running this file directly.
    from convert_robomimic_data_to_lerobot import ACTION_DIM
    from convert_robomimic_data_to_lerobot import DEFAULT_DATA_PATH
    from convert_robomimic_data_to_lerobot import DEFAULT_TASK_PROMPT
    from convert_robomimic_data_to_lerobot import IMAGE_HEIGHT
    from convert_robomimic_data_to_lerobot import IMAGE_SHAPE
    from convert_robomimic_data_to_lerobot import IMAGE_WIDTH
    from convert_robomimic_data_to_lerobot import STATE_DIM
    from convert_robomimic_data_to_lerobot import panda_gripper_qpos_to_closure
    from convert_robomimic_data_to_lerobot import validate_environment_metadata


@dataclasses.dataclass
class Args:
    """Closed-loop evaluation options. HDF5 is used only for environment metadata."""

    host: str = "localhost"
    port: int = 8000
    dataset_path: str = str(DEFAULT_DATA_PATH)
    dataset_episode_indices_path: str | None = None
    episode_indices_path: str | None = None
    num_episodes: int = 20
    start_episode: int = 0
    seed: int = 0
    max_steps: int = 1000
    replan_steps: int = 8
    prompt: str = DEFAULT_TASK_PROMPT
    render_gpu_device_id: int = 0
    output_path: str = "data/robomimic_threading_d0_joint_256_eval.json"
    video_dir: str | None = None
    video_fps: int = 20
    concat_wrist_video: bool = False
    stop_on_success: bool = True


MAX_MEAN_ADJACENT_PIXEL_DIFFERENCE = 40.0


def make_environment_kwargs(
    env_args: dict[str, Any],
    *,
    max_steps: int,
    render_gpu_device_id: int,
    seed: int,
    metadata_validator=None,
) -> dict:
    """Keep the recorded absolute controller while setting evaluation options."""
    fps = (metadata_validator or validate_environment_metadata)(env_args)
    env_kwargs = copy.deepcopy(env_args["env_kwargs"])
    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": IMAGE_HEIGHT,
            "camera_widths": IMAGE_WIDTH,
            "camera_depths": False,
            "control_freq": fps,
            "horizon": max_steps,
            "ignore_done": True,
            # Fresh task placements are sampled in _reset_internal(). Reusing
            # one simulation avoids rebuilding and leaking an EGL context per trial.
            "hard_reset": False,
            "render_gpu_device_id": render_gpu_device_id,
            "seed": seed,
        }
    )
    return env_kwargs


def make_state(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Match the converter's [joint angles (7), measured closure (1)] state."""
    joint_position = np.asarray(observation["robot0_joint_pos"], dtype=np.float32)
    closure = np.asarray([panda_gripper_qpos_to_closure(observation["robot0_gripper_qpos"])], dtype=np.float32)
    state = np.concatenate([joint_position, closure])
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid live state: shape={state.shape}, finite={np.all(np.isfinite(state))}")
    return state


def upright_rgb(image: np.ndarray) -> np.ndarray:
    """Match robomimic: raw MuJoCo camera observations are vertically flipped."""
    image = np.asarray(image)
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"Expected a {IMAGE_HEIGHT}x{IMAGE_WIDTH} RGB camera observation, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 + 1e-6 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image[::-1].copy()


def make_video_frame(observation: dict[str, np.ndarray], *, concat_wrist: bool) -> np.ndarray:
    agentview = upright_rgb(observation["agentview_image"])
    if not concat_wrist:
        return agentview
    wrist = upright_rgb(observation["robot0_eye_in_hand_image"])
    return np.concatenate([agentview, wrist], axis=1)


def validate_live_camera_images(observation: dict[str, np.ndarray]) -> None:
    """Fail fast on the high-frequency noise produced by a broken EGL context."""
    for camera_key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.asarray(observation[camera_key], dtype=np.float32)
        vertical = np.abs(np.diff(image, axis=0)).mean()
        horizontal = np.abs(np.diff(image, axis=1)).mean()
        mean_adjacent_difference = float((vertical + horizontal) / 2.0)
        if mean_adjacent_difference > MAX_MEAN_ADJACENT_PIXEL_DIFFERENCE:
            raise RuntimeError(
                f"Corrupted {camera_key} render: mean adjacent-pixel difference "
                f"{mean_adjacent_difference:.1f} exceeds {MAX_MEAN_ADJACENT_PIXEL_DIFFERENCE:.1f}"
            )


def prepare_policy_observation(observation: dict[str, np.ndarray], prompt: str) -> dict[str, Any]:
    # Keep 256px input here; the OpenPI model transform performs the single resize to 224px.
    return {
        "observation/agentview_image": upright_rgb(observation["agentview_image"]),
        "observation/eye_in_hand_image": upright_rgb(observation["robot0_eye_in_hand_image"]),
        "observation/state": make_state(observation),
        "prompt": prompt,
    }


def prepare_action_chunk(
    response: dict[str, Any],
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    replan_steps: int,
) -> np.ndarray:
    """Validate and clip policy-server output for the absolute robosuite controller."""
    if replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    actions = np.asarray(response.get("actions"), dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"Policy returned actions with shape {actions.shape}; expected (T, >= {ACTION_DIM})")
    if len(actions) < replan_steps:
        raise ValueError(f"Policy returned only {len(actions)} actions, fewer than replan_steps={replan_steps}")
    actions = actions[:replan_steps, :ACTION_DIM]
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned non-finite actions")
    action_low = np.asarray(action_low)
    action_high = np.asarray(action_high)
    if action_low.shape != (ACTION_DIM,) or action_high.shape != (ACTION_DIM,):
        raise ValueError(f"Expected an {ACTION_DIM}-D robosuite action spec")
    # Both policy modes expose the same evaluation contract. Delta-trained configs
    # reconstruct absolute arm targets on the server; absolute-trained configs
    # already emit them. Both map gripper closure to robosuite's [-1, 1] sign.
    return np.clip(actions, action_low, action_high)


def ensure_offscreen_context_current(env) -> None:
    """Rebind EGL before rendering observations after CUDA policy inference."""
    render_context = getattr(env.sim, "_render_context_offscreen", None)
    if render_context is None:
        raise RuntimeError("Offscreen rendering is enabled but the MuJoCo render context is missing")
    render_context.gl_ctx.make_current()


def run_rollout(
    env,
    policy,
    observation: dict[str, np.ndarray],
    *,
    prompt: str,
    max_steps: int,
    replan_steps: int,
    stop_on_success: bool,
    concat_wrist_video: bool = False,
    video_writer=None,
) -> dict[str, Any]:
    action_low, action_high = env.action_spec
    validate_live_camera_images(observation)
    env._check_success()  # noqa: SLF001 -- initialize task's displacement baseline.
    success = False
    grasp_success = False
    final_success = False
    steps = 0
    inference_times = []

    if video_writer is not None:
        video_writer.append_data(make_video_frame(observation, concat_wrist=concat_wrist_video))

    while steps < max_steps and not (stop_on_success and success):
        started = time.monotonic()
        response = policy.infer(prepare_policy_observation(observation, prompt))
        inference_times.append(time.monotonic() - started)
        chunk = prepare_action_chunk(response, action_low, action_high, replan_steps=replan_steps)

        done = False
        for action in chunk:
            if steps >= max_steps:
                break
            # action[:7] is always an absolute q target, regardless of the model's training action space.
            # Robosuite's MjSim.render() assumes its EGL context is still current.
            # Explicitly rebind it because inference runs CUDA on the same GPU.
            ensure_offscreen_context_current(env)
            observation, reward, done, _ = env.step(action)
            validate_live_camera_images(observation)
            steps += 1
            grasp_success = grasp_success or bool(
                env._check_grasp(  # noqa: SLF001 -- standard robosuite bilateral-contact grasp check.
                    gripper=env.robots[0].gripper,
                    object_geoms=env.needle,
                )
            )
            final_success = bool(reward > 0.0 or env._check_success())  # noqa: SLF001
            success = success or final_success
            if video_writer is not None:
                video_writer.append_data(make_video_frame(observation, concat_wrist=concat_wrist_video))
            if done or (stop_on_success and success):
                break
        if done:
            break

    debug = getattr(env, "_threading_success_debug", {})
    return {
        "success": bool(success),
        "grasp_success": bool(grasp_success),
        "insertion_success": bool(success),
        "final_success": bool(final_success),
        "steps": steps,
        "policy_queries": len(inference_times),
        "mean_inference_seconds": float(np.mean(inference_times)) if inference_times else 0.0,
        "success_debug": _jsonable(debug),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize zero evaluation episodes")
    grasp_results = [bool(result["grasp_success"]) for result in results if "grasp_success" in result]
    insertion_results = [bool(result.get("insertion_success", result["success"])) for result in results]
    return {
        "episodes": len(results),
        "successes": sum(bool(result["success"]) for result in results),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "grasp_evaluated_episodes": len(grasp_results),
        "grasp_successes": sum(grasp_results),
        "grasp_success_rate": float(np.mean(grasp_results)) if grasp_results else None,
        "insertion_successes": sum(insertion_results),
        "insertion_success_rate": float(np.mean(insertion_results)),
        "mean_steps": float(np.mean([result["steps"] for result in results])),
    }


def load_resume_report(
    output_path: Path,
    *,
    start_episode: int,
    seed: int,
    prompt: str,
    replan_steps: int,
    max_steps: int,
    reset_source: str = "env.reset",
) -> dict[str, Any] | None:
    """Load and validate the existing prefix before appending new episodes."""
    if start_episode == 0:
        return None
    if not output_path.is_file():
        raise ValueError(f"Cannot resume at episode {start_episode}: report does not exist: {output_path}")

    report = json.loads(output_path.read_text())
    expected_fields = {
        "reset_source": reset_source,
        "seed": seed,
        "prompt": prompt,
        "replan_steps": replan_steps,
        "max_steps": max_steps,
    }
    for field, expected in expected_fields.items():
        actual = report.get(field)
        if actual != expected:
            raise ValueError(f"Cannot resume: existing {field}={actual!r}, expected {expected!r}")

    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Cannot resume: existing report has no episode list")
    episode_indices = [episode.get("episode") for episode in episodes]
    expected_indices = list(range(start_episode))
    if episode_indices != expected_indices:
        raise ValueError(
            f"Cannot resume: existing episode indices must be {expected_indices}, got {episode_indices}"
        )
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_episode_indices(path: str | None, *, start_episode: int, num_episodes: int) -> list[int]:
    if path is None:
        return list(range(start_episode, start_episode + num_episodes))
    if start_episode != 0:
        raise ValueError("start_episode must be 0 when episode_indices_path is set")
    episode_indices = json.loads(Path(path).read_text())
    if not isinstance(episode_indices, list) or not all(isinstance(index, int) for index in episode_indices):
        raise ValueError("episode_indices_path must contain a JSON list of integer episode indices")
    if len(episode_indices) != num_episodes:
        raise ValueError(
            f"num_episodes={num_episodes}, but episode_indices_path contains {len(episode_indices)} indices"
        )
    if episode_indices != sorted(set(episode_indices)) or any(index < 0 for index in episode_indices):
        raise ValueError("episode_indices_path indices must be non-negative, unique, and increasing")
    return episode_indices


def main(args: Args) -> None:
    if args.num_episodes <= 0 or args.max_steps <= 0 or args.video_fps <= 0:
        raise ValueError("num_episodes, max_steps, and video_fps must be positive")
    if args.start_episode < 0:
        raise ValueError("start_episode must be non-negative")
    if args.episode_indices_path is not None and args.dataset_episode_indices_path is not None:
        raise ValueError("episode_indices_path cannot be combined with dataset_episode_indices_path")

    episode_indices = load_episode_indices(
        args.episode_indices_path,
        start_episode=args.start_episode,
        num_episodes=args.num_episodes,
    )

    output_path = Path(args.output_path)
    reset_source = "dataset_initial_state" if args.dataset_episode_indices_path is not None else "env.reset"
    existing_report = load_resume_report(
        output_path,
        start_episode=args.start_episode,
        seed=args.seed,
        prompt=args.prompt,
        replan_steps=args.replan_steps,
        max_steps=args.max_steps,
        reset_source=reset_source,
    )

    dataset_path = Path(args.dataset_path).expanduser()
    dataset_initial_states = None
    with h5py.File(dataset_path, "r") as dataset:
        env_args = json.loads(dataset["data"].attrs["env_args"])
        validate_environment_metadata(env_args)
        if args.dataset_episode_indices_path is not None:
            episode_indices = json.loads(Path(args.dataset_episode_indices_path).read_text())
            if not isinstance(episode_indices, list) or not all(isinstance(index, int) for index in episode_indices):
                raise ValueError("dataset_episode_indices_path must contain a JSON list of integer episode indices")
            demo_names = sorted(
                dataset["data"].keys(), key=lambda name: int(name.removeprefix("demo_"))
            )
            if any(index < 0 or index >= len(demo_names) for index in episode_indices):
                raise ValueError("dataset episode index is out of range")
            episode_stop = args.start_episode + args.num_episodes
            if episode_stop > len(episode_indices):
                raise ValueError(
                    f"Requested episodes through {episode_stop}, but dataset index file has {len(episode_indices)} entries"
                )
            dataset_initial_states = [
                (episode_indices[index], demo_names[episode_indices[index]], dataset[f"data/{demo_names[episode_indices[index]]}/states"][0])
                for index in range(args.start_episode, episode_stop)
            ]

    # Deferred so helper unit tests do not initialize MuJoCo / EGL.
    import imageio
    from openpi_client import websocket_client_policy
    import robosuite

    env_kwargs = make_environment_kwargs(
        env_args,
        max_steps=args.max_steps,
        render_gpu_device_id=args.render_gpu_device_id,
        seed=args.seed,
    )
    env = robosuite.make(env_args["env_name"], **env_kwargs)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir) if args.video_dir is not None else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    results = [] if existing_report is None else list(existing_report["episodes"])
    try:
        next_reset_index = 0
        for episode_index in tqdm.tqdm(
            episode_indices,
            desc=f"Evaluating {env_args['env_name']} joint policy",
        ):
            # This is the closed-loop evaluation distribution: each trial is a
            # fresh environment reset sampled from the task's seeded RNG.
            observation = None
            if dataset_initial_states is None:
                while next_reset_index <= episode_index:
                    observation = env.reset()
                    next_reset_index += 1
                assert observation is not None
            else:
                observation = env.reset()
            source_episode = None
            source_demo = None
            if dataset_initial_states is not None:
                source_episode, source_demo, initial_state = dataset_initial_states[episode_index - args.start_episode]
                env.sim.set_state_from_flattened(initial_state)
                env.sim.forward()
                env._threading_initial_tripod_pos = None  # noqa: SLF001
                env._threading_max_insert_progress = -np.inf  # noqa: SLF001
                observation = env._get_observations(force_update=True)  # noqa: SLF001
            episode_name = f"episode_{episode_index:03d}"
            writer = None
            if video_dir is not None:
                writer = imageio.get_writer(video_dir / f"{episode_name}.mp4", fps=args.video_fps)
            try:
                result = run_rollout(
                    env,
                    policy,
                    observation,
                    prompt=args.prompt,
                    max_steps=args.max_steps,
                    replan_steps=args.replan_steps,
                    stop_on_success=args.stop_on_success,
                    concat_wrist_video=args.concat_wrist_video,
                    video_writer=writer,
                )
            finally:
                if writer is not None:
                    writer.close()
            result["episode"] = episode_index
            if source_demo is not None:
                result["source_episode"] = source_episode
                result["source_demo"] = source_demo
            results.append(result)
            print(f"{episode_name}: success={result['success']} steps={result['steps']}", flush=True)
    finally:
        env.close()

    report = {
        "environment_metadata_dataset": str(dataset_path.resolve()),
        "reset_source": reset_source,
        "dataset_episode_indices_path": args.dataset_episode_indices_path,
        "episode_indices_path": args.episode_indices_path,
        "evaluation_segments": (
            [{"episode_indices": episode_indices}]
            if args.episode_indices_path is not None
            else (
                ([{"start_episode": 0, "num_episodes": args.start_episode}] if existing_report is not None else [])
                + [{"start_episode": args.start_episode, "num_episodes": args.num_episodes}]
            )
        ),
        "seed": args.seed,
        "prompt": args.prompt,
        "replan_steps": args.replan_steps,
        "max_steps": args.max_steps,
        "video_views": "agentview+wrist" if args.concat_wrist_video else "agentview",
        "action_space": "absolute Panda joint target (7) + robosuite gripper sign (1)",
        "environment": env_args,
        "summary": summarize_results(results),
        "episodes": results,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
