import importlib
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

conversion = importlib.import_module("examples.robomimic.threading_d05_joint_v3.convert_robomimic_data_to_lerobot")


def env_args_fixture() -> dict:
    return {
        "env_name": "Threading_D05",
        "env_version": "1.5.2",
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": {
                "body_parts": {
                    "right": {
                        "type": "JOINT_POSITION",
                        "input_type": "absolute",
                        "input_min": [-3.0] * 7,
                        "input_max": [3.0] * 7,
                        "gripper": {"type": "GRIP"},
                    }
                }
            },
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": 256,
            "camera_widths": 256,
            "control_freq": 20,
        },
    }


def test_environment_and_manifest_preserve_complete_order(tmp_path: Path):
    source_path = tmp_path / "source.hdf5"
    source_path.write_bytes(b"source")
    frame_counts = {"demo_1": 3, "demo_2": 5}

    assert conversion.validate_environment_metadata(env_args_fixture()) == 20
    manifest = conversion.build_source_manifest(
        source_path=source_path,
        repo_id=conversion.DEFAULT_REPO_ID,
        env_args=env_args_fixture(),
        fps=20,
        frame_counts=frame_counts,
    )

    assert manifest["total_episodes"] == 2
    assert manifest["total_frames"] == 8
    assert manifest["episodes"] == [
        {"episode_index": 0, "source_demo": "demo_1", "length": 3},
        {"episode_index": 1, "source_demo": "demo_2", "length": 5},
    ]
    assert len(manifest["episode_order_length_sha256"]) == 64


def test_all_selection_ignores_train_valid_masks(tmp_path: Path):
    path = tmp_path / "source.hdf5"
    with h5py.File(path, "w") as source:
        data = source.create_group("data")
        data.create_group("demo_10")
        data.create_group("demo_2")
        data.create_group("demo_1")
        mask = source.create_group("mask")
        mask.create_dataset("train", data=np.asarray([b"demo_1"]))
        mask.create_dataset("valid", data=np.asarray([b"demo_2"]))
        assert conversion.selected_demo_names(source, "all") == ["demo_1", "demo_2", "demo_10"]


def test_joint_frame_keeps_arm_absolute_and_maps_gripper():
    observation = {
        "robot0_joint_pos": np.asarray([[0.1, 0.2, 0.3, -1.8, 0.5, 2.0, 0.7]]),
        "robot0_gripper_qpos": np.asarray([[0.04, -0.04]]),
        "agentview_image": np.zeros((1, 256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((1, 256, 256, 3), dtype=np.uint8),
    }
    episode = {
        "obs": observation,
        "actions": np.asarray([[0.11, 0.21, 0.31, -1.79, 0.51, 2.01, 0.71, 1.0]]),
    }
    frame = conversion.convert_frame(episode, 0, conversion.DEFAULT_TASK_PROMPT)
    np.testing.assert_allclose(frame["actions"][:7], episode["actions"][0, :7])
    assert frame["actions"][7] == 1.0
    assert frame["state"][7] == 0.0


def test_wrong_task_or_controller_is_rejected():
    args = env_args_fixture()
    args["env_name"] = "Threading_D07"
    with pytest.raises(ValueError, match="env_name"):
        conversion.validate_environment_metadata(args)

    args = env_args_fixture()
    args["env_kwargs"]["controller_configs"]["body_parts"]["right"]["input_type"] = "delta"
    with pytest.raises(ValueError, match="absolute"):
        conversion.validate_environment_metadata(args)
