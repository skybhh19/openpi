"""Focused tests for the Threading_D06_Harder converter."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.robomimic.threading_d06_harder_joint_600 import convert_robomimic_data_to_lerobot as conversion


def _env_args() -> dict:
    return {
        "env_name": "Threading_D06_Harder",
        "env_version": "1.5.2",
        "env_kwargs": {
            "robots": ["Panda"],
            "controller_configs": {
                "body_parts": {
                    "right": {
                        "type": "JOINT_POSITION",
                        "input_type": "absolute",
                        "input_min": [-1.0] * 7,
                        "input_max": [1.0] * 7,
                        "gripper": {"type": "GRIP"},
                    }
                }
            },
            "camera_names": ["agentview_full", "robot0_eye_in_hand"],
            "camera_heights": 256,
            "camera_widths": 256,
        },
    }


def test_environment_metadata_requires_d06_harder_absolute_joint_schema():
    assert conversion.validate_environment_metadata(_env_args()) == 20

    wrong_task = _env_args()
    wrong_task["env_name"] = "Threading_D06_Hard"
    with pytest.raises(ValueError, match="Threading_D06_Harder"):
        conversion.validate_environment_metadata(wrong_task)


def test_staged_conversion_records_canonical_source_provenance(tmp_path: Path):
    staged = tmp_path / "staged.hdf5"
    canonical = tmp_path / "canonical.hdf5"
    staged.write_bytes(b"same-size")
    canonical.write_bytes(b"canonical")
    assert conversion.resolve_source_provenance_path(staged, str(canonical)) == canonical

    canonical.write_bytes(b"wrong-size")
    with pytest.raises(ValueError, match="sizes differ"):
        conversion.resolve_source_provenance_path(staged, str(canonical))


def test_episode_validation_and_frame_conversion(tmp_path: Path):
    source_path = tmp_path / "episode.hdf5"
    with h5py.File(source_path, "w") as source:
        episode = source.create_group("data/demo_1")
        obs = episode.create_group("obs")
        next_obs = episode.create_group("next_obs")
        action_dict = episode.create_group("action_dict")
        current = np.zeros((2, 7), dtype=np.float64)
        actions = np.full((2, 8), 0.059, dtype=np.float64)
        actions[:, -1] = [-1.0, 1.0]
        episode.create_dataset("actions", data=actions)
        action_dict.create_dataset("abs_joint_pos", data=actions[:, :7])
        obs.create_dataset("robot0_joint_pos", data=current)
        next_obs.create_dataset("robot0_joint_pos", data=np.full((2, 7), 0.04))
        obs.create_dataset("robot0_gripper_qpos", data=np.asarray([[0.04, -0.04], [0.0, 0.0]]))
        obs.create_dataset("agentview_full_image", data=np.zeros((2, 256, 256, 3), dtype=np.uint8))
        obs.create_dataset("robot0_eye_in_hand_image", data=np.zeros((2, 256, 256, 3), dtype=np.uint8))
        episode.create_dataset("rewards", data=[0, 1])
        episode.create_dataset("dones", data=[0, 1])

        assert conversion.validate_episode("demo_1", episode) == 2
        loaded = conversion.load_episode(episode)

    frame = conversion.convert_frame(loaded, 1, conversion.DEFAULT_TASK_PROMPT)
    np.testing.assert_allclose(frame["state"], np.asarray([0.0] * 7 + [1.0]))
    np.testing.assert_allclose(frame["actions"], np.asarray([0.059] * 7 + [1.0]))
    assert frame["agentview_image"].shape == (256, 256, 3)


def test_real_source_metadata_totals_and_masks_are_stable():
    with h5py.File(conversion.DEFAULT_DATA_PATH, "r") as source:
        assert conversion.validate_environment_metadata(json.loads(source["data"].attrs["env_args"])) == 20
        demos = conversion.selected_demo_names(source, "all")
        lengths = {name: int(source[f"data/{name}/actions"].shape[0]) for name in demos}
        full = {value.decode() for value in source["mask/full"][()]}
        partial = {value.decode() for value in source["mask/partial"][()]}

    conversion.validate_complete_selection(lengths)
    assert len(full) == len(partial) == 300
    assert not full & partial
    assert full | partial == set(demos)
