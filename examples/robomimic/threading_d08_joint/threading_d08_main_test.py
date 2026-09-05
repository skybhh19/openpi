"""Focused tests for Threading_D08 evaluation camera integration."""

import numpy as np

from examples.robomimic.threading_d08_joint import main as evaluation
from examples.robomimic.threading_d08_joint.threading_d08_conversion_test import _env_args


def test_environment_kwargs_preserve_full_agentview():
    kwargs = evaluation.make_environment_kwargs(_env_args(), max_steps=123, render_gpu_device_id=2, seed=7)
    assert kwargs["camera_names"] == ["agentview_full", "robot0_eye_in_hand"]
    assert kwargs["horizon"] == 123
    assert kwargs["render_gpu_device_id"] == 2
    assert kwargs["seed"] == 7


def test_agentview_alias_maps_without_copying_pixels():
    full = np.zeros((256, 256, 3), dtype=np.uint8)
    observation = {"agentview_full_image": full, "robot0_joint_pos": np.zeros(7)}
    aliased = evaluation._alias_agentview(observation)  # noqa: SLF001
    assert aliased["agentview_image"] is full
    assert "agentview_image" not in observation
