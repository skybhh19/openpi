"""Regression tests for the joint policy convention shared by Threading D0 and D05."""

import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import robomimic_policy


@pytest.mark.parametrize("task_name", ["Threading_D0", "Threading_D05"])
def test_joint_inputs_match_threading_d0_and_d05_data(task_name):
    del task_name  # Both environment variants use the same converted policy convention.
    agentview = np.zeros((3, 256, 256), dtype=np.float32)
    agentview[0] = 1.0
    wrist = np.full((3, 256, 256), 0.5, dtype=np.float32)
    state = np.arange(8, dtype=np.float32)
    actions = np.arange(24, dtype=np.float32).reshape(3, 8)

    transformed = robomimic_policy.RobomimicInputs(
        model_type=_model.ModelType.PI05,
        task_config=robomimic_policy.THREADING_JOINT,
    )(
        {
            "observation/agentview_image": agentview,
            "observation/eye_in_hand_image": wrist,
            "observation/state": state,
            "actions": actions,
            "prompt": b"thread it",
        }
    )

    assert transformed["image"]["base_0_rgb"].shape == (256, 256, 3)
    assert transformed["image"]["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(transformed["image"]["base_0_rgb"][..., 0], 255)
    np.testing.assert_array_equal(transformed["state"], state)
    np.testing.assert_array_equal(transformed["actions"], actions)
    assert transformed["prompt"] == "thread it"
    assert not transformed["image_mask"]["right_wrist_0_rgb"]


def test_joint_outputs_restore_robosuite_gripper_sign():
    actions = np.zeros((3, 32), dtype=np.float32)
    actions[:, :7] = np.asarray([0.1, 0.2, 0.3])[:, None]
    actions[:, 7] = [-0.2, 0.25, 1.2]

    output = robomimic_policy.RobomimicOutputs(task_config=robomimic_policy.THREADING_JOINT)({"actions": actions})[
        "actions"
    ]

    assert output.shape == (3, 8)
    np.testing.assert_allclose(output[:, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(output[:, 7], [-1.0, -0.5, 1.0])


def test_joint_inputs_reject_wrong_state_dimension():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="8-D state"):
        robomimic_policy.RobomimicInputs(
            model_type=_model.ModelType.PI05,
            task_config=robomimic_policy.THREADING_JOINT,
        )(
            {
                "observation/agentview_image": image,
                "observation/eye_in_hand_image": image,
                "observation/state": np.zeros(9),
            }
        )
