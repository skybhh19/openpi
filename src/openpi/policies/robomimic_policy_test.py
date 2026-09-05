import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import robomimic_policy as policy


@pytest.mark.parametrize(
    ("task_config", "state_dim", "action_dim"),
    [
        (policy.THREADING_OSC, 9, 7),
        (policy.THREADING_JOINT, 8, 8),
        (policy.THREADING_JOINT_ABSOLUTE, 8, 8),
    ],
)
def test_inputs_convert_images_and_preserve_task_data(task_config, state_dim, action_dim):
    agentview = np.zeros((3, 256, 256), dtype=np.float32)
    agentview[0] = 1.0
    wrist = np.full((3, 256, 256), 0.5, dtype=np.float32)
    actions = np.arange(3 * action_dim, dtype=np.float32).reshape(3, action_dim)
    state = np.arange(state_dim, dtype=np.float32)

    transformed = policy.RobomimicInputs(model_type=_model.ModelType.PI05, task_config=task_config)(
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
    np.testing.assert_array_equal(transformed["actions"], actions)
    np.testing.assert_array_equal(transformed["state"], state)
    assert not transformed["image_mask"]["right_wrist_0_rgb"]
    assert transformed["prompt"] == "thread it"


def test_inputs_support_another_robomimic_task_configuration():
    task_config = policy.RobomimicTaskConfig(
        name="lift_joint",
        default_prompt="Lift the cube",
        state_dim=4,
        action_dim=3,
        image_shape=(128, 160, 3),
    )
    image = np.zeros((128, 160, 3), dtype=np.uint8)
    transformed = policy.RobomimicInputs(model_type=_model.ModelType.PI05, task_config=task_config)(
        {
            "observation/agentview_image": image,
            "observation/eye_in_hand_image": image,
            "observation/state": np.zeros(4),
        }
    )

    assert transformed["state"].shape == (4,)
    assert transformed["image"]["base_0_rgb"].shape == (128, 160, 3)


def test_osc_outputs_remove_model_action_padding_without_changing_values():
    actions = np.arange(64, dtype=np.float32).reshape(2, 32)
    output = policy.RobomimicOutputs(task_config=policy.THREADING_OSC)({"actions": actions})["actions"]
    np.testing.assert_array_equal(output, actions[:, :7])


def test_joint_outputs_crop_padding_and_map_closure_to_robosuite_sign():
    actions = np.zeros((3, 32), dtype=np.float32)
    actions[:, :7] = np.asarray([0.1, 0.2, 0.3])[:, None]
    actions[:, 7] = [-0.2, 0.25, 1.2]
    output = policy.RobomimicOutputs(task_config=policy.THREADING_JOINT)({"actions": actions})["actions"]

    assert output.shape == (3, 8)
    np.testing.assert_allclose(output[:, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(output[:, 7], [-1.0, -0.5, 1.0])


def test_absolute_joint_outputs_preserve_arm_targets_and_map_gripper():
    actions = np.zeros((2, 32), dtype=np.float32)
    expected_arm_targets = np.asarray(
        [[0.1, -0.2, 0.3, -1.8, 0.5, 2.0, 0.7], [0.2, -0.1, 0.4, -1.7, 0.6, 2.1, 0.8]],
        dtype=np.float32,
    )
    actions[:, :7] = expected_arm_targets
    actions[:, 7] = [0.0, 1.0]

    output = policy.RobomimicOutputs(task_config=policy.THREADING_JOINT_ABSOLUTE)({"actions": actions})[
        "actions"
    ]

    np.testing.assert_array_equal(output[:, :7], expected_arm_targets)
    np.testing.assert_array_equal(output[:, 7], [-1.0, 1.0])


@pytest.mark.parametrize(
    "task_config", [policy.THREADING_OSC, policy.THREADING_JOINT, policy.THREADING_JOINT_ABSOLUTE]
)
def test_inputs_reject_wrong_state_dimension(task_config):
    image = np.zeros(task_config.image_shape, dtype=np.uint8)
    with pytest.raises(ValueError, match=rf"{task_config.state_dim}-D state"):
        policy.RobomimicInputs(model_type=_model.ModelType.PI05, task_config=task_config)(
            {
                "observation/agentview_image": image,
                "observation/eye_in_hand_image": image,
                "observation/state": np.zeros(task_config.state_dim + 1),
            }
        )


def test_inputs_reject_wrong_image_resolution():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="256x256x3"):
        policy.RobomimicInputs(model_type=_model.ModelType.PI05, task_config=policy.THREADING_OSC)(
            {
                "observation/agentview_image": image,
                "observation/eye_in_hand_image": image,
                "observation/state": np.zeros(9),
            }
        )


def test_make_example_uses_task_configuration():
    example = policy.make_robomimic_example(policy.THREADING_JOINT)
    assert example["observation/state"].shape == (8,)
    assert example["observation/agentview_image"].shape == (256, 256, 3)
    assert example["prompt"] == policy.THREADING_JOINT.default_prompt
