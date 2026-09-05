"""Reusable input and output transforms for Robomimic policies."""

import dataclasses
import enum

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


class GripperOutputTransform(enum.Enum):
    """Conversion applied to a gripper action after model inference."""

    NONE = "none"
    CLOSURE_TO_SIGN = "closure_to_sign"


@dataclasses.dataclass(frozen=True)
class RobomimicTaskConfig:
    """Describes the observation and action conventions for a Robomimic dataset."""

    name: str
    default_prompt: str
    state_dim: int
    action_dim: int
    image_shape: tuple[int, int, int] = (256, 256, 3)
    delta_action_mask: tuple[bool, ...] | None = None
    gripper_output_transform: GripperOutputTransform = GripperOutputTransform.NONE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Robomimic task name must not be empty")
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Robomimic state and action dimensions must be positive")
        if len(self.image_shape) != 3 or self.image_shape[-1] != 3 or any(dim <= 0 for dim in self.image_shape):
            raise ValueError(f"Expected a positive HWC RGB image shape, got {self.image_shape}")
        if self.delta_action_mask is not None and len(self.delta_action_mask) > min(self.state_dim, self.action_dim):
            raise ValueError("Delta-action mask cannot exceed the task's state or action dimension")
        if self.gripper_output_transform != GripperOutputTransform.NONE and self.action_dim < 1:
            raise ValueError("A gripper output transform requires at least one action dimension")


THREADING_OSC = RobomimicTaskConfig(
    name="threading_osc",
    default_prompt="Thread the needle through the ring",
    state_dim=9,
    action_dim=7,
)


THREADING_JOINT = RobomimicTaskConfig(
    name="threading_joint",
    default_prompt="Thread the needle through the ring",
    state_dim=8,
    action_dim=8,
    delta_action_mask=(True, True, True, True, True, True, True, False),
    gripper_output_transform=GripperOutputTransform.CLOSURE_TO_SIGN,
)


THREADING_JOINT_ABSOLUTE = RobomimicTaskConfig(
    name="threading_joint_absolute",
    default_prompt="Thread the needle through the ring",
    state_dim=8,
    action_dim=8,
    # The converted dataset already stores absolute Panda joint targets. With no
    # delta mask, training and inference preserve those targets end to end.
    delta_action_mask=None,
    gripper_output_transform=GripperOutputTransform.CLOSURE_TO_SIGN,
)


def make_robomimic_example(task_config: RobomimicTaskConfig) -> dict:
    """Create a random example matching a configured Robomimic task."""
    return {
        "observation/state": np.random.rand(task_config.state_dim).astype(np.float32),
        "observation/agentview_image": np.random.randint(256, size=task_config.image_shape, dtype=np.uint8),
        "observation/eye_in_hand_image": np.random.randint(256, size=task_config.image_shape, dtype=np.uint8),
        "prompt": task_config.default_prompt,
    }


def _parse_image(image, *, task_config: RobomimicTaskConfig) -> np.ndarray:
    """Convert LeRobot CHW floats or inference HWC images to HWC uint8."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3-D image for {task_config.name}, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape != task_config.image_shape:
        height, width, channels = task_config.image_shape
        raise ValueError(
            f"Expected a {height}x{width}x{channels} image for {task_config.name}, got shape {image.shape}"
        )
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 + 1e-6 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class RobomimicInputs(transforms.DataTransformFn):
    """Convert standard Robomimic observations into Pi model inputs."""

    model_type: _model.ModelType
    task_config: RobomimicTaskConfig

    def __call__(self, data: dict) -> dict:
        agentview = _parse_image(data["observation/agentview_image"], task_config=self.task_config)
        eye_in_hand = _parse_image(data["observation/eye_in_hand_image"], task_config=self.task_config)
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (self.task_config.state_dim,):
            raise ValueError(
                f"Expected {self.task_config.state_dim}-D state for {self.task_config.name}, got {state.shape}"
            )

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": agentview,
                "left_wrist_0_rgb": eye_in_hand,
                "right_wrist_0_rgb": np.zeros_like(agentview),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim < 1 or actions.shape[-1] != self.task_config.action_dim:
                raise ValueError(
                    f"Expected {self.task_config.action_dim}-D actions for {self.task_config.name}, got {actions.shape}"
                )
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode() if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class RobomimicOutputs(transforms.DataTransformFn):
    """Remove model padding and restore a task's inference action convention."""

    task_config: RobomimicTaskConfig

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        action_dim = self.task_config.action_dim
        if actions.ndim != 2 or actions.shape[1] < action_dim:
            raise ValueError(
                f"Expected action chunk shaped (T, >= {action_dim}) for {self.task_config.name}, got {actions.shape}"
            )
        actions = actions[:, :action_dim].copy()
        if self.task_config.gripper_output_transform == GripperOutputTransform.CLOSURE_TO_SIGN:
            actions[:, -1] = 2.0 * np.clip(actions[:, -1], 0.0, 1.0) - 1.0
        return {"actions": actions}
