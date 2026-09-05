import json
from pathlib import Path

import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import robomimic_policy
from openpi.training import config

OSC_REPO_ID = "local/robomimic_threading_d0_osc_v3_256"
JOINT_REPO_ID = "local/robomimic_threading_d0_joint_v3_256"
D05_JOINT_REPO_ID = "local/robomimic_threading_d05_joint_v2_256"
D05_JOINT_V3_REPO_ID = "local/robomimic_threading_d05_joint_v3_256"


def _make_data_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_config):
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda self, model_config: transforms.Group())
    model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=32,
        action_horizon=16,
        discrete_state_input=True,
    )
    return config.LeRobotRobomimicDataConfig(
        repo_id="local/test",
        task_config=task_config,
        base_config=config.DataConfig(prompt_from_task=True),
    ).create(tmp_path, model_config)


def test_osc_data_pipeline_preserves_controller_actions(monkeypatch, tmp_path):
    data_config = _make_data_config(tmp_path, monkeypatch, robomimic_policy.THREADING_OSC)
    actions = np.arange(112, dtype=np.float32).reshape(16, 7)
    raw = {
        "agentview_image": np.zeros((3, 256, 256), dtype=np.float32),
        "eye_in_hand_image": np.zeros((3, 256, 256), dtype=np.float32),
        "state": np.arange(9, dtype=np.float32),
        "actions": actions.copy(),
        "prompt": "thread it",
    }

    repacked = data_config.repack_transforms.inputs[0](raw)
    transformed = data_config.data_transforms.inputs[0](repacked)

    np.testing.assert_array_equal(transformed["actions"], actions)
    assert transformed["state"].shape == (9,)
    assert data_config.action_sequence_keys == ("actions",)
    assert data_config.prompt_from_task
    assert data_config.use_quantile_norm


def test_joint_data_pipeline_round_trips_delta_actions(monkeypatch, tmp_path):
    data_config = _make_data_config(tmp_path, monkeypatch, robomimic_policy.THREADING_JOINT)
    state = np.asarray([0.1, 0.2, 0.3, -1.9, 0.5, 2.1, 0.7, 0.25], dtype=np.float32)
    absolute_actions = np.tile(state, (16, 1))
    absolute_actions[:, :7] += 0.04
    absolute_actions[:, 7] = 0.75
    raw = {
        "agentview_image": np.zeros((3, 256, 256), dtype=np.float32),
        "eye_in_hand_image": np.zeros((3, 256, 256), dtype=np.float32),
        "state": state,
        # DeltaActions operates in place, so keep the expected absolute targets independent.
        "actions": absolute_actions.copy(),
        "prompt": "thread it",
    }

    transformed = data_config.repack_transforms.inputs[0](raw)
    for transform in data_config.data_transforms.inputs:
        transformed = transform(transformed)
    np.testing.assert_allclose(transformed["actions"][:, :7], 0.04, atol=1e-7)
    np.testing.assert_allclose(transformed["actions"][:, 7], 0.75)

    model_output = {"state": state.copy(), "actions": np.pad(transformed["actions"], ((0, 0), (0, 24)))}
    for transform in data_config.data_transforms.outputs:
        model_output = transform(model_output)
    np.testing.assert_allclose(model_output["actions"][:, :7], absolute_actions[:, :7], atol=1e-7)
    # AbsoluteActions leaves the gripper untouched; RobomimicOutputs maps closure 0.75 to sign +0.5.
    np.testing.assert_allclose(model_output["actions"][:, 7], 0.5)
    assert data_config.action_sequence_keys == ("actions",)
    assert data_config.prompt_from_task
    assert data_config.use_quantile_norm


def test_absolute_joint_data_pipeline_preserves_absolute_actions(monkeypatch, tmp_path):
    data_config = _make_data_config(tmp_path, monkeypatch, robomimic_policy.THREADING_JOINT_ABSOLUTE)
    state = np.asarray([0.1, 0.2, 0.3, -1.9, 0.5, 2.1, 0.7, 0.25], dtype=np.float32)
    absolute_actions = np.tile(state, (16, 1))
    absolute_actions[:, :7] += np.linspace(0.01, 0.16, 16, dtype=np.float32)[:, None]
    absolute_actions[:, 7] = 0.75
    raw = {
        "agentview_image": np.zeros((3, 256, 256), dtype=np.float32),
        "eye_in_hand_image": np.zeros((3, 256, 256), dtype=np.float32),
        "state": state,
        "actions": absolute_actions.copy(),
        "prompt": "thread it",
    }

    transformed = data_config.repack_transforms.inputs[0](raw)
    for transform in data_config.data_transforms.inputs:
        transformed = transform(transformed)
    np.testing.assert_array_equal(transformed["actions"], absolute_actions)

    model_output = {"state": state.copy(), "actions": np.pad(transformed["actions"], ((0, 0), (0, 24)))}
    for transform in data_config.data_transforms.outputs:
        model_output = transform(model_output)
    np.testing.assert_array_equal(model_output["actions"][:, :7], absolute_actions[:, :7])
    np.testing.assert_allclose(model_output["actions"][:, 7], 0.5)


@pytest.mark.parametrize(
    ("name", "repo_id", "task_config", "low_mem", "asset_id"),
    [
        (
            "pi05_robomimic_threading_d0_osc_finetune",
            OSC_REPO_ID,
            robomimic_policy.THREADING_OSC,
            False,
            None,
        ),
        (
            "pi05_robomimic_threading_d0_osc_low_mem_finetune",
            OSC_REPO_ID,
            robomimic_policy.THREADING_OSC,
            True,
            None,
        ),
        (
            "pi05_robomimic_threading_d0_joint_finetune",
            JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            False,
            None,
        ),
        (
            "pi05_robomimic_threading_d0_joint_low_mem_finetune",
            JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            True,
            None,
        ),
        (
            "pi05_robomimic_threading_d0_joint_franka_stats_finetune",
            JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            False,
            "franka",
        ),
        (
            "pi05_robomimic_threading_d0_joint_franka_stats_low_mem_finetune",
            JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            True,
            "franka",
        ),
        (
            "pi05_robomimic_threading_d05_joint_low_mem_finetune",
            D05_JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            True,
            None,
        ),
        (
            "pi05_robomimic_threading_d05_joint_v3_low_mem_finetune",
            D05_JOINT_V3_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            True,
            None,
        ),
    ],
)
def test_registered_training_configs_match_the_data_and_finetuning_recipe(
    name, repo_id, task_config, low_mem, asset_id
):
    train_config = config.get_config(name)

    assert train_config.data.repo_id == repo_id
    assert train_config.data.task_config == task_config
    assert train_config.data.assets.asset_id == asset_id
    assert train_config.model.action_horizon == 16
    assert train_config.model.action_dim == 32
    assert train_config.model.discrete_state_input
    if low_mem:
        assert train_config.ema_decay is None
    else:
        assert train_config.ema_decay == 0.999
    assert ("lora" in train_config.model.paligemma_variant) is low_mem
    assert ("lora" in train_config.model.action_expert_variant) is low_mem


@pytest.mark.parametrize(
    ("name", "repo_id", "task_config", "expected_indices", "assets_dir"),
    [
        (
            "pi05_robomimic_threading_d0_osc_label_full_low_mem_finetune",
            OSC_REPO_ID,
            robomimic_policy.THREADING_OSC,
            list(range(100, 200)),
            "assets/pi05_robomimic_threading_d0_osc_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d0_osc_label_partial_low_mem_finetune",
            OSC_REPO_ID,
            robomimic_policy.THREADING_OSC,
            list(range(100)),
            "assets/pi05_robomimic_threading_d0_osc_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d0_joint_label_full_low_mem_finetune",
            JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            list(range(100, 200)),
            "assets/pi05_robomimic_threading_d0_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d0_joint_label_partial_low_mem_finetune",
            JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            list(range(100)),
            "assets/pi05_robomimic_threading_d0_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d05_joint_full_only_low_mem_finetune",
            D05_JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            list(range(100, 200)),
            "assets/pi05_robomimic_threading_d05_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d05_joint_partial_only_low_mem_finetune",
            D05_JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            list(range(100)),
            "assets/pi05_robomimic_threading_d05_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d05_joint_v3_full_only_low_mem_finetune",
            D05_JOINT_V3_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            list(range(150, 300)),
            "assets/pi05_robomimic_threading_d05_joint_v3_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d05_joint_v3_partial_only_low_mem_finetune",
            D05_JOINT_V3_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            list(range(150)),
            "assets/pi05_robomimic_threading_d05_joint_v3_low_mem_finetune",
        ),
    ],
)
def test_filtered_configs_select_episode_masks_and_reuse_all_data_stats(
    name, repo_id, task_config, expected_indices, assets_dir
):
    train_config = config.get_config(name)

    assert train_config.data.repo_id == repo_id
    assert train_config.data.task_config == task_config
    assert train_config.data.base_config is not None
    filter_path = train_config.data.base_config.lerobot_episode_indices_path
    assert filter_path is not None
    assert json.loads(Path(filter_path).read_text()) == expected_indices
    assert train_config.data.assets == config.AssetsConfig(assets_dir=assets_dir, asset_id=repo_id)
    assert train_config.ema_decay is None
    assert "lora" in train_config.model.paligemma_variant
    assert "lora" in train_config.model.action_expert_variant
