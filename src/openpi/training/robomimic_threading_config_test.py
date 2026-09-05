"""Regression tests for the registered Threading D0 and D05 fine-tuning configs."""

import json
from pathlib import Path

import pytest

from openpi.policies import robomimic_policy
from openpi.training import config

D0_OSC_REPO_ID = "local/robomimic_threading_d0_osc_v3_256"
D0_JOINT_REPO_ID = "local/robomimic_threading_d0_joint_v3_256"
D05_JOINT_REPO_ID = "local/robomimic_threading_d05_joint_v2_256"


@pytest.mark.parametrize(
    "name",
    [
        train_config.name
        for train_config in config._CONFIGS  # noqa: SLF001 - audit the complete registered config set.
        if "robomimic_threading" in train_config.name
    ],
)
def test_all_threading_configs_save_and_keep_checkpoints_every_5000_steps(name):
    train_config = config.get_config(name)
    assert train_config.save_interval == 5_000
    assert train_config.keep_period == 5_000


@pytest.mark.parametrize(
    ("name", "repo_id", "task_config", "low_mem"),
    [
        (
            "pi05_robomimic_threading_d0_osc_finetune",
            D0_OSC_REPO_ID,
            robomimic_policy.THREADING_OSC,
            False,
        ),
        (
            "pi05_robomimic_threading_d0_osc_low_mem_finetune",
            D0_OSC_REPO_ID,
            robomimic_policy.THREADING_OSC,
            True,
        ),
        (
            "pi05_robomimic_threading_d0_joint_finetune",
            D0_JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            False,
        ),
        (
            "pi05_robomimic_threading_d0_joint_low_mem_finetune",
            D0_JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            True,
        ),
        (
            "pi05_robomimic_threading_d05_joint_low_mem_finetune",
            D05_JOINT_REPO_ID,
            robomimic_policy.THREADING_JOINT,
            True,
        ),
    ],
)
def test_threading_training_configs(name, repo_id, task_config, low_mem):
    train_config = config.get_config(name)

    assert train_config.data.repo_id == repo_id
    assert train_config.data.task_config == task_config
    assert train_config.model.action_horizon == 16
    assert train_config.model.action_dim == 32
    assert train_config.model.discrete_state_input
    assert (train_config.ema_decay is None) is low_mem
    assert ("lora" in train_config.model.paligemma_variant) is low_mem
    assert ("lora" in train_config.model.action_expert_variant) is low_mem


@pytest.mark.parametrize(
    ("name", "repo_id", "expected_indices", "assets_dir"),
    [
        (
            "pi05_robomimic_threading_d0_joint_label_full_low_mem_finetune",
            D0_JOINT_REPO_ID,
            list(range(100, 200)),
            "assets/pi05_robomimic_threading_d0_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d0_joint_label_partial_low_mem_finetune",
            D0_JOINT_REPO_ID,
            list(range(100)),
            "assets/pi05_robomimic_threading_d0_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d05_joint_full_only_low_mem_finetune",
            D05_JOINT_REPO_ID,
            list(range(100, 200)),
            "assets/pi05_robomimic_threading_d05_joint_low_mem_finetune",
        ),
        (
            "pi05_robomimic_threading_d05_joint_partial_only_low_mem_finetune",
            D05_JOINT_REPO_ID,
            list(range(100)),
            "assets/pi05_robomimic_threading_d05_joint_low_mem_finetune",
        ),
    ],
)
def test_threading_filtered_configs_reuse_all_data_stats(name, repo_id, expected_indices, assets_dir):
    train_config = config.get_config(name)
    assert train_config.data.base_config is not None
    filter_path = train_config.data.base_config.lerobot_episode_indices_path

    assert filter_path is not None
    assert json.loads(Path(filter_path).read_text()) == expected_indices
    assert train_config.data.assets == config.AssetsConfig(assets_dir=assets_dir, asset_id=repo_id)
    assert train_config.data.task_config == robomimic_policy.THREADING_JOINT
