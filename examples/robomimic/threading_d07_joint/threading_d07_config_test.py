"""Regression tests for the three Threading_D07 training configs."""

import json
from pathlib import Path

import pytest

from openpi.policies import robomimic_policy
from openpi.training import config

REPO_ID = "local/robomimic_threading_d07_joint_256"
ALL_CONFIG = "pi05_robomimic_threading_d07_joint_low_mem_finetune"
ASSETS_DIR = f"assets/{ALL_CONFIG}"


@pytest.mark.parametrize(
    "name",
    [
        ALL_CONFIG,
        "pi05_robomimic_threading_d07_joint_full_only_low_mem_finetune",
        "pi05_robomimic_threading_d07_joint_partial_only_low_mem_finetune",
    ],
)
def test_d07_configs_use_joint_lora_recipe(name):
    train_config = config.get_config(name)
    assert train_config.data.repo_id == REPO_ID
    assert train_config.data.task_config == robomimic_policy.THREADING_JOINT
    assert train_config.model.action_horizon == 16
    assert train_config.model.action_dim == 32
    assert train_config.model.discrete_state_input
    assert train_config.ema_decay is None
    assert "lora" in train_config.model.paligemma_variant
    assert "lora" in train_config.model.action_expert_variant


@pytest.mark.parametrize(
    ("name", "expected_indices"),
    [
        ("pi05_robomimic_threading_d07_joint_full_only_low_mem_finetune", list(range(200, 400))),
        ("pi05_robomimic_threading_d07_joint_partial_only_low_mem_finetune", list(range(200))),
    ],
)
def test_d07_filtered_configs_use_checked_keys_and_all_data_stats(name, expected_indices):
    train_config = config.get_config(name)
    assert train_config.data.base_config is not None
    filter_path = train_config.data.base_config.lerobot_episode_indices_path
    assert filter_path is not None
    assert json.loads(Path(filter_path).read_text()) == expected_indices
    assert train_config.data.assets == config.AssetsConfig(assets_dir=ASSETS_DIR, asset_id=REPO_ID)
