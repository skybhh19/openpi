"""Regression tests for Threading_D06_Hard training configs."""

import json
from pathlib import Path
import random

import pytest

from openpi.policies import robomimic_policy
from openpi.training import config

REPO_ID = "local/robomimic_threading_d06_hard_joint_256"
ALL_CONFIG = "pi05_robomimic_threading_d06_hard_joint_low_mem_finetune"
ASSETS_DIR = f"assets/{ALL_CONFIG}"
ABSOLUTE_ALL_CONFIG = "pi05_robomimic_threading_d06_hard_joint_absolute_low_mem_finetune"
ABSOLUTE_ASSETS_DIR = f"assets/{ABSOLUTE_ALL_CONFIG}"
FULL_FILTER = list(range(200, 400))
PARTIAL_FILTER = list(range(200))
_SAMPLE_RNG = random.Random(0)
SAMPLED_PARTIAL_FILTER = sorted(_SAMPLE_RNG.sample(PARTIAL_FILTER, 50))
SAMPLED_FULL_FILTER = sorted(_SAMPLE_RNG.sample(FULL_FILTER, 50))
SAMPLED_100_FILTER = sorted(SAMPLED_PARTIAL_FILTER + SAMPLED_FULL_FILTER)


@pytest.mark.parametrize(
    "name",
    [
        ALL_CONFIG,
        "pi05_robomimic_threading_d06_hard_joint_full_only_low_mem_finetune",
        "pi05_robomimic_threading_d06_hard_joint_partial_only_low_mem_finetune",
        "pi05_robomimic_threading_d06_hard_joint_sampled_100_low_mem_finetune",
        "pi05_robomimic_threading_d06_hard_joint_sampled_full_only_low_mem_finetune",
        "pi05_robomimic_threading_d06_hard_joint_sampled_partial_only_low_mem_finetune",
    ],
)
def test_d06_hard_configs_use_joint_lora_recipe(name):
    train_config = config.get_config(name)
    assert train_config.data.repo_id == REPO_ID
    assert train_config.data.task_config == robomimic_policy.THREADING_JOINT
    assert train_config.model.action_horizon == 16
    assert train_config.model.action_dim == 32
    assert train_config.model.discrete_state_input
    assert train_config.ema_decay is None
    assert "lora" in train_config.model.paligemma_variant
    assert "lora" in train_config.model.action_expert_variant
    assert train_config.num_train_steps == 20_000
    assert train_config.save_interval == 5_000
    assert train_config.keep_period == 5_000


@pytest.mark.parametrize(
    ("name", "expected_indices"),
    [
        ("pi05_robomimic_threading_d06_hard_joint_full_only_low_mem_finetune", FULL_FILTER),
        ("pi05_robomimic_threading_d06_hard_joint_partial_only_low_mem_finetune", PARTIAL_FILTER),
        ("pi05_robomimic_threading_d06_hard_joint_sampled_100_low_mem_finetune", SAMPLED_100_FILTER),
        (
            "pi05_robomimic_threading_d06_hard_joint_sampled_full_only_low_mem_finetune",
            SAMPLED_FULL_FILTER,
        ),
        (
            "pi05_robomimic_threading_d06_hard_joint_sampled_partial_only_low_mem_finetune",
            SAMPLED_PARTIAL_FILTER,
        ),
    ],
)
def test_d06_hard_filtered_configs_use_checked_keys_and_all_data_stats(name, expected_indices):
    train_config = config.get_config(name)
    assert train_config.data.base_config is not None
    filter_path = train_config.data.base_config.lerobot_episode_indices_path
    assert filter_path is not None
    assert json.loads(Path(filter_path).read_text()) == expected_indices
    assert train_config.data.assets == config.AssetsConfig(assets_dir=ASSETS_DIR, asset_id=REPO_ID)


@pytest.mark.parametrize(
    ("name", "expected_indices"),
    [
        (ABSOLUTE_ALL_CONFIG, None),
        ("pi05_robomimic_threading_d06_hard_joint_absolute_full_only_low_mem_finetune", FULL_FILTER),
        ("pi05_robomimic_threading_d06_hard_joint_absolute_partial_only_low_mem_finetune", PARTIAL_FILTER),
    ],
)
def test_d06_hard_absolute_configs_preserve_joint_targets_and_share_absolute_stats(name, expected_indices):
    train_config = config.get_config(name)
    assert train_config.data.repo_id == REPO_ID
    assert train_config.data.task_config == robomimic_policy.THREADING_JOINT_ABSOLUTE
    assert train_config.data.task_config.delta_action_mask is None
    assert train_config.num_train_steps == 20_000
    assert train_config.save_interval == 5_000
    assert train_config.keep_period == 5_000

    if expected_indices is None:
        assert train_config.data.base_config.lerobot_episode_indices_path is None
        assert train_config.data.assets == config.AssetsConfig()
    else:
        filter_path = train_config.data.base_config.lerobot_episode_indices_path
        assert json.loads(Path(filter_path).read_text()) == expected_indices
        assert train_config.data.assets == config.AssetsConfig(
            assets_dir=ABSOLUTE_ASSETS_DIR,
            asset_id=REPO_ID,
        )
