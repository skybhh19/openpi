#!/usr/bin/env python3
"""
Merge two pi05 checkpoints: image encoder + main backbone from one, action expert from another.

Use case: Combine pi05_droid (vision + backbone tuned on DROID) with pi05_base (action expert)
so you get DROID visual/backbone features with the base action expert.

Usage:
    # Save merged checkpoint to a local directory (default: ./checkpoints/pi05_base_droid_merged)
    uv run helper_scripts/merge_checkpoints.py --output_dir ./checkpoints/pi05_base_droid_merged

    # Custom source checkpoints (defaults shown)
    uv run helper_scripts/merge_checkpoints.py \
        --backbone_checkpoint "gs://openpi-assets/checkpoints/pi05_droid/params" \
        --action_expert_checkpoint "gs://openpi-assets/checkpoints/pi05_base/params" \
        --output_dir ./checkpoints/pi05_merged
"""

import argparse
import logging
import pathlib

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.download as download

# Force logging to stderr so messages always appear (e.g. when run via uv run)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def _ensure_commit_success(path: pathlib.Path) -> None:
    """Orbax expects commit_success.txt in checkpoint dirs; openpi-assets may not have it."""
    commit_file = path / "commit_success.txt"
    if not commit_file.exists():
        commit_file.write_text("")
        logger.debug("Wrote %s for orbax compatibility", commit_file)


def _is_action_expert_or_projection_key(key: str) -> bool:
    """True if this flat param key belongs to action expert or projection (take from base)."""
    # Action expert: PaliGemma/llm parts with _1 (e.g. attn_vec_einsum_1, kv_einsum_1, mlp_1, final_norm_1, ...)
    if key.startswith("PaliGemma/llm/") and "_1" in key:
        return True
    # Pi05 projection layers: take from base
    if key.startswith("action_in_proj/") or key.startswith("action_out_proj/"):
        return True
    if key.startswith("time_mlp_in/") or key.startswith("time_mlp_out/"):
        return True
    return False


def merge_params(backbone_params, action_expert_params):
    """
    Merge params: image + backbone from backbone_params (e.g. pi05_droid),
    action expert + projection from action_expert_params (e.g. pi05_base).

    Returns a new params dict (nested) with the merged values.
    """
    flat_backbone = flax.traverse_util.flatten_dict(backbone_params, sep="/")
    flat_base = flax.traverse_util.flatten_dict(action_expert_params, sep="/")

    result = dict(flat_backbone)
    for key, value in flat_base.items():
        if _is_action_expert_or_projection_key(key):
            result[key] = value

    return flax.traverse_util.unflatten_dict(result, sep="/")


def main():
    parser = argparse.ArgumentParser(
        description="Merge checkpoints: backbone+image from one, action expert from the other."
    )
    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_droid/params",
        help="Path to params for image encoder + main backbone (e.g. pi05_droid).",
    )
    parser.add_argument(
        "--action_expert_checkpoint",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_base/params",
        help="Path to params for action expert + projection layers (e.g. pi05_base).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/pi05_base_droid_merged",
        help="Directory to write merged checkpoint (params will be in output_dir/params).",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Only merge in memory and print key counts; do not save.",
    )
    args = parser.parse_args()

    logger.info("Merge checkpoints: backbone=%s, action_expert=%s", args.backbone_checkpoint, args.action_expert_checkpoint)

    # Download to local cache first (same as weight_loaders); ensures GCS works and avoids
    # orbax "incomplete checkpoint" errors when commit_success.txt is missing.
    backbone_path = download.maybe_download(args.backbone_checkpoint, gs={"token": "anon"})
    _ensure_commit_success(backbone_path)
    logger.info("Loading backbone (image + main) params from %s", backbone_path)
    backbone_params = _model.restore_params(
        backbone_path,
        restore_type=np.ndarray,
    )

    action_expert_path = download.maybe_download(args.action_expert_checkpoint, gs={"token": "anon"})
    _ensure_commit_success(action_expert_path)
    logger.info("Loading action expert params from %s", action_expert_path)
    action_expert_params = _model.restore_params(
        action_expert_path,
        restore_type=np.ndarray,
    )

    merged = merge_params(backbone_params, action_expert_params)

    flat_merged = flax.traverse_util.flatten_dict(merged, sep="/")
    flat_backbone = flax.traverse_util.flatten_dict(backbone_params, sep="/")
    from_base = sum(1 for k in flat_merged if _is_action_expert_or_projection_key(k))
    logger.info(
        "Merged checkpoint: %s total keys, %s from action-expert checkpoint (base).",
        len(flat_merged),
        from_base,
    )

    if args.no_save:
        logger.info("--no_save: skipping save.")
        return

    out_path = pathlib.Path(args.output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    params_dir = out_path / "params"

    # PyTreeCheckpointer saves the given item; openpi restores with item {"params": ...}.
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(params_dir, args=ocp.args.PyTreeSave({"params": merged}))

    logger.info("Saved merged params to %s", params_dir)
    logger.info(
        "Use in training with: weight_loader=CheckpointWeightLoader(%r)",
        str(params_dir),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Merge failed: %s", e)
        raise
