#!/usr/bin/env python3
"""Benchmark inference latency for OpenPI Pi-DROID.

Compares batch size 32 vs 320: reports mean inference time (ms) and GPU memory (GB).
"""

import copy
import statistics
import time

import numpy as np

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

CONFIG = "pi05_droid"
CHECKPOINT = None  # None = use default gs:// for CONFIG
NUM_WARMUP = 5
NUM_ITERS = 50
USE_CARTESIAN_STATE = False
BATCH_SIZES = (32, 320)


def _stack_batch(transformed_single: dict, batch_size: int) -> dict:
    """Stack a single transformed example into a batch (numpy)."""
    batch = {}
    for k, v in transformed_single.items():
        if isinstance(v, dict):
            batch[k] = {
                k2: np.stack([np.asarray(v[k2])] * batch_size, axis=0)
                for k2 in v
            }
        else:
            arr = np.asarray(v)
            if arr.ndim == 0:
                batch[k] = np.broadcast_to(arr, (batch_size,))
            else:
                batch[k] = np.stack([arr] * batch_size, axis=0)
    return batch


def _get_transformed_single(policy, example: dict) -> dict:
    """Run input transform on a single example (mutates copy of example)."""
    example_copy = {
        k: np.array(v).copy() if isinstance(v, np.ndarray) else v
        for k, v in example.items()
    }
    return policy._input_transform(example_copy)


def _run_batch_inference(policy, batch_np: dict, batch_size: int):
    """Run one batched forward pass using policy internals."""
    if policy._is_pytorch_model:
        import torch
        device = policy._pytorch_device
        batch = {
            k: (
                {k2: torch.from_numpy(np.asarray(v)).to(device) for k2, v in v.items()}
                if isinstance(v, dict)
                else torch.from_numpy(np.asarray(v)).to(device)
            )
            for k, v in batch_np.items()
        }
        observation = _model.Observation.from_dict(copy.deepcopy(batch))
        return policy._sample_actions(device, observation)
    else:
        import jax
        import jax.numpy as jnp
        batch_jax = jax.tree.map(jnp.asarray, copy.deepcopy(batch_np))
        observation = _model.Observation.from_dict(batch_jax)
        policy._rng, rng = jax.random.split(policy._rng)
        return policy._sample_actions(rng, observation)


def _gpu_memory_gb(*, prefer_jax: bool = True):
    """Return current GPU memory used in GB. Prefer JAX when prefer_jax=True (default for pi05_droid)."""
    if prefer_jax:
        try:
            import jax
            devs = jax.devices()
            if devs:
                stats = devs[0].memory_stats()
                if stats:
                    bytes_used = stats.get("bytes_in_use") or stats.get("peak_bytes_in_use") or 0
                    return bytes_used / 1e9
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    if not prefer_jax:
        try:
            import jax
            devs = jax.devices()
            if devs:
                stats = devs[0].memory_stats()
                if stats:
                    return (stats.get("bytes_in_use") or stats.get("peak_bytes_in_use") or 0) / 1e9
        except Exception:
            pass
    return 0.0


def _gpu_memory_max_gb(*, prefer_jax: bool = True):
    """Return peak GPU memory in GB. JAX: peak_bytes_in_use; PyTorch: max_memory_allocated."""
    if prefer_jax:
        try:
            import jax
            devs = jax.devices()
            if devs:
                stats = devs[0].memory_stats()
                if stats:
                    peak = stats.get("peak_bytes_in_use") or stats.get("bytes_in_use") or 0
                    return peak / 1e9
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return _gpu_memory_gb(prefer_jax=prefer_jax)


def main() -> None:
    train_config = _config.get_config(CONFIG)
    if CHECKPOINT is not None:
        checkpoint_dir = download.maybe_download(CHECKPOINT)
    else:
        default_uris = {
            "pi05_droid": "gs://openpi-assets/checkpoints/pi05_droid",
            "pi0_droid": "gs://openpi-assets/checkpoints/pi0_droid",
            "pi0_fast_droid": "gs://openpi-assets/checkpoints/pi0_fast_droid",
        }
        uri = default_uris.get(CONFIG, f"gs://openpi-assets/checkpoints/{CONFIG}")
        checkpoint_dir = download.maybe_download(uri)

    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)
    example = droid_policy.make_droid_example(use_cartesian_state=USE_CARTESIAN_STATE)

    for _ in range(NUM_WARMUP):
        policy.infer(example)

    # Single-sample baseline
    latencies_ms: list[float] = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        policy.infer(example)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    mean_single_ms = statistics.mean(latencies_ms)
    use_jax = not getattr(policy, "_is_pytorch_model", False)
    mem_single_gb = _gpu_memory_gb(prefer_jax=use_jax)

    # Batched inference: build batch from transformed single example
    transformed_single = _get_transformed_single(policy, example)

    backend = "JAX" if use_jax else "PyTorch"
    print("=" * 60)
    print(f"Backend: {backend}  |  GPU memory from {'JAX device.memory_stats()' if use_jax else 'torch.cuda'}")
    print("Single-sample (batch_size=1)")
    print(f"  Mean inference time: {mean_single_ms:.2f} ms")
    print(f"  GPU memory (after):  {mem_single_gb:.3f} GB")
    print()

    for batch_size in BATCH_SIZES:
        batch_np = _stack_batch(transformed_single, batch_size)
        for _ in range(min(2, NUM_WARMUP)):
            _run_batch_inference(policy, batch_np, batch_size)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        times_ms: list[float] = []
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            _run_batch_inference(policy, batch_np, batch_size)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_batch_ms = statistics.mean(times_ms)
        mem_gb = _gpu_memory_max_gb(prefer_jax=use_jax)
        per_sample_ms = mean_batch_ms / batch_size
        print(f"Batch size {batch_size}")
        print(f"  Mean inference time (total): {mean_batch_ms:.2f} ms")
        print(f"  Mean time per sample:         {per_sample_ms:.2f} ms")
        print(f"  GPU memory (peak/current):    {mem_gb:.3f} GB")
        print(f"  Throughput:                   {1000.0 / per_sample_ms:.1f} samples/s")
        print()
    print("=" * 60)


if __name__ == "__main__":
    main()
