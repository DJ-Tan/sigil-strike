"""
gpu.py
──────
PyTorch device configuration for shared inference / training.

Unlike TensorFlow, PyTorch is cooperative by default — it only allocates
GPU memory as it needs it and releases it back to a managed cache when
tensors go out of scope.  Two processes (e.g., one inference.py per
player) can share a single GPU naturally with no extra setup.

`configure_gpu()` resolves the right `torch.device`, optionally caps
this process's GPU memory fraction, and reports the device choice.
Call this BEFORE building or loading any model.
"""

from __future__ import annotations

import os


def configure_gpu(
    memory_limit_mb: int | None = None,
    prefer_gpu: bool = True,
):
    """Pick the best available torch.device and apply memory settings.

    Args:
        memory_limit_mb : per-process hard cap in MB.  None → grow as needed.
        prefer_gpu      : if False, force CPU regardless of hardware.

    Returns:
        (device, status_string)
            device         : torch.device or None (if torch missing)
            status_string  : human-readable line for startup logs
    """
    try:
        import torch
    except ImportError:
        return None, "CPU (PyTorch not installed)"

    if not prefer_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return torch.device("cpu"), "CPU (forced via --no-gpu)"

    if not torch.cuda.is_available():
        return torch.device("cpu"), "CPU (no CUDA GPU detected)"

    device = torch.device("cuda:0")
    name   = torch.cuda.get_device_name(0)

    cap = ""
    if memory_limit_mb is not None and memory_limit_mb > 0:
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        fraction = min(memory_limit_mb / total_mb, 1.0)
        try:
            torch.cuda.set_per_process_memory_fraction(fraction, device=0)
            cap = f", capped at {memory_limit_mb} MB ({fraction:.0%} of {int(total_mb)} MB)"
        except Exception as e:
            cap = f" (memory cap failed: {e})"

    return device, f"GPU ({name}{cap})"
