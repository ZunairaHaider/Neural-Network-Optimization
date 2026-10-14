"""
runtime_analysis/timing_utils.py
---------------------------------
Lightweight timing and memory profiling utilities.

These are PASSIVE measurement tools only — they wrap code sections
with time.perf_counter() and optionally tracemalloc / CUDA memory.
They do not touch any pruning logic.

Usage
-----
    from runtime_analysis.timing_utils import Timer, MemoryProfiler

    with Timer() as t:
        ... your code ...
    print(t.elapsed_s)   # seconds as float

    with MemoryProfiler() as mp:
        ... your code ...
    print(mp.peak_mb)    # peak RAM delta in MB (tracemalloc)
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Optional

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ─────────────────────────────────────────────────────────────────────────────
# Wall-clock timer
# ─────────────────────────────────────────────────────────────────────────────

class Timer:
    """Context manager for wall-clock timing via time.perf_counter().

    Examples
    --------
    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed_s  # float seconds
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_s: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_s = time.perf_counter() - self._start


# ─────────────────────────────────────────────────────────────────────────────
# CPU memory profiler (tracemalloc)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryProfiler:
    """Context manager that measures peak RAM allocated during the block.

    Uses Python's built-in tracemalloc — no external dependencies.
    Returns peak delta in MB (memory allocated inside the block, not total).

    If tracemalloc is already running this profiler starts a new snapshot
    frame and restores the prior state on exit.

    Attributes
    ----------
    peak_mb : float
        Peak memory delta (MB) measured inside the block.
    enabled : bool
        Set to False to disable profiling (returns 0.0 for all fields).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.peak_mb: float = 0.0
        self._was_tracing: bool = False

    def __enter__(self) -> "MemoryProfiler":
        if self.enabled:
            self._was_tracing = tracemalloc.is_tracing()
            if not self._was_tracing:
                tracemalloc.start()
            tracemalloc.clear_traces()
        return self

    def __exit__(self, *_) -> None:
        if self.enabled:
            _, peak = tracemalloc.get_traced_memory()
            self.peak_mb = peak / (1024 * 1024)
            if not self._was_tracing:
                tracemalloc.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CUDA memory snapshot (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class CUDAMemoryProfiler:
    """Context manager for CUDA memory delta measurement.

    Only active when torch is available and a CUDA device is in use.
    Returns 0.0 silently when CUDA is not available.

    Attributes
    ----------
    peak_mb : float
        Peak CUDA memory allocated during the block (MB).
    enabled : bool
        True only when torch + CUDA is available.
    """

    def __init__(self, device=None) -> None:
        self.enabled = _HAS_TORCH and torch.cuda.is_available()
        self._device = device
        self.peak_mb: float = 0.0

    def __enter__(self) -> "CUDAMemoryProfiler":
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self._device)
        return self

    def __exit__(self, *_) -> None:
        if self.enabled:
            self.peak_mb = (
                torch.cuda.max_memory_allocated(self._device) / (1024 * 1024)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Combined snapshot dataclass — one per pruning step
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeRecord:
    """All timing/memory measurements for one (method, dataset, sparsity) step."""

    method: str = ""
    dataset: str = ""
    dim: int = 1
    sparsity_pct: float = 0.0

    # Timings (seconds)
    scoring_time_s: float = 0.0
    finetune_time_s: float = 0.0
    total_time_s: float = 0.0

    # Quality metrics (from existing pruning pipeline)
    final_test_mse: float = 0.0
    baseline_mse: float = 0.0
    mse_ratio: float = 0.0       # final_test_mse / baseline_mse
    param_count: int = 0
    inference_ms: float = 0.0

    # Memory (optional — 0.0 if not measured)
    cpu_peak_mb: float = 0.0
    cuda_peak_mb: float = 0.0

    # FLOPs estimate (optional)
    flops_estimate: float = 0.0  # multiply-adds for one forward pass

    def to_dict(self) -> dict:
        return {
            "method":           self.method,
            "dataset":          self.dataset,
            "dim":              self.dim,
            "sparsity_pct":     self.sparsity_pct,
            "scoring_time_s":   round(self.scoring_time_s, 4),
            "finetune_time_s":  round(self.finetune_time_s, 4),
            "total_time_s":     round(self.total_time_s, 4),
            "final_test_mse":   self.final_test_mse,
            "baseline_mse":     self.baseline_mse,
            "mse_ratio":        round(self.mse_ratio, 4),
            "param_count":      self.param_count,
            "inference_ms":     round(self.inference_ms, 4),
            "cpu_peak_mb":      round(self.cpu_peak_mb, 3),
            "cuda_peak_mb":     round(self.cuda_peak_mb, 3),
            "flops_estimate":   self.flops_estimate,
        }
