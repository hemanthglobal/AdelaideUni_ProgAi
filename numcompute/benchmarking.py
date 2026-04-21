"""
benchmarking.py
===============

Lightweight micro-benchmark harness for comparing vectorised NumPy
implementations against pure-Python loop implementations.

Design notes
------------
- We call ``timeit.Timer`` under the hood to get a reliable timer and to
  disable garbage collection during the timed section (reduces noise).
- Each function is warmed up before timing (CPU caches, JIT effects in
  some NumPy builds, page faults).
- We report *median* over repeats rather than mean: median is robust to
  occasional slow runs from OS scheduling noise.
- We report *min-per-repeat* separately because for CPU-bound pure
  functions the minimum is the closest estimate of the "true" cost
  (slower runs had interference).

The ``Benchmark`` class stores results and can render a comparison table.
"""

from __future__ import annotations

import gc
import platform
import sys
import timeit
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Result of timing a single function."""
    name: str
    n: int                        # problem size
    median_s: float               # median time per call, seconds
    min_s: float                  # minimum time per call, seconds
    repeats: int                  # number of repeats
    inner_loops: int              # calls per repeat


# ---------------------------------------------------------------------------
# Core timing function
# ---------------------------------------------------------------------------

def time_callable(
    func: Callable[[], None],
    repeats: int = 7,
    inner_loops: Optional[int] = None,
    warmup: int = 1,
) -> Dict[str, float]:
    """
    Time a zero-argument callable and return summary statistics.

    Parameters
    ----------
    func : callable
        The thing to time. Wrap your real call in a lambda:
        ``lambda: my_sort(data)``.
    repeats : int, default=7
        Number of repeat measurements. Odd number gives a clean median.
    inner_loops : int, optional
        Calls per repeat. If None, uses ``timeit.Timer.autorange`` to pick
        a value such that one repeat takes at least 0.2s (reduces timer
        resolution error). Fast functions get many inner loops; slow
        functions get one.
    warmup : int, default=1
        Number of untimed warmup calls.

    Returns
    -------
    stats : dict
        Keys: 'median_s', 'min_s', 'repeats', 'inner_loops'.
    """
    # Warm up (caches, first-call overhead).
    for _ in range(warmup):
        func()

    timer = timeit.Timer(stmt=func)

    if inner_loops is None:
        # autorange picks a loop count s.t. total time >= ~0.2s.
        inner_loops, _ = timer.autorange()

    # Disable GC during timing; timeit does this by default via its own
    # mechanism, but we also suppress any collection we triggered in warmup.
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        raw = timer.repeat(repeat=repeats, number=inner_loops)
    finally:
        if gc_was_enabled:
            gc.enable()

    per_call = [t / inner_loops for t in raw]
    return {
        "median_s": float(np.median(per_call)),
        "min_s": float(np.min(per_call)),
        "repeats": repeats,
        "inner_loops": inner_loops,
    }


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

@dataclass
class Benchmark:
    """Collects and compares benchmark results."""
    results: List[BenchResult] = field(default_factory=list)

    def run(
        self,
        name: str,
        func: Callable[[], None],
        n: int,
        repeats: int = 7,
        inner_loops: Optional[int] = None,
    ) -> BenchResult:
        """Time ``func`` and append the result to the suite."""
        stats = time_callable(func, repeats=repeats, inner_loops=inner_loops)
        result = BenchResult(
            name=name,
            n=n,
            median_s=stats["median_s"],
            min_s=stats["min_s"],
            repeats=stats["repeats"],
            inner_loops=stats["inner_loops"],
        )
        self.results.append(result)
        return result

    def compare(
        self,
        baseline_name: str,
        candidate_name: str,
        n: Optional[int] = None,
    ) -> float:
        """
        Return ``baseline.median / candidate.median`` (the speedup of
        candidate over baseline). Optionally filter by problem size ``n``.
        """
        b = self._find(baseline_name, n)
        c = self._find(candidate_name, n)
        return b.median_s / c.median_s

    def _find(self, name: str, n: Optional[int]) -> BenchResult:
        matches = [r for r in self.results if r.name == name
                   and (n is None or r.n == n)]
        if not matches:
            raise KeyError(f"No result for name={name!r}, n={n}.")
        return matches[-1]

    # -- rendering ---------------------------------------------------------

    def to_markdown(self) -> str:
        """Render results as a Markdown table (paste into README)."""
        if not self.results:
            return "_(no benchmark results)_"
        header = "| Name | n | median (ms) | min (ms) | repeats |"
        sep = "|------|---|-------------|----------|---------|"
        rows = [
            f"| {r.name} | {r.n} | {r.median_s * 1e3:.3f} | "
            f"{r.min_s * 1e3:.3f} | {r.repeats} |"
            for r in self.results
        ]
        return "\n".join([header, sep, *rows])

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [f"{'Name':<30} {'n':>8} {'median (ms)':>14} {'min (ms)':>12}"]
        lines.append("-" * len(lines[0]))
        for r in self.results:
            lines.append(
                f"{r.name:<30} {r.n:>8} {r.median_s * 1e3:>14.3f} "
                f"{r.min_s * 1e3:>12.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Environment reporting (for reproducibility)
# ---------------------------------------------------------------------------

def env_report() -> str:
    """
    Return a multi-line string describing the benchmark environment.

    Include this in benchmark output so results are reproducible and
    interpretable across machines.
    """
    return "\n".join([
        f"Python:   {sys.version.split()[0]}",
        f"NumPy:    {np.__version__}",
        f"Platform: {platform.platform()}",
        f"Machine:  {platform.machine()}",
        f"Processor:{platform.processor() or 'unknown'}",
    ])


# ---------------------------------------------------------------------------
# Example loop-vs-vectorised baselines (useful in the demo notebook)
# ---------------------------------------------------------------------------

def python_sum(values) -> float:
    """Plain Python sum — loop baseline for benchmarking."""
    total = 0.0
    for v in values:
        total += v
    return total


def python_dot(a, b) -> float:
    """Plain Python dot product — loop baseline."""
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def python_standardize(X) -> list:
    """
    Plain Python z-score standardisation (per-column) — loop baseline.
    Returns a list of lists. Intentionally un-optimised.
    """
    n_rows = len(X)
    n_cols = len(X[0]) if n_rows > 0 else 0
    # means
    means = [0.0] * n_cols
    for row in X:
        for j, v in enumerate(row):
            means[j] += v
    for j in range(n_cols):
        means[j] /= n_rows
    # stds
    stds = [0.0] * n_cols
    for row in X:
        for j, v in enumerate(row):
            stds[j] += (v - means[j]) ** 2
    for j in range(n_cols):
        stds[j] = (stds[j] / n_rows) ** 0.5
    # standardize
    out = []
    for row in X:
        out.append([(row[j] - means[j]) / stds[j] if stds[j] > 0 else 0.0
                    for j in range(n_cols)])
    return out


def numpy_standardize(X) -> np.ndarray:
    """Vectorised z-score standardisation — what the benchmark proves out."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return (X - mu) / sigma
