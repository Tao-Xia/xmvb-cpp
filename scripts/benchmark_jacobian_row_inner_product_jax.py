#!/usr/bin/env python3
"""Benchmarks dense and low-rank row-inner-product contractions in JAX.

This script compares the Frobenius inner product of a dense Jacobian-like
matrix against the same quantity computed from a rank-truncated randomized SVD
factorization. Each benchmark runs in its own subprocess so the device-memory
statistics are method-specific.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = 512
DEFAULT_COLS = 3000
DEFAULT_RANK = 32
DEFAULT_ITERATIONS = 1000
DEFAULT_SEED = 0
DEFAULT_RSVD_OVERSAMPLE = 8
DEFAULT_RSVD_POWER_ITERS = 1
DEFAULT_POLL_INTERVAL_SECONDS = 1.0e-3


@dataclass(frozen=True)
class BenchmarkConfig:
  rows: int = DEFAULT_ROWS
  cols: int = DEFAULT_COLS
  rank: int = DEFAULT_RANK
  iterations: int = DEFAULT_ITERATIONS
  seed: int = DEFAULT_SEED
  dtype: str = "float32"
  rsvd_oversample: int = DEFAULT_RSVD_OVERSAMPLE
  rsvd_power_iters: int = DEFAULT_RSVD_POWER_ITERS
  poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
  worker_launcher: str = "python"
  proton_backend: str = "instrumentation"


@dataclass(frozen=True)
class BenchmarkResult:
  method: str
  backend: str
  device: str
  value: float
  relative_error: float | None
  iterations: int
  elapsed_seconds: float
  average_iteration_seconds: float
  setup_seconds: float
  peak_bytes_in_use: int | None
  baseline_bytes_in_use: int | None
  peak_reserved_bytes: int | None
  matrix_shape: tuple[int, int]
  rank: int
  dtype: str


class PeakMemorySampler:
  """Polls device memory while a benchmark loop is running."""

  def __init__(self, probe, poll_interval_seconds: float):
    self._probe = probe
    self._poll_interval_seconds = poll_interval_seconds
    self._stop_event = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)
    self.peak_bytes_in_use: int | None = None

  def _run(self) -> None:
    while not self._stop_event.is_set():
      try:
        sample = self._probe()
      except Exception:
        sample = None
      if sample is not None:
        if self.peak_bytes_in_use is None or sample > self.peak_bytes_in_use:
          self.peak_bytes_in_use = sample
      time.sleep(self._poll_interval_seconds)

  def __enter__(self) -> "PeakMemorySampler":
    self._thread.start()
    return self

  def __exit__(self, exc_type, exc, exc_tb) -> None:
    self._stop_event.set()
    self._thread.join()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--worker", action="store_true", help="Run one method and emit JSON.")
  parser.add_argument(
      "--method",
      choices=("dense", "low-rank"),
      help="Benchmark method used by a worker subprocess.",
  )
  parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
  parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
  parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
  parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
  parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
  parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
  parser.add_argument("--rsvd-oversample", type=int, default=DEFAULT_RSVD_OVERSAMPLE)
  parser.add_argument("--rsvd-power-iters", type=int, default=DEFAULT_RSVD_POWER_ITERS)
  parser.add_argument(
      "--poll-interval-seconds",
      type=float,
      default=DEFAULT_POLL_INTERVAL_SECONDS,
  )
  parser.add_argument(
      "--worker-launcher",
      choices=("python", "proton"),
      default="python",
      help="Launch worker subprocesses directly with Python or through proton.",
  )
  parser.add_argument(
      "--proton-backend",
      choices=("cupti", "roctracer", "instrumentation"),
      default="instrumentation",
      help="Profiler backend used when --worker-launcher=proton.",
  )
  return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
  return BenchmarkConfig(
      rows=args.rows,
      cols=args.cols,
      rank=args.rank,
      iterations=args.iterations,
      seed=args.seed,
      dtype=args.dtype,
      rsvd_oversample=args.rsvd_oversample,
      rsvd_power_iters=args.rsvd_power_iters,
      poll_interval_seconds=args.poll_interval_seconds,
      worker_launcher=args.worker_launcher,
      proton_backend=args.proton_backend,
  )


def configure_worker_environment() -> None:
  """Applies runtime settings before importing JAX."""

  os.environ.setdefault("JAX_PLATFORMS", "cuda")
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  xla_flags = os.environ.get("XLA_FLAGS", "").strip()
  disable_triton = "--xla_gpu_enable_triton_gemm=false"
  if disable_triton not in xla_flags.split():
    os.environ["XLA_FLAGS"] = (
        disable_triton if not xla_flags else f"{disable_triton} {xla_flags}"
    )


def make_python_executable() -> str:
  explicit_python = PROJECT_ROOT / ".venv" / "bin" / "python"
  if explicit_python.exists():
    return str(explicit_python)
  return sys.executable


def make_proton_executable() -> str:
  proton = PROJECT_ROOT / ".venv" / "bin" / "proton"
  if not proton.exists():
    raise FileNotFoundError(f"proton executable not found at {proton}")
  return str(proton)


def run_worker(config: BenchmarkConfig, method: str) -> BenchmarkResult:
  worker_args = [
      str(Path(__file__).resolve()),
      "--worker",
      "--method",
      method,
      "--rows",
      str(config.rows),
      "--cols",
      str(config.cols),
      "--rank",
      str(config.rank),
      "--iterations",
      str(config.iterations),
      "--seed",
      str(config.seed),
      "--dtype",
      config.dtype,
      "--rsvd-oversample",
      str(config.rsvd_oversample),
      "--rsvd-power-iters",
      str(config.rsvd_power_iters),
      "--poll-interval-seconds",
      str(config.poll_interval_seconds),
  ]
  if config.worker_launcher == "proton":
    command = [
        make_proton_executable(),
        "-b",
        config.proton_backend,
        "-n",
        f"jacobian_{method}",
        *worker_args,
    ]
  else:
    command = [make_python_executable(), *worker_args]
  try:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
  except subprocess.CalledProcessError as exc:
    raise RuntimeError(
        f"worker for {method} failed with command: {' '.join(command)}\n"
        f"stdout:\n{exc.stdout}\n"
        f"stderr:\n{exc.stderr}"
    ) from exc
  lines = [line for line in completed.stdout.splitlines() if line.strip()]
  if not lines:
    raise RuntimeError(f"worker for {method} did not produce JSON output")
  payload = json.loads(lines[-1])
  return BenchmarkResult(**payload)


def worker_main(config: BenchmarkConfig, method: str) -> None:
  configure_worker_environment()

  import jax
  import jax.numpy as jnp
  import numpy as np

  if config.dtype == "float64":
    jax.config.update("jax_enable_x64", True)

  dtype = getattr(jnp, config.dtype)
  device = jax.devices()[0]

  def randomized_svd(matrix, key, rank):
    """Computes a rank-truncated randomized SVD without `jnp.linalg.svd`.

    The randomized range finder builds a small orthonormal basis `Q` for the
    dominant column space of `matrix`. We then diagonalize the reduced Gram
    matrix `(Q^T matrix) (Q^T matrix)^T` so the only eigenproblem is
    `(rank + oversample) x (rank + oversample)`.
    """

    target_rank = min(rank, matrix.shape[0], matrix.shape[1])
    sketch_rank = min(
        target_rank + config.rsvd_oversample,
        matrix.shape[0],
        matrix.shape[1],
    )
    omega = jax.random.normal(key, (matrix.shape[1], sketch_rank), dtype=matrix.dtype)

    # Power iterations sharpen the dominant subspace when the spectrum decays
    # slowly. All intermediates follow the standard `(rows, sketch_rank)` and
    # `(cols, sketch_rank)` conventions from randomized linear algebra.
    sample = matrix @ omega
    for _ in range(config.rsvd_power_iters):
      sample = matrix @ (matrix.T @ sample)

    q_basis, _ = jnp.linalg.qr(sample, mode="reduced")
    reduced_matrix = q_basis.T @ matrix
    reduced_gram = reduced_matrix @ reduced_matrix.T

    eigenvalues, left_vectors_small = jnp.linalg.eigh(reduced_gram)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    left_vectors_small = left_vectors_small[:, order]

    singular_values = jnp.sqrt(jnp.clip(eigenvalues[:target_rank], a_min=0.0))
    left_vectors = q_basis @ left_vectors_small[:, :target_rank]

    # Recover the right factor from `B = U^T matrix`. The safe divide avoids
    # NaNs when the truncated spectrum contains tiny or zero singular values.
    reduced_rows = left_vectors.T @ matrix
    scale = jnp.where(singular_values > 0.0, singular_values, 1.0)
    right_vectors_t = reduced_rows / scale[:, None]

    # Returning `A = U` and `B = Sigma V^T` keeps the low-rank inner-product
    # contraction faithful to the requested `A @ (B @ B.T)` form.
    return left_vectors, singular_values[:, None] * right_vectors_t

  @jax.jit
  def dense_inner_product(matrix):
    return jnp.sum(matrix * matrix)

  @jax.jit
  def low_rank_inner_product(left_factor, right_factor):
    # `sum((A @ (B @ B.T)) * A)` is equivalent to
    # `sum((A.T @ A) * (B @ B.T))`, which avoids materializing the `(H, rank)`
    # intermediate after the matrix-matrix product.
    left_gram = left_factor.T @ left_factor
    right_gram = right_factor @ right_factor.T
    return jnp.sum(left_gram * right_gram)

  randomized_svd_jit = jax.jit(randomized_svd, static_argnames=("rank",))
  key = jax.random.key(config.seed)
  matrix_key, rsvd_key = jax.random.split(key)
  matrix = jax.random.normal(matrix_key, (config.rows, config.cols), dtype=dtype)

  def block_array(value):
    return jax.block_until_ready(value)

  def current_memory_stats() -> dict[str, int] | None:
    if not hasattr(device, "memory_stats"):
      return None
    stats = device.memory_stats()
    if not isinstance(stats, dict):
      return None
    return {
        key: int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }

  def current_bytes_in_use() -> int | None:
    stats = current_memory_stats()
    if stats is None:
      return None
    return stats.get("bytes_in_use")

  baseline_stats = current_memory_stats()
  setup_start = time.perf_counter()

  if method == "dense":
    reference_value = block_array(dense_inner_product(matrix))
    setup_seconds = time.perf_counter() - setup_start
    baseline_stats = current_memory_stats()

    with PeakMemorySampler(current_bytes_in_use, config.poll_interval_seconds) as sampler:
      timing_start = time.perf_counter()
      value = reference_value
      for _ in range(config.iterations):
        value = dense_inner_product(matrix)
        block_array(value)
      elapsed_seconds = time.perf_counter() - timing_start

    result = BenchmarkResult(
        method=method,
        backend=jax.default_backend(),
        device=str(device),
        value=float(value),
        relative_error=None,
        iterations=config.iterations,
        elapsed_seconds=elapsed_seconds,
        average_iteration_seconds=elapsed_seconds / config.iterations,
        setup_seconds=setup_seconds,
        peak_bytes_in_use=sampler.peak_bytes_in_use,
        baseline_bytes_in_use=(
            None if baseline_stats is None else baseline_stats.get("bytes_in_use")
        ),
        peak_reserved_bytes=(
            None
            if baseline_stats is None
            else baseline_stats.get("peak_bytes_reserved")
        ),
        matrix_shape=(config.rows, config.cols),
        rank=config.rank,
        dtype=config.dtype,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return

  factor_left, factor_right = randomized_svd_jit(matrix, rsvd_key, config.rank)
  factor_left = block_array(factor_left)
  factor_right = block_array(factor_right)

  warmup_value = low_rank_inner_product(factor_left, factor_right)
  warmup_value = block_array(warmup_value)
  setup_seconds = time.perf_counter() - setup_start
  baseline_stats = current_memory_stats()

  with PeakMemorySampler(current_bytes_in_use, config.poll_interval_seconds) as sampler:
    timing_start = time.perf_counter()
    value = warmup_value
    for _ in range(config.iterations):
      value = low_rank_inner_product(factor_left, factor_right)
      block_array(value)
    elapsed_seconds = time.perf_counter() - timing_start

  dense_reference = block_array(dense_inner_product(matrix))
  relative_error = abs(float(value) - float(dense_reference)) / max(
      abs(float(dense_reference)),
      sys.float_info.epsilon,
  )
  result = BenchmarkResult(
      method=method,
      backend=jax.default_backend(),
      device=str(device),
      value=float(value),
      relative_error=relative_error,
      iterations=config.iterations,
      elapsed_seconds=elapsed_seconds,
      average_iteration_seconds=elapsed_seconds / config.iterations,
      setup_seconds=setup_seconds,
      peak_bytes_in_use=sampler.peak_bytes_in_use,
      baseline_bytes_in_use=(
          None if baseline_stats is None else baseline_stats.get("bytes_in_use")
      ),
      peak_reserved_bytes=(
          None if baseline_stats is None else baseline_stats.get("peak_bytes_reserved")
      ),
      matrix_shape=(config.rows, config.cols),
      rank=config.rank,
      dtype=config.dtype,
  )
  print(json.dumps(asdict(result), sort_keys=True))


def format_mib(byte_count: int | None) -> str:
  if byte_count is None:
    return "unavailable"
  return f"{byte_count / (1024 ** 2):.2f} MiB"


def parent_main(config: BenchmarkConfig) -> None:
  dense_result = run_worker(config, "dense")
  low_rank_result = run_worker(config, "low-rank")

  relative_error = abs(low_rank_result.value - dense_result.value) / max(
      abs(dense_result.value),
      sys.float_info.epsilon,
  )

  print(
      "JAX row-inner-product benchmark\n"
      f"  matrix shape: {config.rows} x {config.cols}\n"
      f"  rank: {config.rank}\n"
      f"  iterations: {config.iterations}\n"
      f"  dtype: {config.dtype}\n"
      f"  worker launcher: {config.worker_launcher}\n"
      f"  backend: {dense_result.backend}\n"
      f"  device: {dense_result.device}\n"
  )
  print(
      "Dense method\n"
      f"  value: {dense_result.value:.8e}\n"
      f"  setup time: {dense_result.setup_seconds:.6f} s\n"
      f"  total time: {dense_result.elapsed_seconds:.6f} s\n"
      f"  average time: {dense_result.average_iteration_seconds * 1.0e6:.3f} us/iter\n"
      f"  baseline memory: {format_mib(dense_result.baseline_bytes_in_use)}\n"
      f"  peak memory: {format_mib(dense_result.peak_bytes_in_use)}\n"
  )
  print(
      "Low-rank method\n"
      f"  value: {low_rank_result.value:.8e}\n"
      f"  relative error vs dense: {relative_error:.8e}\n"
      f"  setup time (includes rSVD): {low_rank_result.setup_seconds:.6f} s\n"
      f"  total time: {low_rank_result.elapsed_seconds:.6f} s\n"
      f"  average time: {low_rank_result.average_iteration_seconds * 1.0e6:.3f} us/iter\n"
      f"  baseline memory: {format_mib(low_rank_result.baseline_bytes_in_use)}\n"
      f"  peak memory: {format_mib(low_rank_result.peak_bytes_in_use)}\n"
  )
  if low_rank_result.average_iteration_seconds > 0.0:
    speedup = (
        dense_result.average_iteration_seconds
        / low_rank_result.average_iteration_seconds
    )
    print(f"Speedup (dense / low-rank): {speedup:.3f}x")


def main() -> None:
  args = parse_args()
  config = build_config(args)
  if args.worker:
    if args.method is None:
      raise ValueError("--worker requires --method")
    worker_main(config, args.method)
    return
  parent_main(config)


if __name__ == "__main__":
  main()
