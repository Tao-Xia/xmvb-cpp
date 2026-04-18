"""Training dataclasses and shared helpers for DeepVBH."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import os
from pathlib import Path
import sys
from typing import Any


def json_ready(value: Any) -> Any:
  """Converts `Path` and tuple values into JSON-friendly structures."""

  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {key: json_ready(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_ready(item) for item in value]
  return value


@dataclass
class RuntimeConfig:
  """Environment settings that must be applied before importing JAX."""

  jax_platforms: str | None = "cuda"
  disable_triton_gemm: bool = True
  tf_cpp_min_log_level: str | None = "1"
  xla_flags_append: tuple[str, ...] = field(default_factory=tuple)
  xla_python_client_preallocate: bool | None = False
  xla_python_client_mem_fraction: float | None = None
  xla_python_client_allocator: str | None = None

  def append_xla_flags(self, flags: tuple[str, ...]) -> None:
    existing = os.environ.get("XLA_FLAGS", "").strip()
    merged = [flag for flag in flags if flag]
    if existing:
      merged.append(existing)
    if merged:
      os.environ["XLA_FLAGS"] = " ".join(merged)

  def apply_environment(self) -> None:
    """Applies runtime environment variables before JAX is imported."""

    if self.jax_platforms:
      os.environ["JAX_PLATFORMS"] = self.jax_platforms
    if self.tf_cpp_min_log_level:
      os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.tf_cpp_min_log_level
    if self.xla_python_client_preallocate is not None:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
          "true" if self.xla_python_client_preallocate else "false"
      )
    if self.xla_python_client_mem_fraction is not None:
      os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
          self.xla_python_client_mem_fraction
      )
    if self.xla_python_client_allocator:
      os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = self.xla_python_client_allocator

    xla_flags = list(self.xla_flags_append)
    if self.disable_triton_gemm:
      xla_flags.insert(0, "--xla_gpu_enable_triton_gemm=false")
    self.append_xla_flags(tuple(xla_flags))

  @staticmethod
  def sanitize_sys_path(project_root: Path) -> None:
    """Avoids local shadow files such as `jax.py` hijacking imports."""

    project_root = project_root.resolve()
    if not (project_root / "jax.py").exists():
      return

    cleaned_sys_path: list[str] = []
    for entry in sys.path:
      if entry in {"", "."}:
        continue
      try:
        resolved = Path(entry).expanduser().resolve()
      except OSError:
        cleaned_sys_path.append(entry)
        continue
      if resolved == project_root:
        continue
      cleaned_sys_path.append(entry)
    sys.path[:] = cleaned_sys_path


@dataclass
class DatasetConfig:
  """Dataset settings for compressed-train and final-eval `H_{IJ}` regression."""

  data_roots: tuple[Path, ...]
  batch_size: int = 64
  train_selected_steps: int = 5
  nonfinal_offdiagonal_pair_fraction: float = 0.5
  valid_fraction: float = 0.125
  test_fraction: float = 0.125
  split_seed: int = 0
  shuffle_seed: int = 0


@dataclass
class ModelConfig:
  """Hyperparameters for the pair-biased transformer."""

  d_model: int = 64
  num_heads: int = 4
  num_layers: int = 3
  mlp_hidden_dim: int = 256
  readout_hidden_dim: int = 64


@dataclass
class OptimizerConfig:
  """Optimizer and learning-rate schedule settings."""

  learning_rate: float = 1.0e-3
  weight_decay: float = 1.0e-4
  grad_clip_norm: float | None = 1.0
  schedule: str = "cosine_with_warmup"
  warmup_epochs: int = 50
  cosine_final_learning_rate_scale: float = 0.0


@dataclass
class TrainingLoopConfig:
  """Training loop limits and reproducibility settings."""

  epochs: int = 20
  seed: int = 0
  prefetch_batches: int = 2
  max_train_batches_per_epoch: int | None = None
  max_eval_batches: int | None = None
  console_every_n_epochs: int = 25


@dataclass
class OutputConfig:
  """Output path settings for checkpoints and metric logs."""

  log_root: Path | None = None
  run_name_prefix: str | None = None
  checkpoint_filename: str | None = "checkpoint.pkl"
  metrics_json_filename: str | None = "metrics.json"
  csv_log_filename: str | None = "train_metrics.csv"
  metrics_jsonl_filename: str | None = "train_metrics.jsonl"
  text_log_filename: str | None = "train.log"
  checkpoint: Path | None = None
  metrics_json: Path | None = None
  csv_log: Path | None = None
  metrics_jsonl: Path | None = None
  text_log: Path | None = None
  run_dir: Path | None = None

  def normalize_run_name_prefix(self) -> None:
    if self.run_name_prefix is None:
      return
    normalized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in self.run_name_prefix.strip()
    ).strip("-_")
    self.run_name_prefix = normalized or None

  def make_run_dir(self) -> Path:
    if self.log_root is None:
      raise ValueError("log_root must be set before creating a run directory")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        timestamp
        if self.run_name_prefix is None
        else f"{self.run_name_prefix}_{timestamp}"
    )
    run_dir = self.log_root / base_name
    suffix = 1
    while run_dir.exists():
      run_dir = self.log_root / f"{base_name}_{suffix:02d}"
      suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

  def prepare_paths(self) -> None:
    """Materializes derived output paths once the run is about to start."""

    if (
        self.checkpoint is not None
        or self.metrics_json is not None
        or self.csv_log is not None
        or self.metrics_jsonl is not None
        or self.text_log is not None
    ):
      self.run_dir = None
      return

    if self.log_root is None:
      self.run_dir = None
      self.checkpoint = None
      self.metrics_json = None
      self.csv_log = None
      self.metrics_jsonl = None
      self.text_log = None
      return

    if not any(
        name is not None
        for name in (
            self.checkpoint_filename,
            self.metrics_json_filename,
            self.csv_log_filename,
            self.metrics_jsonl_filename,
            self.text_log_filename,
        )
    ):
      self.run_dir = None
      self.checkpoint = None
      self.metrics_json = None
      self.csv_log = None
      self.metrics_jsonl = None
      self.text_log = None
      return

    self.run_dir = self.make_run_dir()
    self.checkpoint = (
        None
        if self.checkpoint_filename is None
        else self.run_dir / self.checkpoint_filename
    )
    self.metrics_json = (
        None
        if self.metrics_json_filename is None
        else self.run_dir / self.metrics_json_filename
    )
    self.csv_log = (
        None if self.csv_log_filename is None else self.run_dir / self.csv_log_filename
    )
    self.metrics_jsonl = (
        None
        if self.metrics_jsonl_filename is None
        else self.run_dir / self.metrics_jsonl_filename
    )
    self.text_log = (
        None if self.text_log_filename is None else self.run_dir / self.text_log_filename
    )


@dataclass
class TrainingConfig:
  """Fully validated training configuration."""

  config_path: Path
  runtime: RuntimeConfig
  dataset: DatasetConfig
  model: ModelConfig
  optimizer: OptimizerConfig
  training: TrainingLoopConfig
  output: OutputConfig

  def as_serializable(self) -> dict[str, Any]:
    return json_ready(asdict(self))
