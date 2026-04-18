"""YAML configuration loading for DeepVBH training."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .types import (
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    OutputConfig,
    RuntimeConfig,
    TrainingConfig,
    TrainingLoopConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "deepvbh" / "configs" / "train_jax.yaml"
VALID_OPTIMIZER_SCHEDULES = {"constant", "cosine_with_warmup"}
REMOVED_DATASET_KEYS = {
    "target_key",
    "step_selection",
    "max_steps_per_sample",
    "max_train_pairs_per_step",
    "upper_triangle_only",
    "exclude_diagonal",
}


class TrainingConfigLoader:
  """Loads YAML config into validated dataclasses."""

  default_mapping: dict[str, Any] = {
      "runtime": {
          "jax_platforms": "cuda",
          "disable_triton_gemm": True,
          "tf_cpp_min_log_level": "1",
          "xla_flags_append": [],
          "xla_python_client_preallocate": False,
          "xla_python_client_mem_fraction": None,
          "xla_python_client_allocator": None,
      },
      "dataset": {
          "data_roots": None,
          "batch_size": 64,
          "train_selected_steps": 5,
          "nonfinal_offdiagonal_pair_fraction": 0.5,
          "valid_fraction": 0.125,
          "test_fraction": 0.125,
          "split_seed": 0,
          "shuffle_seed": 0,
      },
      "model": {
          "d_model": 64,
          "num_heads": 4,
          "num_layers": 3,
          "mlp_hidden_dim": 256,
          "readout_hidden_dim": 64,
      },
      "optimizer": {
          "learning_rate": 1.0e-3,
          "weight_decay": 1.0e-4,
          "grad_clip_norm": 1.0,
          "schedule": "cosine_with_warmup",
          "warmup_epochs": 50,
          "cosine_final_learning_rate_scale": 0.0,
      },
      "training": {
          "epochs": 20,
          "seed": 0,
          "prefetch_batches": 2,
          "max_train_batches_per_epoch": None,
          "max_eval_batches": None,
          "console_every_n_epochs": 25,
      },
      "output": {
          "log_root": "deepvbh/logs",
          "run_name_prefix": "deepvbh_f2_c6h6_c6h6full_uppertri_hamiltonian",
          "checkpoint_filename": "checkpoint.pkl",
          "metrics_json_filename": "metrics.json",
          "csv_log_filename": "train_metrics.csv",
          "metrics_jsonl_filename": "train_metrics.jsonl",
          "text_log_filename": "train.log",
      },
  }

  def __init__(self, project_root: Path | None = None):
    self.project_root = PROJECT_ROOT if project_root is None else project_root

  def config_path(self) -> Path:
    override = os.environ.get("DEEPVBH_TRAIN_CONFIG")
    if override:
      return Path(override).expanduser().resolve()
    return DEFAULT_CONFIG_PATH.resolve()

  def merge_mapping(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
      if isinstance(value, dict) and isinstance(base.get(key), dict):
        self.merge_mapping(base[key], value)
      else:
        base[key] = value
    return base

  def resolve_path(self, value: str | None) -> Path | None:
    if value is None:
      return None
    path = Path(value).expanduser()
    if path.is_absolute():
      return path.resolve()
    return (self.project_root / path).resolve()

  def resolve_path_list(self, values: list[str]) -> tuple[Path, ...]:
    return tuple(self.resolve_path(value) for value in values if value is not None)

  def require_mapping(self, config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict):
      raise ValueError(f"config section {name!r} must be a mapping")
    return section

  def check_bool(self, value: Any, name: str) -> bool:
    if not isinstance(value, bool):
      raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value

  def check_int(
      self,
      value: Any,
      name: str,
      *,
      minimum: int = 0,
      allow_none: bool = False,
  ) -> int | None:
    if value is None and allow_none:
      return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
      comparator = "positive integer" if minimum == 1 else f"integer >= {minimum}"
      raise ValueError(f"{name} must be a {comparator}, got {value!r}")
    return value

  def check_float(
      self,
      value: Any,
      name: str,
      *,
      minimum: float | None = None,
      maximum: float | None = None,
      allow_none: bool = False,
  ) -> float | None:
    if value is None and allow_none:
      return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"{name} must be a float, got {value!r}")
    parsed = float(value)
    if minimum is not None and parsed < minimum:
      raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and parsed >= maximum:
      raise ValueError(f"{name} must be < {maximum}, got {value!r}")
    return parsed

  def check_choice(self, value: Any, name: str, choices: set[str]) -> str:
    if value not in choices:
      allowed = ", ".join(sorted(choices))
      raise ValueError(f"{name} must be one of {{{allowed}}}, got {value!r}")
    return str(value)

  def load_runtime(self, mapping: dict[str, Any]) -> RuntimeConfig:
    jax_platforms = mapping.get("jax_platforms")
    if jax_platforms is not None and not isinstance(jax_platforms, str):
      raise ValueError("runtime.jax_platforms must be a string or null")
    tf_cpp_min_log_level = mapping.get("tf_cpp_min_log_level")
    if tf_cpp_min_log_level is not None and not isinstance(tf_cpp_min_log_level, str):
      raise ValueError("runtime.tf_cpp_min_log_level must be a string or null")
    xla_flags_append = mapping.get("xla_flags_append", [])
    if not isinstance(xla_flags_append, list):
      raise ValueError("runtime.xla_flags_append must be a list")
    for flag in xla_flags_append:
      if not isinstance(flag, str):
        raise ValueError("runtime.xla_flags_append must contain only strings")
    allocator = mapping.get("xla_python_client_allocator")
    if allocator is not None and not isinstance(allocator, str):
      raise ValueError("runtime.xla_python_client_allocator must be a string or null")

    preallocate = mapping.get("xla_python_client_preallocate")
    if preallocate is not None:
      preallocate = self.check_bool(
          preallocate,
          "runtime.xla_python_client_preallocate",
      )

    return RuntimeConfig(
        jax_platforms=jax_platforms,
        disable_triton_gemm=self.check_bool(
            mapping["disable_triton_gemm"],
            "runtime.disable_triton_gemm",
        ),
        tf_cpp_min_log_level=tf_cpp_min_log_level,
        xla_flags_append=tuple(xla_flags_append),
        xla_python_client_preallocate=preallocate,
        xla_python_client_mem_fraction=self.check_float(
            mapping.get("xla_python_client_mem_fraction"),
            "runtime.xla_python_client_mem_fraction",
            minimum=0.0,
            maximum=1.0,
            allow_none=True,
        ),
        xla_python_client_allocator=allocator,
    )

  def load_dataset(self, mapping: dict[str, Any]) -> DatasetConfig:
    present_removed_keys = sorted(key for key in REMOVED_DATASET_KEYS if key in mapping)
    if present_removed_keys:
      removed_list = ", ".join(f"dataset.{key}" for key in present_removed_keys)
      raise ValueError(
          f"{removed_list} is no longer supported. "
          "Pairwise training uses compressed train trajectories and "
          "final-step evaluation."
      )

    dataset_roots = mapping.get("data_roots")
    dataset_root = mapping.get("data_root")
    if dataset_roots is not None and dataset_root is not None:
      raise ValueError("specify only one of dataset.data_roots or dataset.data_root")
    if dataset_roots is None and dataset_root is None:
      raise ValueError("dataset config must define dataset.data_roots or dataset.data_root")
    if dataset_roots is not None:
      if not isinstance(dataset_roots, list) or not dataset_roots:
        raise ValueError("dataset.data_roots must be a non-empty list of paths")
      for root in dataset_roots:
        if not isinstance(root, str):
          raise ValueError("dataset.data_roots must contain only string paths")
      data_roots = self.resolve_path_list(dataset_roots)
    else:
      if not isinstance(dataset_root, str):
        raise ValueError("dataset.data_root must be a string path")
      data_roots = (self.resolve_path(dataset_root),)

    valid_fraction = self.check_float(
        mapping["valid_fraction"],
        "dataset.valid_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    test_fraction = self.check_float(
        mapping["test_fraction"],
        "dataset.test_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if valid_fraction + test_fraction >= 1.0:
      raise ValueError("dataset.valid_fraction + dataset.test_fraction must be less than 1")

    return DatasetConfig(
        data_roots=data_roots,
        batch_size=self.check_int(mapping["batch_size"], "dataset.batch_size", minimum=1),
        train_selected_steps=self.check_int(
            mapping["train_selected_steps"],
            "dataset.train_selected_steps",
            minimum=1,
        ),
        nonfinal_offdiagonal_pair_fraction=self.check_float(
            mapping["nonfinal_offdiagonal_pair_fraction"],
            "dataset.nonfinal_offdiagonal_pair_fraction",
            minimum=0.0,
            maximum=1.0000001,
        ),
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
        split_seed=self.check_int(mapping["split_seed"], "dataset.split_seed", minimum=0),
        shuffle_seed=self.check_int(
            mapping["shuffle_seed"],
            "dataset.shuffle_seed",
            minimum=0,
        ),
    )

  def load_model(self, mapping: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        d_model=self.check_int(mapping["d_model"], "model.d_model", minimum=1),
        num_heads=self.check_int(mapping["num_heads"], "model.num_heads", minimum=1),
        num_layers=self.check_int(mapping["num_layers"], "model.num_layers", minimum=1),
        mlp_hidden_dim=self.check_int(
            mapping["mlp_hidden_dim"],
            "model.mlp_hidden_dim",
            minimum=1,
        ),
        readout_hidden_dim=self.check_int(
            mapping["readout_hidden_dim"],
            "model.readout_hidden_dim",
            minimum=1,
        ),
    )

  def load_optimizer(self, mapping: dict[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=self.check_float(
            mapping["learning_rate"],
            "optimizer.learning_rate",
            minimum=0.0,
        ),
        weight_decay=self.check_float(
            mapping["weight_decay"],
            "optimizer.weight_decay",
        ),
        grad_clip_norm=self.check_float(
            mapping["grad_clip_norm"],
            "optimizer.grad_clip_norm",
            minimum=0.0,
            allow_none=True,
        ),
        schedule=self.check_choice(
            mapping.get("schedule", "constant"),
            "optimizer.schedule",
            VALID_OPTIMIZER_SCHEDULES,
        ),
        warmup_epochs=self.check_int(
            mapping.get("warmup_epochs", 0),
            "optimizer.warmup_epochs",
            minimum=0,
        ),
        cosine_final_learning_rate_scale=self.check_float(
            mapping.get("cosine_final_learning_rate_scale", 0.0),
            "optimizer.cosine_final_learning_rate_scale",
            minimum=0.0,
            maximum=1.0,
        ),
    )

  def load_training(self, mapping: dict[str, Any]) -> TrainingLoopConfig:
    return TrainingLoopConfig(
        epochs=self.check_int(mapping["epochs"], "training.epochs", minimum=1),
        seed=self.check_int(mapping["seed"], "training.seed", minimum=0),
        prefetch_batches=self.check_int(
            mapping.get("prefetch_batches", 2),
            "training.prefetch_batches",
            minimum=0,
        ),
        max_train_batches_per_epoch=self.check_int(
            mapping["max_train_batches_per_epoch"],
            "training.max_train_batches_per_epoch",
            minimum=1,
            allow_none=True,
        ),
        max_eval_batches=self.check_int(
            mapping["max_eval_batches"],
            "training.max_eval_batches",
            minimum=1,
            allow_none=True,
        ),
        console_every_n_epochs=self.check_int(
            mapping["console_every_n_epochs"],
            "training.console_every_n_epochs",
            minimum=1,
        ),
    )

  def load_output(self, mapping: dict[str, Any]) -> OutputConfig:
    if mapping.get("log_root") is not None and not isinstance(mapping["log_root"], str):
      raise ValueError("output.log_root must be a string path or null")
    if mapping.get("run_name_prefix") is not None and not isinstance(
        mapping["run_name_prefix"], str
    ):
      raise ValueError("output.run_name_prefix must be a string or null")
    for key in (
        "checkpoint_filename",
        "metrics_json_filename",
        "csv_log_filename",
        "metrics_jsonl_filename",
        "text_log_filename",
    ):
      if mapping.get(key) is not None and not isinstance(mapping[key], str):
        raise ValueError(f"output.{key} must be a string filename or null")
    for key in ("checkpoint", "metrics_json", "csv_log", "metrics_jsonl", "text_log"):
      if mapping.get(key) is not None and not isinstance(mapping[key], str):
        raise ValueError(f"output.{key} must be a string path or null")

    output = OutputConfig(
        log_root=self.resolve_path(mapping.get("log_root")),
        run_name_prefix=mapping.get("run_name_prefix"),
        checkpoint_filename=mapping.get("checkpoint_filename"),
        metrics_json_filename=mapping.get("metrics_json_filename"),
        csv_log_filename=mapping.get("csv_log_filename"),
        metrics_jsonl_filename=mapping.get("metrics_jsonl_filename"),
        text_log_filename=mapping.get("text_log_filename"),
        checkpoint=self.resolve_path(mapping.get("checkpoint")),
        metrics_json=self.resolve_path(mapping.get("metrics_json")),
        csv_log=self.resolve_path(mapping.get("csv_log")),
        metrics_jsonl=self.resolve_path(mapping.get("metrics_jsonl")),
        text_log=self.resolve_path(mapping.get("text_log")),
    )
    output.normalize_run_name_prefix()
    return output

  def load(self, path: Path | None = None) -> TrainingConfig:
    path = self.config_path() if path is None else path.expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"training config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
      raise ValueError("top-level YAML config must be a mapping")

    merged = copy.deepcopy(self.default_mapping)
    self.merge_mapping(merged, raw)

    return TrainingConfig(
        config_path=path,
        runtime=self.load_runtime(self.require_mapping(merged, "runtime")),
        dataset=self.load_dataset(self.require_mapping(merged, "dataset")),
        model=self.load_model(self.require_mapping(merged, "model")),
        optimizer=self.load_optimizer(self.require_mapping(merged, "optimizer")),
        training=self.load_training(self.require_mapping(merged, "training")),
        output=self.load_output(self.require_mapping(merged, "output")),
    )
