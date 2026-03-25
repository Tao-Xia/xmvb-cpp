"""Train the pair-biased transformer from a YAML configuration file."""

from __future__ import annotations

import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "train_jax.yaml"
VALID_TARGET_KEYS = {
    "target_hamiltonian",
    "target_overlap",
    "target_one_electron_hamiltonian",
}
VALID_STEP_SELECTIONS = {"all", "final", "initial"}
VALID_OPTIMIZER_SCHEDULES = {"constant", "cosine_with_warmup"}
DEFAULT_CONFIG: dict[str, Any] = {
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
        "target_key": "target_hamiltonian",
        "batch_size": 64,
        "step_selection": "all",
        "max_steps_per_sample": 1,
        "max_train_pairs_per_step": None,
        "valid_fraction": 0.125,
        "test_fraction": 0.125,
        "split_seed": 0,
        "shuffle_seed": 0,
        "upper_triangle_only": False,
        "exclude_diagonal": False,
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
    },
}


def _config_path() -> Path:
  override = os.environ.get("DEEPVBH_TRAIN_CONFIG")
  if override:
    return Path(override).expanduser().resolve()
  return DEFAULT_CONFIG_PATH.resolve()


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
  for key, value in override.items():
    if isinstance(value, dict) and isinstance(base.get(key), dict):
      _deep_update(base[key], value)
    else:
      base[key] = value
  return base


def _resolve_path(value: str | None) -> Path | None:
  if value is None:
    return None
  path = Path(value).expanduser()
  if path.is_absolute():
    return path.resolve()
  return (PROJECT_ROOT / path).resolve()


def _resolve_path_list(values: list[str]) -> tuple[Path, ...]:
  return tuple(_resolve_path(value) for value in values if value is not None)


def _normalize_run_name_prefix(value: str | None) -> str | None:
  if value is None:
    return None
  normalized = "".join(
      char if char.isalnum() or char in {"-", "_"} else "-"
      for char in value.strip()
  ).strip("-_")
  return normalized or None


def _make_run_dir(root: Path, prefix: str | None) -> Path:
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  base_name = timestamp if prefix is None else f"{prefix}_{timestamp}"
  run_dir = root / base_name
  suffix = 1
  while run_dir.exists():
    run_dir = root / f"{base_name}_{suffix:02d}"
    suffix += 1
  run_dir.mkdir(parents=True, exist_ok=False)
  return run_dir


def _prepare_output_paths(output: dict[str, Any]) -> None:
  legacy_checkpoint = output.get("checkpoint")
  legacy_metrics_json = output.get("metrics_json")
  legacy_csv_log = output.get("csv_log")
  if (
      legacy_checkpoint is not None
      or legacy_metrics_json is not None
      or legacy_csv_log is not None
  ):
    output["run_dir"] = None
    return

  log_root = output.get("log_root")
  checkpoint_filename = output.get("checkpoint_filename")
  metrics_json_filename = output.get("metrics_json_filename")
  csv_log_filename = output.get("csv_log_filename")
  if log_root is None:
    output["run_dir"] = None
    output["checkpoint"] = None
    output["metrics_json"] = None
    output["csv_log"] = None
    return

  if not any(
      name is not None
      for name in (checkpoint_filename, metrics_json_filename, csv_log_filename)
  ):
    output["run_dir"] = None
    output["checkpoint"] = None
    output["metrics_json"] = None
    output["csv_log"] = None
    return

  run_dir = _make_run_dir(log_root, output.get("run_name_prefix"))
  output["run_dir"] = run_dir
  output["checkpoint"] = (
      None if checkpoint_filename is None else run_dir / checkpoint_filename
  )
  output["metrics_json"] = (
      None if metrics_json_filename is None else run_dir / metrics_json_filename
  )
  output["csv_log"] = None if csv_log_filename is None else run_dir / csv_log_filename


def _json_ready(value: Any) -> Any:
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {key: _json_ready(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_ready(item) for item in value]
  return value


def _require_section(config: dict[str, Any], name: str) -> dict[str, Any]:
  section = config.get(name)
  if not isinstance(section, dict):
    raise ValueError(f"config section {name!r} must be a mapping")
  return section


def _check_bool(value: Any, name: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"{name} must be a boolean, got {value!r}")
  return value


def _check_int(value: Any, name: str, *, minimum: int = 0, allow_none: bool = False) -> int | None:
  if value is None and allow_none:
    return None
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    comparator = "positive integer" if minimum == 1 else f"integer >= {minimum}"
    raise ValueError(f"{name} must be a {comparator}, got {value!r}")
  return value


def _check_float(
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


def _check_choice(value: Any, name: str, choices: set[str]) -> str:
  if value not in choices:
    allowed = ", ".join(sorted(choices))
    raise ValueError(f"{name} must be one of {{{allowed}}}, got {value!r}")
  return str(value)


def _load_config(path: Path) -> dict[str, Any]:
  if not path.exists():
    raise FileNotFoundError(f"training config file not found: {path}")

  raw = yaml.safe_load(path.read_text()) or {}
  if not isinstance(raw, dict):
    raise ValueError("top-level YAML config must be a mapping")

  config = copy.deepcopy(DEFAULT_CONFIG)
  _deep_update(config, raw)
  config["config_path"] = path

  runtime = _require_section(config, "runtime")
  dataset = _require_section(config, "dataset")
  model = _require_section(config, "model")
  optimizer = _require_section(config, "optimizer")
  training = _require_section(config, "training")
  output = _require_section(config, "output")

  if runtime["jax_platforms"] is not None and not isinstance(runtime["jax_platforms"], str):
    raise ValueError("runtime.jax_platforms must be a string or null")
  runtime["disable_triton_gemm"] = _check_bool(
      runtime["disable_triton_gemm"],
      "runtime.disable_triton_gemm",
  )
  if runtime["tf_cpp_min_log_level"] is not None and not isinstance(
      runtime["tf_cpp_min_log_level"], str
  ):
    raise ValueError("runtime.tf_cpp_min_log_level must be a string or null")
  if not isinstance(runtime["xla_flags_append"], list):
    raise ValueError("runtime.xla_flags_append must be a list")
  for flag in runtime["xla_flags_append"]:
    if not isinstance(flag, str):
      raise ValueError("runtime.xla_flags_append must contain only strings")
  if runtime.get("xla_python_client_preallocate") is not None:
    runtime["xla_python_client_preallocate"] = _check_bool(
        runtime["xla_python_client_preallocate"],
        "runtime.xla_python_client_preallocate",
    )
  runtime["xla_python_client_mem_fraction"] = _check_float(
      runtime.get("xla_python_client_mem_fraction"),
      "runtime.xla_python_client_mem_fraction",
      minimum=0.0,
      maximum=1.0,
      allow_none=True,
  )
  if runtime.get("xla_python_client_allocator") is not None and not isinstance(
      runtime["xla_python_client_allocator"], str
  ):
    raise ValueError("runtime.xla_python_client_allocator must be a string or null")

  dataset_roots = dataset.get("data_roots")
  dataset_root = dataset.get("data_root")
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
    dataset["data_roots"] = _resolve_path_list(dataset_roots)
  else:
    if not isinstance(dataset_root, str):
      raise ValueError("dataset.data_root must be a string path")
    dataset["data_roots"] = (_resolve_path(dataset_root),)
  dataset.pop("data_root", None)
  dataset["target_key"] = _check_choice(
      dataset["target_key"], "dataset.target_key", VALID_TARGET_KEYS
  )
  dataset["batch_size"] = _check_int(dataset["batch_size"], "dataset.batch_size", minimum=1)
  dataset["step_selection"] = _check_choice(
      dataset["step_selection"], "dataset.step_selection", VALID_STEP_SELECTIONS
  )
  dataset["max_steps_per_sample"] = _check_int(
      dataset["max_steps_per_sample"],
      "dataset.max_steps_per_sample",
      minimum=1,
      allow_none=True,
  )
  dataset["max_train_pairs_per_step"] = _check_int(
      dataset.get("max_train_pairs_per_step"),
      "dataset.max_train_pairs_per_step",
      minimum=1,
      allow_none=True,
  )
  dataset["valid_fraction"] = _check_float(
      dataset["valid_fraction"], "dataset.valid_fraction", minimum=0.0, maximum=1.0
  )
  dataset["test_fraction"] = _check_float(
      dataset["test_fraction"], "dataset.test_fraction", minimum=0.0, maximum=1.0
  )
  if dataset["valid_fraction"] + dataset["test_fraction"] >= 1.0:
    raise ValueError("dataset.valid_fraction + dataset.test_fraction must be less than 1")
  dataset["split_seed"] = _check_int(dataset["split_seed"], "dataset.split_seed", minimum=0)
  dataset["shuffle_seed"] = _check_int(
      dataset["shuffle_seed"], "dataset.shuffle_seed", minimum=0
  )
  dataset["upper_triangle_only"] = _check_bool(
      dataset["upper_triangle_only"], "dataset.upper_triangle_only"
  )
  dataset["exclude_diagonal"] = _check_bool(
      dataset["exclude_diagonal"], "dataset.exclude_diagonal"
  )

  model["d_model"] = _check_int(model["d_model"], "model.d_model", minimum=1)
  model["num_heads"] = _check_int(model["num_heads"], "model.num_heads", minimum=1)
  model["num_layers"] = _check_int(model["num_layers"], "model.num_layers", minimum=1)
  model["mlp_hidden_dim"] = _check_int(
      model["mlp_hidden_dim"], "model.mlp_hidden_dim", minimum=1
  )
  model["readout_hidden_dim"] = _check_int(
      model["readout_hidden_dim"], "model.readout_hidden_dim", minimum=1
  )

  optimizer["learning_rate"] = _check_float(
      optimizer["learning_rate"], "optimizer.learning_rate", minimum=0.0
  )
  optimizer["weight_decay"] = _check_float(
      optimizer["weight_decay"], "optimizer.weight_decay"
  )
  optimizer["grad_clip_norm"] = _check_float(
      optimizer["grad_clip_norm"],
      "optimizer.grad_clip_norm",
      minimum=0.0,
      allow_none=True,
  )
  optimizer["schedule"] = _check_choice(
      optimizer.get("schedule", "constant"),
      "optimizer.schedule",
      VALID_OPTIMIZER_SCHEDULES,
  )
  optimizer["warmup_epochs"] = _check_int(
      optimizer.get("warmup_epochs", 0),
      "optimizer.warmup_epochs",
      minimum=0,
  )
  optimizer["cosine_final_learning_rate_scale"] = _check_float(
      optimizer.get("cosine_final_learning_rate_scale", 0.0),
      "optimizer.cosine_final_learning_rate_scale",
      minimum=0.0,
      maximum=1.0,
  )

  training["epochs"] = _check_int(training["epochs"], "training.epochs", minimum=1)
  training["seed"] = _check_int(training["seed"], "training.seed", minimum=0)
  training["max_train_batches_per_epoch"] = _check_int(
      training["max_train_batches_per_epoch"],
      "training.max_train_batches_per_epoch",
      minimum=1,
      allow_none=True,
  )
  training["max_eval_batches"] = _check_int(
      training["max_eval_batches"],
      "training.max_eval_batches",
      minimum=1,
      allow_none=True,
  )
  training["console_every_n_epochs"] = _check_int(
      training["console_every_n_epochs"],
      "training.console_every_n_epochs",
      minimum=1,
  )

  if output.get("log_root") is not None and not isinstance(output["log_root"], str):
    raise ValueError("output.log_root must be a string path or null")
  if output.get("run_name_prefix") is not None and not isinstance(
      output["run_name_prefix"], str
  ):
    raise ValueError("output.run_name_prefix must be a string or null")
  for key in ("checkpoint_filename", "metrics_json_filename", "csv_log_filename"):
    if output.get(key) is not None and not isinstance(output[key], str):
      raise ValueError(f"output.{key} must be a string filename or null")
  for key in ("checkpoint", "metrics_json", "csv_log"):
    if output.get(key) is not None and not isinstance(output[key], str):
      raise ValueError(f"output.{key} must be a string path or null")

  output["log_root"] = _resolve_path(output.get("log_root"))
  output["run_name_prefix"] = _normalize_run_name_prefix(
      output.get("run_name_prefix")
  )
  output["checkpoint"] = _resolve_path(output.get("checkpoint"))
  output["metrics_json"] = _resolve_path(output.get("metrics_json"))
  output["csv_log"] = _resolve_path(output.get("csv_log"))
  output["run_dir"] = None

  return config


def _append_xla_flags(flags: list[str]) -> None:
  existing = os.environ.get("XLA_FLAGS", "").strip()
  merged = [flag for flag in flags if flag]
  if existing:
    merged.append(existing)
  if merged:
    os.environ["XLA_FLAGS"] = " ".join(merged)


def _apply_runtime_config(runtime: dict[str, Any]) -> None:
  if runtime["jax_platforms"]:
    os.environ["JAX_PLATFORMS"] = runtime["jax_platforms"]
  if runtime["tf_cpp_min_log_level"]:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = runtime["tf_cpp_min_log_level"]
  if runtime.get("xla_python_client_preallocate") is not None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = (
        "true" if runtime["xla_python_client_preallocate"] else "false"
    )
  if runtime.get("xla_python_client_mem_fraction") is not None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
        runtime["xla_python_client_mem_fraction"]
    )
  if runtime.get("xla_python_client_allocator"):
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = runtime["xla_python_client_allocator"]

  xla_flags = list(runtime["xla_flags_append"])
  if runtime["disable_triton_gemm"]:
    xla_flags.insert(0, "--xla_gpu_enable_triton_gemm=false")
  _append_xla_flags(xla_flags)


def _sanitize_sys_path() -> None:
  """Avoid local shadow files such as `jax.py` hijacking package imports."""
  project_root = PROJECT_ROOT.resolve()
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


CONFIG = _load_config(_config_path())
_apply_runtime_config(CONFIG["runtime"])
_sanitize_sys_path()

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .grain_dataset import create_grain_pair_datasets
from .model import VBHamiltonianPredictor


def _init_jax_runtime() -> dict[str, Any]:
  requested_platforms = os.environ.get("JAX_PLATFORMS")
  try:
    return {
        "default_backend": jax.default_backend(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "local_devices": [str(device) for device in jax.local_devices()],
    }
  except Exception as exc:
    raise RuntimeError(
        "failed to initialize the JAX backend"
        + (
            f" for JAX_PLATFORMS={requested_platforms!r}"
            if requested_platforms is not None
            else ""
        )
        + ". If CUDA is unavailable on this machine, set "
          "`runtime.jax_platforms: cpu` in `deepvbh/configs/train_jax.yaml`."
    ) from exc


def _iter_batches(dataset: Any, max_batches: int | None):
  iterator = iter(dataset)
  if max_batches is None:
    yield from iterator
    return
  for batch_index, batch in enumerate(iterator):
    if batch_index >= max_batches:
      break
    yield batch


def _batch_size(batch: dict[str, Any]) -> int:
  if "sample_index" in batch:
    return int(np.asarray(batch["sample_index"]).shape[0])
  return int(np.asarray(next(iter(batch.values()))).shape[0])


def _compute_target_stats(
    dataset: Any,
    target_key: str,
    max_batches: int | None,
) -> dict[str, float]:
  total_count = 0
  total = 0.0
  total_sq = 0.0
  for batch in _iter_batches(dataset, max_batches):
    values = np.asarray(batch[target_key], dtype=np.float64).reshape(-1)
    total_count += int(values.size)
    total += float(values.sum())
    total_sq += float(np.square(values).sum())
  if total_count == 0:
    raise RuntimeError("training split produced no target values")
  mean = total / total_count
  variance = max(total_sq / total_count - mean * mean, 0.0)
  return {"mean": mean, "std": max(variance**0.5, 1.0e-6)}


def _train_batches_per_epoch(
    train_record_count: int,
    batch_size: int | None,
    max_train_batches_per_epoch: int | None,
) -> int:
  if batch_size is None:
    raise ValueError("dataset.batch_size must be set for the configured optimizer schedule")
  batches = (train_record_count + batch_size - 1) // batch_size
  if max_train_batches_per_epoch is not None:
    batches = min(batches, max_train_batches_per_epoch)
  if batches <= 0:
    raise ValueError("training split produced zero batches")
  return batches


def _make_learning_rate_schedule(
    optimizer_config: dict[str, Any],
    training_config: dict[str, Any],
    *,
    train_record_count: int,
    batch_size: int | None,
    max_train_batches_per_epoch: int | None,
):
  base_learning_rate = float(optimizer_config["learning_rate"])
  schedule_name = optimizer_config["schedule"]
  train_batches = _train_batches_per_epoch(
      train_record_count,
      batch_size,
      max_train_batches_per_epoch,
  )
  total_steps = max(1, training_config["epochs"] * train_batches)

  if schedule_name == "constant":
    return (
        base_learning_rate,
        {
            "name": schedule_name,
            "train_batches_per_epoch": train_batches,
            "total_steps": total_steps,
            "warmup_steps": 0,
            "end_learning_rate": base_learning_rate,
        },
    )

  warmup_steps = optimizer_config["warmup_epochs"] * train_batches
  if total_steps > 1:
    warmup_steps = min(warmup_steps, total_steps - 1)
  else:
    warmup_steps = 0
  end_learning_rate = (
      base_learning_rate * optimizer_config["cosine_final_learning_rate_scale"]
  )
  schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0 if warmup_steps > 0 else base_learning_rate,
      peak_value=base_learning_rate,
      warmup_steps=warmup_steps,
      decay_steps=total_steps,
      end_value=end_learning_rate,
  )
  return (
      schedule,
      {
          "name": schedule_name,
          "train_batches_per_epoch": train_batches,
          "total_steps": total_steps,
          "warmup_steps": warmup_steps,
          "end_learning_rate": end_learning_rate,
      },
  )


def _make_optimizer(
    model: VBHamiltonianPredictor,
    config: dict[str, Any],
    learning_rate: Any,
) -> nnx.Optimizer:
  transforms = []
  if config["grad_clip_norm"] is not None and config["grad_clip_norm"] > 0.0:
    transforms.append(optax.clip_by_global_norm(config["grad_clip_norm"]))
  transforms.append(
      optax.adamw(
          learning_rate=learning_rate,
          weight_decay=config["weight_decay"],
      )
  )
  return nnx.Optimizer(model, optax.chain(*transforms), wrt=nnx.Param)


def _build_step_functions(target_key: str):
  def loss_fn(
      model: VBHamiltonianPredictor,
      batch: dict[str, jax.Array],
      target_std: jax.Array,
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    predictions = jnp.squeeze(
        model(
            batch["pair_input"],
            node_mask=batch["node_mask"],
            structure_overlap=batch["structure_overlap"],
        ),
        axis=-1,
    )
    targets = jnp.asarray(batch[target_key], dtype=jnp.float32)
    residual = predictions - targets
    normalized_residual = residual / target_std
    loss = jnp.mean(jnp.square(normalized_residual))
    return loss, {
        "loss": loss,
        "mae": jnp.mean(jnp.abs(residual)),
        "rmse": jnp.sqrt(jnp.mean(jnp.square(residual))),
    }

  grad_fn = nnx.value_and_grad(
      loss_fn,
      argnums=nnx.DiffState(0, nnx.Param),
      has_aux=True,
  )

  @nnx.jit
  def train_step(
      model: VBHamiltonianPredictor,
      optimizer: nnx.Optimizer,
      batch: dict[str, jax.Array],
      target_std: jax.Array,
  ) -> dict[str, jax.Array]:
    (_, metrics), grads = grad_fn(model, batch, target_std)
    optimizer.update(model, grads)
    return metrics

  @nnx.jit
  def eval_step(
      model: VBHamiltonianPredictor,
      batch: dict[str, jax.Array],
      target_std: jax.Array,
  ) -> dict[str, jax.Array]:
    _, metrics = loss_fn(model, batch, target_std)
    return metrics

  return train_step, eval_step


def _run_epoch(
    dataset: Any,
    *,
    step_fn: Any,
    model: VBHamiltonianPredictor,
    optimizer: nnx.Optimizer | None,
    target_std: jax.Array,
    max_batches: int | None,
) -> dict[str, float] | None:
  totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0}
  total_examples = 0

  for batch in _iter_batches(dataset, max_batches):
    batch_examples = _batch_size(batch)
    metrics = (
        step_fn(model, batch, target_std)
        if optimizer is None
        else step_fn(model, optimizer, batch, target_std)
    )
    total_examples += batch_examples
    for key in totals:
      totals[key] += float(np.asarray(metrics[key])) * batch_examples

  if total_examples == 0:
    return None
  return {key: value / total_examples for key, value in totals.items()}


def _save_pickle(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as output:
    pickle.dump(payload, output)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")


class _CsvMetricLogger:
  FIELDNAMES = (
      "epoch",
      "learning_rate",
      "train_loss",
      "train_mae",
      "train_rmse",
      "valid_loss",
      "valid_mae",
      "valid_rmse",
      "test_loss",
      "test_mae",
      "test_rmse",
      "best_valid_epoch",
      "best_valid_loss",
      "epoch_wall_time_seconds",
  )

  def __init__(self, path: Path | None):
    self._handle = None
    self._writer = None
    if path is None:
      return
    path.parent.mkdir(parents=True, exist_ok=True)
    self._handle = path.open("w", newline="")
    self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
    self._writer.writeheader()
    self._handle.flush()

  def append(self, row: dict[str, Any]) -> None:
    if self._writer is None or self._handle is None:
      return
    self._writer.writerow(
        {key: "" if row.get(key) is None else row.get(key) for key in self.FIELDNAMES}
    )
    self._handle.flush()

  def close(self) -> None:
    if self._handle is not None:
      self._handle.close()


def _metric_or_na(value: float | None) -> str:
  if value is None:
    return "na"
  return f"{value:.6f}"


def _should_print_epoch(epoch: int, epochs: int, interval: int) -> bool:
  return epoch == 1 or epoch == epochs or epoch % interval == 0


def main() -> None:
  runtime_info = _init_jax_runtime()
  dataset_config = CONFIG["dataset"]
  model_config = CONFIG["model"]
  optimizer_config = CONFIG["optimizer"]
  training_config = CONFIG["training"]
  output_config = CONFIG["output"]
  _prepare_output_paths(output_config)
  csv_logger = _CsvMetricLogger(output_config["csv_log"])

  datasets = create_grain_pair_datasets(
      dataset_config["data_roots"],
      batch_size=dataset_config["batch_size"],
      valid_fraction=dataset_config["valid_fraction"],
      test_fraction=dataset_config["test_fraction"],
      split_seed=dataset_config["split_seed"],
      shuffle_train=True,
      shuffle_seed=dataset_config["shuffle_seed"],
      include_diagonal=not dataset_config["exclude_diagonal"],
      upper_triangle_only=dataset_config["upper_triangle_only"],
      step_selection=dataset_config["step_selection"],
      max_steps_per_sample=dataset_config["max_steps_per_sample"],
      max_train_pairs_per_step=dataset_config["max_train_pairs_per_step"],
      pair_sample_seed=dataset_config["shuffle_seed"],
      feature_dtype="float32",
      target_dtype="float32",
      train_num_epochs=None,
      eval_num_epochs=1,
      drop_remainder=False,
      shard_count=runtime_info["process_count"],
      shard_index=runtime_info["process_index"],
  )

  target_stats = _compute_target_stats(
      datasets.train,
      dataset_config["target_key"],
      training_config["max_train_batches_per_epoch"],
  )
  target_std = jnp.asarray(target_stats["std"], dtype=jnp.float32)
  learning_rate_schedule, schedule_metadata = _make_learning_rate_schedule(
      optimizer_config,
      training_config,
      train_record_count=datasets.metadata.train_record_count,
      batch_size=dataset_config["batch_size"],
      max_train_batches_per_epoch=training_config["max_train_batches_per_epoch"],
  )

  model = VBHamiltonianPredictor(
      d_model=model_config["d_model"],
      num_heads=model_config["num_heads"],
      num_layers=model_config["num_layers"],
      mlp_hidden_dim=model_config["mlp_hidden_dim"],
      readout_hidden_dim=model_config["readout_hidden_dim"],
      rngs=nnx.Rngs(training_config["seed"]),
  )
  optimizer = _make_optimizer(model, optimizer_config, learning_rate_schedule)
  train_step, eval_step = _build_step_functions(dataset_config["target_key"])

  final_metrics: dict[str, Any] | None = None
  best_valid_loss = float("inf")
  best_valid_epoch = 0

  print(f"config_path = {CONFIG['config_path']}")
  print(
      "run = "
      f"backend:{runtime_info['default_backend']} "
      f"devices:{runtime_info['local_devices']} "
      f"run_dir:{output_config['run_dir']} "
      f"csv_log:{output_config['csv_log']}"
  )
  print(
      "schedule = "
      f"{schedule_metadata['name']} "
      f"batches_per_epoch:{schedule_metadata['train_batches_per_epoch']} "
      f"warmup_steps:{schedule_metadata['warmup_steps']} "
      f"total_steps:{schedule_metadata['total_steps']}"
  )
  print(
      "dataset = "
      f"samples:{datasets.metadata.sample_count} "
      f"train:{datasets.metadata.train_record_count} "
      f"valid:{datasets.metadata.valid_record_count} "
      f"test:{datasets.metadata.test_record_count} "
      f"train_pair_cap:{dataset_config['max_train_pairs_per_step']}"
  )
  print(
      "target = "
      f"{dataset_config['target_key']} "
      f"mean:{target_stats['mean']:.9f} "
      f"std:{target_stats['std']:.9f}"
  )

  try:
    for epoch in range(1, training_config["epochs"] + 1):
      epoch_start = time.perf_counter()
      train_metrics = _run_epoch(
          datasets.train,
          step_fn=train_step,
          model=model,
          optimizer=optimizer,
          target_std=target_std,
          max_batches=training_config["max_train_batches_per_epoch"],
      )
      if train_metrics is None:
        raise RuntimeError("training split produced no batches")

      valid_metrics = _run_epoch(
          datasets.valid,
          step_fn=eval_step,
          model=model,
          optimizer=None,
          target_std=target_std,
          max_batches=training_config["max_eval_batches"],
      )
      test_metrics = _run_epoch(
          datasets.test,
          step_fn=eval_step,
          model=model,
          optimizer=None,
          target_std=target_std,
          max_batches=training_config["max_eval_batches"],
      )

      train_step_index = epoch * schedule_metadata["train_batches_per_epoch"]
      current_learning_rate = (
          float(learning_rate_schedule(train_step_index))
          if callable(learning_rate_schedule)
          else float(learning_rate_schedule)
      )
      metrics = {
          "epoch": epoch,
          "learning_rate": current_learning_rate,
          "train_loss": train_metrics["loss"],
          "train_mae": train_metrics["mae"],
          "train_rmse": train_metrics["rmse"],
          "valid_loss": None if valid_metrics is None else valid_metrics["loss"],
          "valid_mae": None if valid_metrics is None else valid_metrics["mae"],
          "valid_rmse": None if valid_metrics is None else valid_metrics["rmse"],
          "test_loss": None if test_metrics is None else test_metrics["loss"],
          "test_mae": None if test_metrics is None else test_metrics["mae"],
          "test_rmse": None if test_metrics is None else test_metrics["rmse"],
          "epoch_wall_time_seconds": time.perf_counter() - epoch_start,
      }
      final_metrics = metrics

      if metrics["valid_loss"] is not None and metrics["valid_loss"] < best_valid_loss:
        best_valid_loss = metrics["valid_loss"]
        best_valid_epoch = epoch

      csv_logger.append(
          {
              **metrics,
              "best_valid_epoch": best_valid_epoch if best_valid_epoch > 0 else None,
              "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
          }
      )

      if _should_print_epoch(
          epoch,
          training_config["epochs"],
          training_config["console_every_n_epochs"],
      ):
        print(
            f"epoch={epoch:04d} "
            f"lr={metrics['learning_rate']:.8f} "
            f"train_mae={metrics['train_mae']:.6f} "
            f"train_rmse={metrics['train_rmse']:.6f} "
            f"valid_mae={_metric_or_na(metrics['valid_mae'])} "
            f"valid_rmse={_metric_or_na(metrics['valid_rmse'])} "
            f"test_mae={_metric_or_na(metrics['test_mae'])} "
            f"test_rmse={_metric_or_na(metrics['test_rmse'])} "
            f"best_valid={_metric_or_na(None if best_valid_epoch == 0 else best_valid_loss)} "
            f"dt_s={metrics['epoch_wall_time_seconds']:.3f}"
        )

    if best_valid_epoch > 0:
      print(f"best_valid = epoch:{best_valid_epoch} loss:{best_valid_loss:.6f}")

    dataset_metadata = _json_ready(datasets.metadata.__dict__)
    config_payload = _json_ready(CONFIG)

    if output_config["checkpoint"] is not None:
      _save_pickle(
          output_config["checkpoint"],
          {
              "model_state": nnx.state(model),
              "optimizer_state": nnx.state(optimizer),
              "target_stats": target_stats,
              "config": config_payload,
              "final_metrics": final_metrics,
              "best_valid_epoch": best_valid_epoch,
              "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
          },
      )
      print(f"checkpoint = {output_config['checkpoint']}")

    if output_config["metrics_json"] is not None:
      _save_json(
          output_config["metrics_json"],
          {
              "config": config_payload,
              "dataset_metadata": dataset_metadata,
              "target_stats": target_stats,
              "final_metrics": final_metrics,
              "best_valid_epoch": best_valid_epoch,
              "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
          },
      )
      print(f"metrics_json = {output_config['metrics_json']}")
  finally:
    csv_logger.close()


if __name__ == "__main__":
  main()
