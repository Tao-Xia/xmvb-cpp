"""Training runtime and epoch loop for DeepVBH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import pickle
import signal
import time
import traceback
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..grain_dataset import create_grain_pair_datasets
from ..model import VBHamiltonianPredictor
from .metrics import MetricAccumulator
from .prefetch import DeviceBatchPrefetcher
from .types import TrainingConfig, json_ready


@dataclass(frozen=True)
class JaxRuntimeInfo:
  """Static information about the initialized JAX runtime."""

  default_backend: str
  process_count: int
  process_index: int
  local_devices: list[str]


@dataclass(frozen=True)
class TargetStats:
  """Normalization statistics computed from the training targets."""

  mean: float
  std: float


@dataclass(frozen=True)
class LearningRatePlan:
  """Optimizer schedule plus metadata used for logging."""

  schedule: Any
  name: str
  train_batches_per_epoch: int
  total_steps: int
  warmup_steps: int
  end_learning_rate: float

  def as_logging_dict(self) -> dict[str, Any]:
    """Builds a JSON-serializable summary of the learning-rate plan.

    Returns:
      Mapping with the logging-relevant scalar fields and without the live
      schedule callable.
    """

    return {
        "name": self.name,
        "train_batches_per_epoch": self.train_batches_per_epoch,
        "total_steps": self.total_steps,
        "warmup_steps": self.warmup_steps,
        "end_learning_rate": self.end_learning_rate,
    }


class CsvMetricLogger:
  """Writes per-epoch metrics to disk when CSV output is enabled."""

  fieldnames = (
      "epoch",
      "learning_rate",
      "train_loss",
      "train_hij_mae",
      "train_hij_rmse",
      "train_batch_count",
      "train_example_count",
      "train_wall_time_seconds",
      "train_examples_per_second",
      "train_batches_per_second",
      "valid_loss",
      "valid_hij_mae",
      "valid_hij_rmse",
      "valid_batch_count",
      "valid_example_count",
      "valid_wall_time_seconds",
      "valid_examples_per_second",
      "valid_batches_per_second",
      "test_loss",
      "test_hij_mae",
      "test_hij_rmse",
      "test_batch_count",
      "test_example_count",
      "test_wall_time_seconds",
      "test_examples_per_second",
      "test_batches_per_second",
      "best_valid_epoch",
      "best_valid_loss",
      "epoch_wall_time_seconds",
      "run_wall_time_seconds",
  )

  def __init__(self, path: Path | None):
    self.handle = None
    self.writer = None
    if path is None:
      return
    path.parent.mkdir(parents=True, exist_ok=True)
    self.handle = path.open("w", newline="")
    self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
    self.writer.writeheader()
    self.handle.flush()

  def append(self, row: dict[str, Any]) -> None:
    if self.writer is None or self.handle is None:
      return
    self.writer.writerow(
        {key: "" if row.get(key) is None else row.get(key) for key in self.fieldnames}
    )
    self.handle.flush()

  def close(self) -> None:
    if self.handle is not None:
      self.handle.close()


class JsonlMetricLogger:
  """Writes structured training records to disk as JSON Lines."""

  def __init__(self, path: Path | None):
    self.handle = None
    if path is None:
      return
    path.parent.mkdir(parents=True, exist_ok=True)
    self.handle = path.open("w")

  def append(self, row: dict[str, Any]) -> None:
    """Appends one structured record.

    Args:
      row: JSON-serializable training record.
    """

    if self.handle is None:
      return
    self.handle.write(json.dumps(json_ready(row)) + "\n")
    self.handle.flush()

  def close(self) -> None:
    """Closes the JSON Lines log file if it is open."""

    if self.handle is not None:
      self.handle.close()


class TextLogWriter:
  """Writes readable training messages to stdout and an optional text log."""

  def __init__(self, path: Path | None):
    self.handle = None
    if path is None:
      return
    path.parent.mkdir(parents=True, exist_ok=True)
    self.handle = path.open("w")

  def write(self, message: str) -> None:
    """Writes one line to stdout and to the text log.

    Args:
      message: Human-readable log line.
    """

    print(message, flush=True)
    if self.handle is not None:
      self.handle.write(message + "\n")
      self.handle.flush()

  def close(self) -> None:
    """Closes the text log file if it is open."""

    if self.handle is not None:
      self.handle.close()


class TrainingApplication:
  """Owns dataset creation, optimization, logging, and checkpointing."""

  def __init__(self, config: TrainingConfig):
    self.config = config

  def init_runtime_info(self) -> JaxRuntimeInfo:
    requested_platforms = self.config.runtime.jax_platforms
    try:
      return JaxRuntimeInfo(
          default_backend=jax.default_backend(),
          process_count=jax.process_count(),
          process_index=jax.process_index(),
          local_devices=[str(device) for device in jax.local_devices()],
      )
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

  def iter_batches(self, dataset: Any, max_batches: int | None):
    iterator = iter(dataset)
    if max_batches is None:
      yield from iterator
      return
    for batch_index, batch in enumerate(iterator):
      if batch_index >= max_batches:
        break
      yield batch

  def batch_size(self, batch: dict[str, Any]) -> int:
    return int(next(iter(batch.values())).shape[0])

  def compute_target_stats(self, dataset: Any) -> TargetStats:
    total_count = 0
    total = 0.0
    total_sq = 0.0
    max_batches = self.config.training.max_train_batches_per_epoch

    for batch in self.iter_batches(dataset, max_batches):
      values = np.asarray(batch["target_hamiltonian"], dtype=np.float64).reshape(-1)
      total_count += int(values.size)
      total += float(values.sum())
      total_sq += float(np.square(values).sum())

    if total_count == 0:
      raise RuntimeError("training split produced no target values")

    mean = total / total_count
    variance = max(total_sq / total_count - mean * mean, 0.0)
    return TargetStats(mean=mean, std=max(variance**0.5, 1.0e-6))

  def train_batches_per_epoch(self, train_record_count: int) -> int:
    batch_size = self.config.dataset.batch_size
    max_train_batches = self.config.training.max_train_batches_per_epoch
    batches = (train_record_count + batch_size - 1) // batch_size
    if max_train_batches is not None:
      batches = min(batches, max_train_batches)
    if batches <= 0:
      raise ValueError("training split produced zero batches")
    return batches

  def make_learning_rate_plan(self, train_record_count: int) -> LearningRatePlan:
    optimizer_config = self.config.optimizer
    training_config = self.config.training
    base_learning_rate = float(optimizer_config.learning_rate)
    train_batches = self.train_batches_per_epoch(train_record_count)
    total_steps = max(1, training_config.epochs * train_batches)

    if optimizer_config.schedule == "constant":
      return LearningRatePlan(
          schedule=base_learning_rate,
          name=optimizer_config.schedule,
          train_batches_per_epoch=train_batches,
          total_steps=total_steps,
          warmup_steps=0,
          end_learning_rate=base_learning_rate,
      )

    warmup_steps = optimizer_config.warmup_epochs * train_batches
    if total_steps > 1:
      warmup_steps = min(warmup_steps, total_steps - 1)
    else:
      warmup_steps = 0
    end_learning_rate = (
        base_learning_rate * optimizer_config.cosine_final_learning_rate_scale
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0 if warmup_steps > 0 else base_learning_rate,
        peak_value=base_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end_learning_rate,
    )
    return LearningRatePlan(
        schedule=schedule,
        name=optimizer_config.schedule,
        train_batches_per_epoch=train_batches,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        end_learning_rate=end_learning_rate,
    )

  def make_optimizer(self, model: VBHamiltonianPredictor, learning_rate: Any) -> nnx.Optimizer:
    transforms = []
    grad_clip_norm = self.config.optimizer.grad_clip_norm
    if grad_clip_norm is not None and grad_clip_norm > 0.0:
      transforms.append(optax.clip_by_global_norm(grad_clip_norm))
    transforms.append(
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
    )
    return nnx.Optimizer(model, optax.chain(*transforms), wrt=nnx.Param)

  def build_step_functions(self):
    def loss_fn(
        model: VBHamiltonianPredictor,
        batch: dict[str, jax.Array],
        target_std: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
      predictions = jnp.squeeze(
          model(
              batch["pair_features"],
              batch["pair_relations"],
              pair_mask=batch["pair_mask"],
              structure_overlap=batch["structure_overlap"],
          ),
          axis=-1,
      )
      targets = jnp.asarray(batch["target_hamiltonian"], dtype=jnp.float32)
      residual = predictions - targets
      normalized_residual = residual / target_std
      loss = jnp.mean(jnp.square(normalized_residual))
      return loss, {
          "loss": loss,
          "hij_mae": jnp.mean(jnp.abs(residual)),
          "hij_rmse": jnp.sqrt(jnp.mean(jnp.square(residual))),
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

  def run_epoch(
      self,
      dataset: Any,
      *,
      step_fn: Any,
      model: VBHamiltonianPredictor,
      optimizer: nnx.Optimizer | None,
      target_std: jax.Array,
      max_batches: int | None,
  ) -> dict[str, float] | None:
    split_start = time.perf_counter()
    accumulator = MetricAccumulator()
    prefetcher = DeviceBatchPrefetcher(self.config.training.prefetch_batches)
    batch_count = 0
    example_count = 0

    for batch in prefetcher.iter_batches(self.iter_batches(dataset, max_batches)):
      batch_examples = self.batch_size(batch)
      metrics = (
          step_fn(model, batch, target_std)
          if optimizer is None
          else step_fn(model, optimizer, batch, target_std)
      )
      accumulator.add(metrics, batch_examples)
      batch_count += 1
      example_count += batch_examples

    split_metrics = accumulator.compute()
    if split_metrics is None:
      return None

    wall_time_seconds = time.perf_counter() - split_start
    split_metrics["batch_count"] = batch_count
    split_metrics["example_count"] = example_count
    split_metrics["wall_time_seconds"] = wall_time_seconds
    split_metrics["examples_per_second"] = example_count / max(
        wall_time_seconds,
        1.0e-8,
    )
    split_metrics["batches_per_second"] = batch_count / max(
        wall_time_seconds,
        1.0e-8,
    )
    return split_metrics

  def save_pickle(self, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as output:
      pickle.dump(payload, output)
    temp_path.replace(path)

  def save_json(self, path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(json_ready(payload), indent=2) + "\n")
    temp_path.replace(path)

  def best_checkpoint_path(self) -> Path | None:
    checkpoint_path = self.config.output.checkpoint
    if checkpoint_path is None:
      return None
    if checkpoint_path.suffix:
      name = f"{checkpoint_path.stem}_best{checkpoint_path.suffix}"
    else:
      name = f"{checkpoint_path.name}_best"
    return checkpoint_path.with_name(name)

  def checkpoint_payload(
      self,
      *,
      model: VBHamiltonianPredictor,
      optimizer: nnx.Optimizer,
      target_stats: TargetStats,
      config_payload: dict[str, Any],
      metrics: dict[str, Any] | None,
      epoch: int,
      best_valid_epoch: int,
      best_valid_loss: float,
      save_reason: str,
  ) -> dict[str, Any]:
    """Builds one resumable checkpoint payload for long-running jobs."""

    return {
        "model_state": nnx.state(model),
        "optimizer_state": nnx.state(optimizer),
        "target_stats": asdict(target_stats),
        "config": config_payload,
        "final_metrics": metrics,
        "epoch": epoch,
        "save_reason": save_reason,
        "best_valid_epoch": best_valid_epoch,
        "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
    }

  def save_checkpoint(
      self,
      path: Path,
      *,
      model: VBHamiltonianPredictor,
      optimizer: nnx.Optimizer,
      target_stats: TargetStats,
      config_payload: dict[str, Any],
      metrics: dict[str, Any] | None,
      epoch: int,
      best_valid_epoch: int,
      best_valid_loss: float,
      save_reason: str,
  ) -> None:
    self.save_pickle(
        path,
        self.checkpoint_payload(
            model=model,
            optimizer=optimizer,
            target_stats=target_stats,
            config_payload=config_payload,
            metrics=metrics,
            epoch=epoch,
            best_valid_epoch=best_valid_epoch,
            best_valid_loss=best_valid_loss,
            save_reason=save_reason,
        ),
    )

  def metric_or_na(self, value: float | int | None, digits: int = 6) -> str:
    """Formats one scalar metric for human-readable logs.

    Args:
      value: Metric value or `None`.
      digits: Number of digits after the decimal point.

    Returns:
      Formatted metric string or `na`.
    """

    if value is None:
      return "na"
    return f"{float(value):.{digits}f}"

  def split_metric(self, metrics: dict[str, Any] | None, key: str) -> Any:
    """Returns one value from optional split metrics.

    Args:
      metrics: Split-level metric mapping or `None`.
      key: Metric name to read.

    Returns:
      Requested scalar value, or `None` when the split is absent.
    """

    if metrics is None:
      return None
    return metrics[key]

  def should_print_epoch(self, epoch: int) -> bool:
    interval = self.config.training.console_every_n_epochs
    return epoch == 1 or epoch == self.config.training.epochs or epoch % interval == 0

  def create_model(self) -> VBHamiltonianPredictor:
    model_config = self.config.model
    return VBHamiltonianPredictor(
        d_model=model_config.d_model,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
        mlp_hidden_dim=model_config.mlp_hidden_dim,
        readout_hidden_dim=model_config.readout_hidden_dim,
        rngs=nnx.Rngs(self.config.training.seed),
    )

  def learning_rate_value(self, schedule: Any, step: int) -> float:
    if callable(schedule):
      return float(schedule(step))
    return float(schedule)

  def run(self) -> None:
    runtime_info = self.init_runtime_info()
    self.config.output.prepare_paths()
    csv_logger = CsvMetricLogger(self.config.output.csv_log)
    jsonl_logger = JsonlMetricLogger(self.config.output.metrics_jsonl)
    text_logger = TextLogWriter(self.config.output.text_log)
    run_start = time.perf_counter()
    previous_signal_handlers: dict[int, Any] = {}

    try:
      datasets = create_grain_pair_datasets(
          self.config.dataset.data_roots,
          batch_size=self.config.dataset.batch_size,
          train_selected_steps=self.config.dataset.train_selected_steps,
          nonfinal_offdiagonal_pair_fraction=(
              self.config.dataset.nonfinal_offdiagonal_pair_fraction
          ),
          valid_fraction=self.config.dataset.valid_fraction,
          test_fraction=self.config.dataset.test_fraction,
          split_seed=self.config.dataset.split_seed,
          shuffle_train=True,
          shuffle_seed=self.config.dataset.shuffle_seed,
          feature_dtype="float32",
          target_dtype="float32",
          train_num_epochs=None,
          eval_num_epochs=1,
          drop_remainder=False,
          shard_count=runtime_info.process_count,
          shard_index=runtime_info.process_index,
      )

      target_stats = self.compute_target_stats(datasets.train)
      target_std = jnp.asarray(target_stats.std, dtype=jnp.float32)
      learning_rate_plan = self.make_learning_rate_plan(
          datasets.metadata.train_record_count
      )

      model = self.create_model()
      optimizer = self.make_optimizer(model, learning_rate_plan.schedule)
      train_step, eval_step = self.build_step_functions()
      dataset_metadata = json_ready(asdict(datasets.metadata))
      config_payload = self.config.as_serializable()

      final_metrics: dict[str, Any] | None = None
      best_valid_loss = float("inf")
      best_valid_epoch = 0
      best_checkpoint = self.best_checkpoint_path()
      stop_signal_name: str | None = None

      def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_signal_name
        if stop_signal_name is None:
          stop_signal_name = signal.Signals(signum).name

      for stop_signal in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[stop_signal] = signal.getsignal(stop_signal)
        signal.signal(stop_signal, request_stop)

      text_logger.write(f"config_path = {self.config.config_path}")
      text_logger.write(
          "run = "
          f"backend:{runtime_info.default_backend} "
          f"devices:{runtime_info.local_devices} "
          f"run_dir:{self.config.output.run_dir} "
          f"csv_log:{self.config.output.csv_log} "
          f"jsonl_log:{self.config.output.metrics_jsonl} "
          f"text_log:{self.config.output.text_log}"
      )
      text_logger.write(
          "schedule = "
          f"{learning_rate_plan.name} "
          f"batches_per_epoch:{learning_rate_plan.train_batches_per_epoch} "
          f"warmup_steps:{learning_rate_plan.warmup_steps} "
          f"total_steps:{learning_rate_plan.total_steps}"
      )
      text_logger.write(
          "dataset = "
          f"samples:{datasets.metadata.sample_count} "
          f"train:{datasets.metadata.train_record_count} "
          f"valid:{datasets.metadata.valid_record_count} "
          f"test:{datasets.metadata.test_record_count} "
          f"train_step_policy:{datasets.metadata.train_step_policy} "
          f"eval_step_policy:{datasets.metadata.eval_step_policy} "
          f"train_selected_steps:{datasets.metadata.train_selected_steps} "
          f"pair_policy:{datasets.metadata.pair_policy}"
      )
      text_logger.write(
          "pair_sampling = "
          f"nonfinal_offdiag_fraction:"
          f"{datasets.metadata.nonfinal_offdiagonal_pair_fraction:.3f} "
          "final_step:all_pairs"
      )
      text_logger.write(
          "target = "
          "target_hamiltonian "
          f"mean:{target_stats.mean:.9f} "
          f"std:{target_stats.std:.9f}"
      )
      if self.config.output.checkpoint is not None:
        checkpoint_policy = f"latest:{self.config.output.checkpoint} every_epoch"
        if best_checkpoint is not None:
          checkpoint_policy += f" best:{best_checkpoint} on_valid_improvement"
        text_logger.write(f"checkpoint_policy = {checkpoint_policy}")
      jsonl_logger.append(
          {
              "event": "run_start",
              "config_path": self.config.config_path,
              "runtime_info": asdict(runtime_info),
              "dataset_metadata": dataset_metadata,
              "target_stats": asdict(target_stats),
              "learning_rate_plan": learning_rate_plan.as_logging_dict(),
              "config": config_payload,
              "output_paths": {
                  "run_dir": self.config.output.run_dir,
                  "csv_log": self.config.output.csv_log,
                  "metrics_jsonl": self.config.output.metrics_jsonl,
                  "text_log": self.config.output.text_log,
                  "metrics_json": self.config.output.metrics_json,
                  "checkpoint": self.config.output.checkpoint,
              },
          }
      )

      for epoch in range(1, self.config.training.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = self.run_epoch(
            datasets.train,
            step_fn=train_step,
            model=model,
            optimizer=optimizer,
            target_std=target_std,
            max_batches=self.config.training.max_train_batches_per_epoch,
        )
        if train_metrics is None:
          raise RuntimeError("training split produced no batches")

        valid_metrics = self.run_epoch(
            datasets.valid,
            step_fn=eval_step,
            model=model,
            optimizer=None,
            target_std=target_std,
            max_batches=self.config.training.max_eval_batches,
        )
        test_metrics = self.run_epoch(
            datasets.test,
            step_fn=eval_step,
            model=model,
            optimizer=None,
            target_std=target_std,
            max_batches=self.config.training.max_eval_batches,
        )

        train_step_index = epoch * learning_rate_plan.train_batches_per_epoch
        current_learning_rate = self.learning_rate_value(
            learning_rate_plan.schedule,
            train_step_index,
        )
        metrics = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_metrics["loss"],
            "train_hij_mae": train_metrics["hij_mae"],
            "train_hij_rmse": train_metrics["hij_rmse"],
            "train_batch_count": train_metrics["batch_count"],
            "train_example_count": train_metrics["example_count"],
            "train_wall_time_seconds": train_metrics["wall_time_seconds"],
            "train_examples_per_second": train_metrics["examples_per_second"],
            "train_batches_per_second": train_metrics["batches_per_second"],
            "valid_loss": self.split_metric(valid_metrics, "loss"),
            "valid_hij_mae": self.split_metric(valid_metrics, "hij_mae"),
            "valid_hij_rmse": self.split_metric(valid_metrics, "hij_rmse"),
            "valid_batch_count": self.split_metric(valid_metrics, "batch_count"),
            "valid_example_count": self.split_metric(valid_metrics, "example_count"),
            "valid_wall_time_seconds": self.split_metric(
                valid_metrics,
                "wall_time_seconds",
            ),
            "valid_examples_per_second": self.split_metric(
                valid_metrics,
                "examples_per_second",
            ),
            "valid_batches_per_second": self.split_metric(
                valid_metrics,
                "batches_per_second",
            ),
            "test_loss": self.split_metric(test_metrics, "loss"),
            "test_hij_mae": self.split_metric(test_metrics, "hij_mae"),
            "test_hij_rmse": self.split_metric(test_metrics, "hij_rmse"),
            "test_batch_count": self.split_metric(test_metrics, "batch_count"),
            "test_example_count": self.split_metric(test_metrics, "example_count"),
            "test_wall_time_seconds": self.split_metric(
                test_metrics,
                "wall_time_seconds",
            ),
            "test_examples_per_second": self.split_metric(
                test_metrics,
                "examples_per_second",
            ),
            "test_batches_per_second": self.split_metric(
                test_metrics,
                "batches_per_second",
            ),
            "epoch_wall_time_seconds": time.perf_counter() - epoch_start,
            "run_wall_time_seconds": time.perf_counter() - run_start,
        }
        final_metrics = metrics

        valid_improved = (
            metrics["valid_loss"] is not None
            and metrics["valid_loss"] < best_valid_loss
        )
        if valid_improved:
          best_valid_loss = metrics["valid_loss"]
          best_valid_epoch = epoch

        csv_logger.append(
            {
                **metrics,
                "best_valid_epoch": best_valid_epoch if best_valid_epoch > 0 else None,
                "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
            }
        )
        jsonl_logger.append(
            {
                "event": "epoch",
                **metrics,
                "best_valid_epoch": best_valid_epoch if best_valid_epoch > 0 else None,
                "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
            }
        )

        if self.should_print_epoch(epoch):
          text_logger.write(
              f"epoch={epoch:04d} "
              f"lr={metrics['learning_rate']:.8f} "
              f"train_hij_mae={metrics['train_hij_mae']:.6f} "
              f"train_hij_rmse={metrics['train_hij_rmse']:.6f} "
              f"train_ex_s={metrics['train_examples_per_second']:.2f} "
              f"valid_hij_mae={self.metric_or_na(metrics['valid_hij_mae'])} "
              f"valid_hij_rmse={self.metric_or_na(metrics['valid_hij_rmse'])} "
              f"test_hij_mae={self.metric_or_na(metrics['test_hij_mae'])} "
              f"test_hij_rmse={self.metric_or_na(metrics['test_hij_rmse'])} "
              f"best_valid={self.metric_or_na(None if best_valid_epoch == 0 else best_valid_loss)} "
              f"dt_s={metrics['epoch_wall_time_seconds']:.3f}"
          )

        if self.config.output.checkpoint is not None:
          self.save_checkpoint(
              self.config.output.checkpoint,
              model=model,
              optimizer=optimizer,
              target_stats=target_stats,
              config_payload=config_payload,
              metrics=metrics,
              epoch=epoch,
              best_valid_epoch=best_valid_epoch,
              best_valid_loss=best_valid_loss,
              save_reason="epoch",
          )
        if valid_improved and best_checkpoint is not None:
          self.save_checkpoint(
              best_checkpoint,
              model=model,
              optimizer=optimizer,
              target_stats=target_stats,
              config_payload=config_payload,
              metrics=metrics,
              epoch=epoch,
              best_valid_epoch=best_valid_epoch,
              best_valid_loss=best_valid_loss,
              save_reason="best_valid",
          )
          text_logger.write(
              f"best_checkpoint = epoch:{epoch} path:{best_checkpoint}"
          )
          jsonl_logger.append(
              {
                  "event": "artifact",
                  "name": "best_checkpoint",
                  "path": best_checkpoint,
                  "epoch": epoch,
                  "valid_loss": metrics["valid_loss"],
              }
          )
        if stop_signal_name is not None:
          text_logger.write(
              f"stop_requested = {stop_signal_name} "
              f"saved_epoch:{epoch} checkpoint:{self.config.output.checkpoint}"
          )
          jsonl_logger.append(
              {
                  "event": "stop_requested",
                  "signal": stop_signal_name,
                  "saved_epoch": epoch,
                  "checkpoint": self.config.output.checkpoint,
              }
          )
          break

      if best_valid_epoch > 0:
        text_logger.write(
            f"best_valid = epoch:{best_valid_epoch} loss:{best_valid_loss:.6f}"
        )

      if self.config.output.checkpoint is not None:
        self.save_checkpoint(
            self.config.output.checkpoint,
            model=model,
            optimizer=optimizer,
            target_stats=target_stats,
            config_payload=config_payload,
            metrics=final_metrics,
            epoch=0 if final_metrics is None else int(final_metrics["epoch"]),
            best_valid_epoch=best_valid_epoch,
            best_valid_loss=best_valid_loss,
            save_reason="final",
        )
        text_logger.write(f"checkpoint = {self.config.output.checkpoint}")
        jsonl_logger.append(
            {
                "event": "artifact",
                "name": "checkpoint",
                "path": self.config.output.checkpoint,
            }
        )

      if self.config.output.metrics_json is not None:
        self.save_json(
            self.config.output.metrics_json,
            {
                "config": config_payload,
                "dataset_metadata": dataset_metadata,
                "target_stats": asdict(target_stats),
                "final_metrics": final_metrics,
                "best_valid_epoch": best_valid_epoch,
                "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
            },
        )
        text_logger.write(f"metrics_json = {self.config.output.metrics_json}")
        jsonl_logger.append(
            {
                "event": "artifact",
                "name": "metrics_json",
                "path": self.config.output.metrics_json,
            }
        )

      jsonl_logger.append(
          {
              "event": "run_end",
              "final_metrics": final_metrics,
              "best_valid_epoch": best_valid_epoch,
              "best_valid_loss": None if best_valid_epoch == 0 else best_valid_loss,
              "run_wall_time_seconds": time.perf_counter() - run_start,
              "stop_signal": stop_signal_name,
          }
      )
    except Exception as exc:
      text_logger.write(f"error = {type(exc).__name__}: {exc}")
      for line in traceback.format_exc().rstrip().splitlines():
        text_logger.write(line)
      jsonl_logger.append(
          {
              "event": "error",
              "error_type": type(exc).__name__,
              "message": str(exc),
              "traceback": traceback.format_exc(),
              "run_wall_time_seconds": time.perf_counter() - run_start,
          }
      )
      raise
    finally:
      for stop_signal, previous_handler in previous_signal_handlers.items():
        signal.signal(stop_signal, previous_handler)
      csv_logger.close()
      jsonl_logger.close()
      text_logger.close()
