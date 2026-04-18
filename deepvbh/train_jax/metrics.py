"""Metric accumulation helpers for JAX training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class MetricAccumulator:
  """Aggregates batch metrics on device and materializes them once per epoch."""

  totals: dict[str, Any] | None = None
  total_examples: int = 0

  def add(self, metrics: dict[str, jax.Array], example_count: int) -> None:
    example_weight = jnp.asarray(example_count, dtype=jnp.float32)
    weighted_metrics = jax.tree_util.tree_map(
        lambda value: value * example_weight,
        metrics,
    )
    if self.totals is None:
      self.totals = weighted_metrics
    else:
      self.totals = jax.tree_util.tree_map(
          lambda total, value: total + value,
          self.totals,
          weighted_metrics,
      )
    self.total_examples += example_count

  def compute(self) -> dict[str, float] | None:
    if self.totals is None or self.total_examples == 0:
      return None
    averaged = jax.tree_util.tree_map(
        lambda value: value / self.total_examples,
        self.totals,
    )
    host_metrics = jax.device_get(averaged)
    return {
        key: float(np.asarray(value))
        for key, value in host_metrics.items()
    }
