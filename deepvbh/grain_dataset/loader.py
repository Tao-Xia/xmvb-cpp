"""Trace pair loading backed by reusable sample and step caches."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cache import TraceStaticCache, TraceStepCache
from .reader import TraceArrayReader
from .schema import TracePairRecord


@dataclass
class TracePairLoader:
  """Loads one trace-pair record into pair-space arrays for the model."""

  max_active_orbitals: int
  feature_dtype: str = "float32"
  target_dtype: str = "float32"
  static_cache_entries: int | None = 128
  step_cache_entries: int | None = 128
  reader: TraceArrayReader = field(default_factory=TraceArrayReader)
  static_cache: TraceStaticCache = field(init=False)
  step_cache: TraceStepCache = field(init=False)

  def __post_init__(self) -> None:
    self.static_cache = TraceStaticCache(
        max_active_orbitals=self.max_active_orbitals,
        feature_dtype=self.feature_dtype,
        reader=self.reader,
        max_entries=self.static_cache_entries,
    )
    self.step_cache = TraceStepCache(
        max_active_orbitals=self.max_active_orbitals,
        feature_dtype=self.feature_dtype,
        target_dtype=self.target_dtype,
        reader=self.reader,
        max_entries=self.step_cache_entries,
    )

  def __call__(self, record: TracePairRecord) -> dict[str, np.ndarray]:
    static_data = self.static_cache.get(record)
    step_data = self.step_cache.get(record)

    pair_count = static_data.pair_mask.shape[0]
    pair_features = np.empty(
        (pair_count, 5),
        dtype=np.dtype(self.feature_dtype),
    )
    left = record.left_structure_index
    right = record.right_structure_index
    left_pair_occupancy = static_data.structure_pair_occupancies[left]
    right_pair_occupancy = static_data.structure_pair_occupancies[right]
    pair_features[:, :3] = step_data.pair_features
    pair_features[:, 3] = left_pair_occupancy + right_pair_occupancy
    pair_features[:, 4] = np.abs(left_pair_occupancy - right_pair_occupancy)

    return {
        "pair_features": pair_features,
        "pair_relations": step_data.pair_relations,
        "pair_mask": static_data.pair_mask,
        "structure_overlap": np.asarray(
            step_data.overlap[left, right],
            dtype=np.dtype(self.target_dtype),
        ),
        "target_hamiltonian": np.asarray(
            step_data.hamiltonian[left, right],
            dtype=np.dtype(self.target_dtype),
        ),
    }
