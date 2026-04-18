"""Dataset records shared by the DeepVBH Grain pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SampleDirectoryMetadata:
  """Static metadata discovered for one exported training sample."""

  sample_index: int
  sample_name: str
  sample_dir: Path
  source_input_path: str
  n_structures: int
  n_total_electrons: int
  n_active_electrons: int
  n_active_orbitals: int
  n_orbitals: int

  @property
  def active_start(self) -> int:
    return (self.n_total_electrons - self.n_active_electrons) // 2


@dataclass(frozen=True)
class TracePairRecord:
  """One pairwise structure example at a chosen optimization step."""

  sample_index: int
  sample_name: str
  sample_dir: Path
  step_dir: Path
  accepted_iteration_index: int
  left_structure_index: int
  right_structure_index: int
  n_structures: int
  n_active_orbitals: int
  n_orbitals: int
  active_start: int
  source_input_path: str


@dataclass(frozen=True)
class GrainDatasetMetadata:
  """Summary of the dataset bundle for one training run."""

  roots: tuple[Path, ...]
  sample_count: int
  train_record_count: int
  valid_record_count: int
  test_record_count: int
  max_active_orbitals: int
  max_packed_pairs: int
  train_step_policy: str
  eval_step_policy: str
  train_selected_steps: int
  nonfinal_offdiagonal_pair_fraction: float
  pair_policy: str


@dataclass(frozen=True)
class GrainDatasetBundle:
  """Train, validation, and test Grain datasets with shared metadata."""

  train: Any
  valid: Any
  test: Any
  metadata: GrainDatasetMetadata


class TracePairDataSource:
  """Random-access record source that Grain consumes before batching."""

  def __init__(self, records: list[TracePairRecord]):
    self.records = list(records)

  def __len__(self) -> int:
    return len(self.records)

  def __getitem__(self, index: int) -> TracePairRecord:
    return self.records[index]
