"""Reusable in-memory caches for DeepVBH Grain trace loading."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .pair_space import PairSpaceLayout
from .reader import TraceArrayReader
from .schema import TracePairRecord


@dataclass(frozen=True)
class TraceStaticData:
  """Sample-level arrays shared by all pair records from one sample."""

  structure_pair_occupancies: np.ndarray
  pair_mask: np.ndarray


@dataclass(frozen=True)
class TraceStepData:
  """Step-level arrays shared by all pair records from one optimization step."""

  pair_features: np.ndarray
  pair_relations: np.ndarray
  hamiltonian: np.ndarray
  overlap: np.ndarray


@dataclass
class TraceStaticCache:
  """Caches static sample data keyed by the exported `static/` directory."""

  max_active_orbitals: int
  feature_dtype: str = "float32"
  reader: TraceArrayReader = field(default_factory=TraceArrayReader)
  pair_space: PairSpaceLayout = field(init=False)
  max_entries: int | None = 128
  entries: OrderedDict[Path, TraceStaticData] = field(default_factory=OrderedDict)

  def __post_init__(self) -> None:
    self.pair_space = PairSpaceLayout(self.max_active_orbitals)

  def get(self, record: TracePairRecord) -> TraceStaticData:
    static_dir = record.sample_dir / "static"
    cached = self.entries.get(static_dir)
    if cached is not None:
      self.entries.move_to_end(static_dir)
      return cached

    data = self.read(static_dir, record)
    self.entries[static_dir] = data
    self.trim()
    return data

  def read(self, static_dir: Path, record: TracePairRecord) -> TraceStaticData:
    feature_dtype = np.dtype(self.feature_dtype)
    static_metadata = self.reader.read_json(static_dir / "metadata.json")
    pair_count = int(static_metadata["structure_pair_orbital_indices_layout"][1])
    structure_pair_shape = (record.n_structures, pair_count, 2)

    local_pair_indices = self.reader.read_optional_vector(
        static_dir / "structure_pair_active_orbital_indices_i32.bin",
        np.int32,
    )
    if local_pair_indices is None:
      structure_pair_orbital_indices = self.reader.read_vector(
          static_dir / "structure_pair_orbital_indices_i32.bin",
          np.int32,
      ).reshape(structure_pair_shape)
      structure_pair_orbital_indices = (
          structure_pair_orbital_indices - record.active_start
      )
    else:
      structure_pair_orbital_indices = local_pair_indices.reshape(structure_pair_shape)

    structure_pair_mask = self.reader.read_vector(
        static_dir / "structure_pair_mask_u8.bin",
        np.uint8,
    ).reshape((record.n_structures, pair_count)).astype(bool, copy=False)

    structure_pair_occupancies = self.pair_space.structure_pair_occupancies(
        structure_pair_orbital_indices,
        structure_pair_mask,
        feature_dtype,
    )

    return TraceStaticData(
        structure_pair_occupancies=structure_pair_occupancies,
        pair_mask=self.pair_space.pair_mask(record.n_active_orbitals),
    )

  def trim(self) -> None:
    if self.max_entries is None:
      return
    while len(self.entries) > self.max_entries:
      self.entries.popitem(last=False)


@dataclass
class TraceStepCache:
  """Caches per-step matrices keyed by each exported `step_*` directory."""

  max_active_orbitals: int
  feature_dtype: str = "float32"
  target_dtype: str = "float32"
  reader: TraceArrayReader = field(default_factory=TraceArrayReader)
  pair_space: PairSpaceLayout = field(init=False)
  max_entries: int | None = 128
  entries: OrderedDict[Path, TraceStepData] = field(default_factory=OrderedDict)

  def __post_init__(self) -> None:
    self.pair_space = PairSpaceLayout(self.max_active_orbitals)

  def get(self, record: TracePairRecord) -> TraceStepData:
    cached = self.entries.get(record.step_dir)
    if cached is not None:
      self.entries.move_to_end(record.step_dir)
      return cached

    data = self.read(record)
    self.entries[record.step_dir] = data
    self.trim()
    return data

  def read(self, record: TracePairRecord) -> TraceStepData:
    feature_dtype = np.dtype(self.feature_dtype)
    target_dtype = np.dtype(self.target_dtype)
    n_active_orbitals = record.n_active_orbitals

    active_overlap = self.reader.read_square_matrix(
        record.step_dir / "active_orbital_overlap_matrix_f64.bin",
        n_active_orbitals,
        np.float64,
    )
    active_h1e = self.reader.read_square_matrix(
        record.step_dir / "active_one_electron_integrals_f64.bin",
        n_active_orbitals,
        np.float64,
    )
    packed_two_electron_integrals = self.reader.read_vector(
        record.step_dir / "packed_active_two_electron_integrals_f64.bin",
        np.float64,
    )

    pair_features = self.pair_space.pair_feature_matrix(
        active_h1e,
        active_overlap,
        feature_dtype,
    )
    pair_relations = self.pair_space.pair_relation_matrix(
        packed_two_electron_integrals,
        n_active_orbitals,
        feature_dtype,
    )

    return TraceStepData(
        pair_features=pair_features,
        pair_relations=pair_relations,
        hamiltonian=self.reader.read_square_matrix(
            record.step_dir / "hamiltonian_matrix_f64.bin",
            record.n_structures,
            np.float64,
        ).astype(target_dtype, copy=False),
        overlap=self.reader.read_square_matrix(
            record.step_dir / "overlap_matrix_f64.bin",
            record.n_structures,
            np.float64,
        ).astype(target_dtype, copy=False),
    )

  def trim(self) -> None:
    if self.max_entries is None:
      return
    while len(self.entries) > self.max_entries:
      self.entries.popitem(last=False)
