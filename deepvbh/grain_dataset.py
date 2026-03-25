"""Grain input pipeline for DeepVBH JAX pair-wise structure samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
  import grain  # type: ignore


def _import_grain():
  try:
    import grain  # type: ignore
  except ImportError:
    try:
      import grain.python as grain  # type: ignore
    except ImportError as exc:
      raise ImportError(
          "Grain is required for the JAX dataset pipeline. "
          "Install it with `pip install grain`."
      ) from exc
  return grain


def _read_json(path: Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _read_scalar(path: Path, dtype: np.dtype[Any]) -> Any:
  values = np.fromfile(path, dtype=dtype)
  if values.size != 1:
    raise ValueError(f"expected scalar file at {path}, got {values.size} values")
  return values[0]


def _read_vector(path: Path, dtype: np.dtype[Any]) -> np.ndarray:
  return np.fromfile(path, dtype=dtype)


def _read_vector_if_exists(
    path: Path,
    dtype: np.dtype[Any],
) -> np.ndarray | None:
  if not path.exists():
    return None
  return _read_vector(path, dtype)


def _read_matrix_column_major(
    path: Path,
    size: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
  values = _read_vector(path, dtype)
  expected_size = size * size
  if values.size != expected_size:
    raise ValueError(
        f"expected {expected_size} values for square matrix at {path}, got {values.size}"
    )
  return values.reshape((size, size), order="F")
 

def _pad_square_matrix(
    matrix: np.ndarray,
    target_size: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError(f"expected square matrix, got shape {matrix.shape}")
  if matrix.shape[0] > target_size:
    raise ValueError(
        f"cannot pad matrix of size {matrix.shape[0]} into target size {target_size}"
    )
  padded = np.zeros((target_size, target_size), dtype=dtype)
  size = matrix.shape[0]
  padded[:size, :size] = matrix.astype(dtype, copy=False)
  return padded


def _build_local_active_edge_matrix(
    structure_pair_orbital_indices: np.ndarray,
    structure_pair_mask: np.ndarray,
    structure_index: int,
    target_size: int,
    dtype: np.dtype[Any],
) -> np.ndarray:
  edge = np.zeros((target_size, target_size), dtype=dtype)
  if structure_pair_orbital_indices.size == 0:
    return edge

  structure_pairs = structure_pair_orbital_indices[structure_index]
  structure_mask = structure_pair_mask[structure_index]
  for pair_index, valid in enumerate(structure_mask):
    if not valid:
      continue
    left_local = int(structure_pairs[pair_index, 0])
    right_local = int(structure_pairs[pair_index, 1])
    if not (0 <= left_local < target_size and 0 <= right_local < target_size):
      raise ValueError(
          "structure pair orbital index is outside the active orbital window: "
          f"pair=({left_local}, {right_local}), target_size={target_size}"
      )
    edge[left_local, right_local] = 1
    edge[right_local, left_local] = 1
  return edge


@dataclass(frozen=True)
class SampleDirectoryMetadata:
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
  roots: tuple[Path, ...]
  sample_count: int
  train_record_count: int
  valid_record_count: int
  test_record_count: int
  max_active_orbitals: int
  include_diagonal: bool
  upper_triangle_only: bool
  step_selection: str
  max_steps_per_sample: int | None
  max_train_pairs_per_step: int | None


@dataclass(frozen=True)
class GrainDatasetBundle:
  train: Any
  valid: Any
  test: Any
  metadata: GrainDatasetMetadata


class TracePairDataSource:
  """Random-access source of pair-sample descriptors for Grain."""

  def __init__(self, records: list[TracePairRecord]):
    self._records = list(records)

  def __len__(self) -> int:
    return len(self._records)

  def __getitem__(self, index: int) -> TracePairRecord:
    return self._records[index]


@dataclass(frozen=True)
class TracePairLoader:
  """Loads one descriptor into padded model inputs and targets."""

  max_active_orbitals: int
  feature_dtype: str = "float32"
  target_dtype: str = "float32"

  def __call__(self, record: TracePairRecord) -> dict[str, np.ndarray]:
    feature_dtype = np.dtype(self.feature_dtype)
    target_dtype = np.dtype(self.target_dtype)
    static_dir = record.sample_dir / "static"

    static_metadata = _read_json(static_dir / "metadata.json")
    pair_layout = static_metadata["structure_pair_orbital_indices_layout"]
    pair_count = int(pair_layout[1])
    structure_pair_shape = (record.n_structures, pair_count, 2)

    local_pair_indices = _read_vector_if_exists(
        static_dir / "structure_pair_active_orbital_indices_i32.bin", np.int32
    )
    if local_pair_indices is None:
      structure_pair_orbital_indices = _read_vector(
          static_dir / "structure_pair_orbital_indices_i32.bin", np.int32
      ).reshape(structure_pair_shape)
      structure_pair_orbital_indices = (
          structure_pair_orbital_indices - record.active_start
      )
    else:
      structure_pair_orbital_indices = local_pair_indices.reshape(structure_pair_shape)
    structure_pair_mask = _read_vector(
        static_dir / "structure_pair_mask_u8.bin", np.uint8
    ).reshape((record.n_structures, pair_count)).astype(bool, copy=False)

    n = record.n_active_orbitals
    active_overlap = _read_matrix_column_major(
        record.step_dir / "active_orbital_overlap_matrix_f64.bin", n, np.float64
    )
    active_h1e = _read_matrix_column_major(
        record.step_dir / "active_one_electron_integrals_f64.bin", n, np.float64
    )
    coulomb = _read_matrix_column_major(
        record.step_dir / "coulomb_diagonal_matrix_f64.bin", n, np.float64
    )
    exchange = _read_matrix_column_major(
        record.step_dir / "exchange_diagonal_matrix_f64.bin", n, np.float64
    )
    hamiltonian = _read_matrix_column_major(
        record.step_dir / "hamiltonian_matrix_f64.bin", record.n_structures, np.float64
    )
    overlap = _read_matrix_column_major(
        record.step_dir / "overlap_matrix_f64.bin", record.n_structures, np.float64
    )
    one_electron_hamiltonian = _read_matrix_column_major(
        record.step_dir / "one_electron_hamiltonian_matrix_f64.bin",
        record.n_structures,
        np.float64,
    )
    average_structure_overlap = float(
        _read_scalar(record.step_dir / "average_structure_overlap_f64.bin", np.float64)
    )

    left_edge = _build_local_active_edge_matrix(
        structure_pair_orbital_indices,
        structure_pair_mask,
        record.left_structure_index,
        self.max_active_orbitals,
        feature_dtype,
    )
    right_edge = _build_local_active_edge_matrix(
        structure_pair_orbital_indices,
        structure_pair_mask,
        record.right_structure_index,
        self.max_active_orbitals,
        feature_dtype,
    )

    pair_input = np.stack(
        [
            _pad_square_matrix(active_h1e, self.max_active_orbitals, feature_dtype),
            _pad_square_matrix(active_overlap, self.max_active_orbitals, feature_dtype),
            _pad_square_matrix(coulomb, self.max_active_orbitals, feature_dtype),
            _pad_square_matrix(exchange, self.max_active_orbitals, feature_dtype),
            left_edge,
            right_edge,
        ],
        axis=-1,
    )
    node_mask = np.zeros((self.max_active_orbitals,), dtype=bool)
    node_mask[:n] = True

    left = record.left_structure_index
    right = record.right_structure_index
    return {
        "pair_input": pair_input,
        "node_mask": node_mask,
        "structure_overlap": np.asarray(overlap[left, right], dtype=target_dtype),
        "target_hamiltonian": np.asarray(
            hamiltonian[left, right], dtype=target_dtype
        ),
        "target_overlap": np.asarray(overlap[left, right], dtype=target_dtype),
        "target_one_electron_hamiltonian": np.asarray(
            one_electron_hamiltonian[left, right], dtype=target_dtype
        ),
        "average_structure_overlap": np.asarray(
            average_structure_overlap, dtype=target_dtype
        ),
        "left_structure_index": np.asarray(left, dtype=np.int32),
        "right_structure_index": np.asarray(right, dtype=np.int32),
        "accepted_iteration_index": np.asarray(
            record.accepted_iteration_index, dtype=np.int32
        ),
        "sample_index": np.asarray(record.sample_index, dtype=np.int32),
        "n_active_orbitals": np.asarray(record.n_active_orbitals, dtype=np.int32),
    }


def _discover_sample_dirs(data_root: Path) -> list[Path]:
  if not data_root.exists():
    raise FileNotFoundError(f"dataset root does not exist: {data_root}")
  if not data_root.is_dir():
    raise ValueError(f"dataset root is not a directory: {data_root}")
  sample_dirs = []
  for child in sorted(data_root.iterdir()):
    if not child.is_dir() or child.name == "logs":
      continue
    if (child / "metadata.json").exists() and (child / "steps").is_dir():
      sample_dirs.append(child)
  if not sample_dirs:
    raise ValueError(f"did not find any sample directories under {data_root}")
  return sample_dirs


def _load_sample_directory_metadata(
    sample_dirs: list[Path],
) -> list[SampleDirectoryMetadata]:
  result = []
  for sample_index, sample_dir in enumerate(sample_dirs):
    sample_metadata = _read_json(sample_dir / "metadata.json")
    result.append(
        SampleDirectoryMetadata(
            sample_index=sample_index,
            sample_name=sample_metadata["sample_name"],
            sample_dir=sample_dir,
            source_input_path=sample_metadata["source_input_path"],
            n_structures=int(sample_metadata["n_structures"]),
            n_total_electrons=int(sample_metadata["n_total_electrons"]),
            n_active_electrons=int(sample_metadata["n_active_electrons"]),
            n_active_orbitals=int(sample_metadata["n_active_orbitals"]),
            n_orbitals=int(sample_metadata["n_orbitals"]),
        )
    )
  return result


def _select_step_dirs(
    sample_dir: Path,
    step_selection: str,
    max_steps_per_sample: int | None,
) -> list[Path]:
  step_dirs = sorted(
      path
      for path in (sample_dir / "steps").iterdir()
      if path.is_dir() and path.name.startswith("step_")
  )
  if not step_dirs:
    raise ValueError(f"did not find any step directories under {sample_dir / 'steps'}")

  if step_selection == "all":
    selected = step_dirs
  elif step_selection == "final":
    selected = [step_dirs[-1]]
  elif step_selection == "initial":
    selected = [step_dirs[0]]
  else:
    raise ValueError(
        "step_selection must be one of {'all', 'final', 'initial'}, "
        f"got {step_selection!r}"
    )

  if max_steps_per_sample is not None:
    if max_steps_per_sample <= 0:
      raise ValueError("max_steps_per_sample must be positive")
    selected = selected[:max_steps_per_sample]
  return selected


def _build_records_for_samples(
    samples: list[SampleDirectoryMetadata],
    *,
    include_diagonal: bool,
    upper_triangle_only: bool,
    step_selection: str,
    max_steps_per_sample: int | None,
    max_pairs_per_step: int | None = None,
    pair_sample_seed: int = 0,
) -> list[TracePairRecord]:
  records: list[TracePairRecord] = []
  for sample in samples:
    for step_dir in _select_step_dirs(
        sample.sample_dir,
        step_selection=step_selection,
        max_steps_per_sample=max_steps_per_sample,
    ):
      step_metadata = _read_json(step_dir / "metadata.json")
      accepted_iteration_index = int(step_metadata["accepted_iteration_index"])
      step_records: list[TracePairRecord] = []
      for left_structure_index in range(sample.n_structures):
        for right_structure_index in range(sample.n_structures):
          if not include_diagonal and left_structure_index == right_structure_index:
            continue
          if upper_triangle_only and left_structure_index > right_structure_index:
            continue
          step_records.append(
              TracePairRecord(
                  sample_index=sample.sample_index,
                  sample_name=sample.sample_name,
                  sample_dir=sample.sample_dir,
                  step_dir=step_dir,
                  accepted_iteration_index=accepted_iteration_index,
                  left_structure_index=left_structure_index,
                  right_structure_index=right_structure_index,
                  n_structures=sample.n_structures,
                  n_active_orbitals=sample.n_active_orbitals,
                  n_orbitals=sample.n_orbitals,
                  active_start=sample.active_start,
                  source_input_path=sample.source_input_path,
              )
          )
      if max_pairs_per_step is not None and len(step_records) > max_pairs_per_step:
        rng = random.Random(
            pair_sample_seed
            + sample.sample_index * 1_000_003
            + accepted_iteration_index * 10_007
        )
        step_records = rng.sample(step_records, max_pairs_per_step)
      records.extend(step_records)
  return records


def _split_samples(
    samples: list[SampleDirectoryMetadata],
    *,
    valid_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[SampleDirectoryMetadata], list[SampleDirectoryMetadata], list[SampleDirectoryMetadata]]:
  if valid_fraction < 0.0 or test_fraction < 0.0:
    raise ValueError("validation and test fractions must be non-negative")
  if valid_fraction + test_fraction >= 1.0:
    raise ValueError("validation and test fractions must sum to less than 1")
  shuffled = list(samples)
  random.Random(seed).shuffle(shuffled)
  n_samples = len(shuffled)
  n_test = int(round(n_samples * test_fraction))
  n_valid = int(round(n_samples * valid_fraction))
  n_test = min(n_test, n_samples)
  n_valid = min(n_valid, n_samples - n_test)
  test_samples = shuffled[:n_test]
  valid_samples = shuffled[n_test : n_test + n_valid]
  train_samples = shuffled[n_test + n_valid :]
  if not train_samples:
    raise ValueError("split configuration left the training split empty")
  return train_samples, valid_samples, test_samples


def _shard_records_for_jax_processes(
    records: list[TracePairRecord],
    *,
    shard_count: int,
    shard_index: int,
) -> list[TracePairRecord]:
  if shard_count <= 0:
    raise ValueError("shard_count must be positive")
  if not (0 <= shard_index < shard_count):
    raise ValueError(
        f"shard_index must be in [0, {shard_count}), got {shard_index}"
    )
  return records[shard_index::shard_count]


def _build_grain_dataset(
    records: list[TracePairRecord],
    *,
    loader: TracePairLoader,
    batch_size: int | None,
    shuffle: bool,
    shuffle_seed: int,
    num_epochs: int | None,
    drop_remainder: bool,
) -> Any:
  grain = _import_grain()
  dataset = grain.MapDataset.source(TracePairDataSource(records))
  if num_epochs is not None:
    dataset = dataset.repeat(num_epochs=num_epochs)
  if shuffle and len(records) > 1:
    dataset = dataset.shuffle(seed=shuffle_seed)
  dataset = dataset.map(loader)
  if batch_size is not None:
    dataset = dataset.batch(
        batch_size=batch_size, drop_remainder=drop_remainder
    )
  return dataset


def create_grain_pair_datasets(
    data_roots: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    batch_size: int | None = None,
    valid_fraction: float = 0.125,
    test_fraction: float = 0.125,
    split_seed: int = 0,
    shuffle_train: bool = True,
    shuffle_seed: int = 0,
    include_diagonal: bool = True,
    upper_triangle_only: bool = False,
    step_selection: str = "all",
    max_steps_per_sample: int | None = None,
    max_train_pairs_per_step: int | None = None,
    pair_sample_seed: int = 0,
    feature_dtype: str = "float32",
    target_dtype: str = "float32",
    train_num_epochs: int | None = None,
    eval_num_epochs: int | None = 1,
    drop_remainder: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
) -> GrainDatasetBundle:
  """Creates train/valid/test Grain datasets for the pair-biased model."""

  if isinstance(data_roots, (str, Path)):
    resolved_roots = (Path(data_roots).resolve(),)
  elif isinstance(data_roots, (list, tuple)):
    if not data_roots:
      raise ValueError("data_roots must contain at least one dataset root")
    resolved_roots = tuple(Path(root).resolve() for root in data_roots)
  else:
    raise TypeError(
        "data_roots must be a path or a list/tuple of paths, "
        f"got {type(data_roots).__name__}"
    )

  sample_dirs: list[Path] = []
  for data_root in resolved_roots:
    sample_dirs.extend(_discover_sample_dirs(data_root))
  samples = _load_sample_directory_metadata(sample_dirs)
  max_active_orbitals = max(sample.n_active_orbitals for sample in samples)
  train_samples, valid_samples, test_samples = _split_samples(
      samples,
      valid_fraction=valid_fraction,
      test_fraction=test_fraction,
      seed=split_seed,
  )

  build_records_kwargs = dict(
      include_diagonal=include_diagonal,
      upper_triangle_only=upper_triangle_only,
      step_selection=step_selection,
      max_steps_per_sample=max_steps_per_sample,
  )
  train_records = _build_records_for_samples(
      train_samples,
      max_pairs_per_step=max_train_pairs_per_step,
      pair_sample_seed=pair_sample_seed,
      **build_records_kwargs,
  )
  valid_records = _build_records_for_samples(valid_samples, **build_records_kwargs)
  test_records = _build_records_for_samples(test_samples, **build_records_kwargs)

  if shard_count > 1:
    train_records = _shard_records_for_jax_processes(
        train_records, shard_count=shard_count, shard_index=shard_index
    )
    valid_records = _shard_records_for_jax_processes(
        valid_records, shard_count=shard_count, shard_index=shard_index
    )
    test_records = _shard_records_for_jax_processes(
        test_records, shard_count=shard_count, shard_index=shard_index
    )

  loader = TracePairLoader(
      max_active_orbitals=max_active_orbitals,
      feature_dtype=feature_dtype,
      target_dtype=target_dtype,
  )
  metadata = GrainDatasetMetadata(
      roots=resolved_roots,
      sample_count=len(samples),
      train_record_count=len(train_records),
      valid_record_count=len(valid_records),
      test_record_count=len(test_records),
      max_active_orbitals=max_active_orbitals,
      include_diagonal=include_diagonal,
      upper_triangle_only=upper_triangle_only,
      step_selection=step_selection,
      max_steps_per_sample=max_steps_per_sample,
      max_train_pairs_per_step=max_train_pairs_per_step,
  )
  return GrainDatasetBundle(
      train=_build_grain_dataset(
          train_records,
          loader=loader,
          batch_size=batch_size,
          shuffle=shuffle_train,
          shuffle_seed=shuffle_seed,
          num_epochs=train_num_epochs,
          drop_remainder=drop_remainder,
      ),
      valid=_build_grain_dataset(
          valid_records,
          loader=loader,
          batch_size=batch_size,
          shuffle=False,
          shuffle_seed=shuffle_seed,
          num_epochs=eval_num_epochs,
          drop_remainder=drop_remainder,
      ),
      test=_build_grain_dataset(
          test_records,
          loader=loader,
          batch_size=batch_size,
          shuffle=False,
          shuffle_seed=shuffle_seed,
          num_epochs=eval_num_epochs,
          drop_remainder=drop_remainder,
      ),
      metadata=metadata,
  )


def main() -> None:
  import argparse
  from pprint import pprint

  parser = argparse.ArgumentParser(
      description="Smoke-test the Grain pair dataset for DeepVBH."
  )
  parser.add_argument(
      "--data-root",
      type=Path,
      default=Path("artifacts/f2_scan_training_trace"),
      help="Root directory containing the exported C++ training trace.",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=4,
      help="Optional batch size for the Grain dataset.",
  )
  parser.add_argument(
      "--step-selection",
      choices=("all", "final", "initial"),
      default="all",
      help="Which accepted iterations to include from each sample.",
  )
  parser.add_argument(
      "--max-steps-per-sample",
      type=int,
      default=1,
      help="Cap how many steps to include per sample for the smoke test.",
  )
  args = parser.parse_args()

  datasets = create_grain_pair_datasets(
      args.data_root,
      batch_size=args.batch_size,
      step_selection=args.step_selection,
      max_steps_per_sample=args.max_steps_per_sample,
      valid_fraction=0.125,
      test_fraction=0.125,
      split_seed=0,
      shuffle_train=False,
      train_num_epochs=1,
      eval_num_epochs=1,
  )
  first_batch = next(iter(datasets.train))
  pprint(datasets.metadata)
  for key, value in first_batch.items():
    print(key, np.asarray(value).shape, np.asarray(value).dtype)


if __name__ == "__main__":
  main()
