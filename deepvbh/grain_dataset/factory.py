"""Dataset discovery and Grain dataset construction for DeepVBH."""

from __future__ import annotations

from pathlib import Path
import random
from typing import TYPE_CHECKING, Any

from .loader import TracePairLoader
from .reader import TraceArrayReader
from .schema import (
    GrainDatasetBundle,
    GrainDatasetMetadata,
    SampleDirectoryMetadata,
    TracePairDataSource,
    TracePairRecord,
)
from .step_sampling import CompressedTrajectoryStepSelector

if TYPE_CHECKING:
  import grain  # type: ignore


def import_grain():
  """Imports Grain from either supported package layout."""

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


class GrainPairDatasetFactory:
  """Builds compressed-train and final-eval pairwise `H_{IJ}` datasets."""

  def __init__(
      self,
      *,
      batch_size: int | None = None,
      valid_fraction: float = 0.125,
      test_fraction: float = 0.125,
      split_seed: int = 0,
      shuffle_train: bool = True,
      shuffle_seed: int = 0,
      train_selected_steps: int = 5,
      nonfinal_offdiagonal_pair_fraction: float = 0.5,
      feature_dtype: str = "float32",
      target_dtype: str = "float32",
      train_num_epochs: int | None = None,
      eval_num_epochs: int | None = 1,
      drop_remainder: bool = False,
      shard_count: int = 1,
      shard_index: int = 0,
  ):
    self.batch_size = batch_size
    self.valid_fraction = valid_fraction
    self.test_fraction = test_fraction
    self.split_seed = split_seed
    self.shuffle_train = shuffle_train
    self.shuffle_seed = shuffle_seed
    self.train_selected_steps = train_selected_steps
    self.nonfinal_offdiagonal_pair_fraction = nonfinal_offdiagonal_pair_fraction
    self.feature_dtype = feature_dtype
    self.target_dtype = target_dtype
    self.train_num_epochs = train_num_epochs
    self.eval_num_epochs = eval_num_epochs
    self.drop_remainder = drop_remainder
    self.shard_count = shard_count
    self.shard_index = shard_index
    self.reader = TraceArrayReader()
    self.step_selector = CompressedTrajectoryStepSelector(
        train_selected_steps=self.train_selected_steps,
    )
    if not (0.0 < self.nonfinal_offdiagonal_pair_fraction <= 1.0):
      raise ValueError(
          "nonfinal_offdiagonal_pair_fraction must be in (0, 1]"
      )

  def resolve_roots(
      self,
      data_roots: str | Path | list[str | Path] | tuple[str | Path, ...],
  ) -> tuple[Path, ...]:
    if isinstance(data_roots, (str, Path)):
      return (Path(data_roots).resolve(),)
    if isinstance(data_roots, (list, tuple)):
      if not data_roots:
        raise ValueError("data_roots must contain at least one dataset root")
      return tuple(Path(root).resolve() for root in data_roots)
    raise TypeError(
        "data_roots must be a path or a list/tuple of paths, "
        f"got {type(data_roots).__name__}"
    )

  def discover_sample_dirs(self, data_root: Path) -> list[Path]:
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

  def load_sample_metadata(
      self,
      sample_dirs: list[Path],
  ) -> list[SampleDirectoryMetadata]:
    samples = []
    for sample_index, sample_dir in enumerate(sample_dirs):
      sample_metadata = self.reader.read_json(sample_dir / "metadata.json")
      samples.append(
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
    return samples

  def list_step_dirs(self, sample: SampleDirectoryMetadata) -> list[Path]:
    sample_dir = sample.sample_dir
    step_dirs = sorted(
        path
        for path in (sample_dir / "steps").iterdir()
        if path.is_dir() and path.name.startswith("step_")
    )
    if not step_dirs:
      raise ValueError(
          f"did not find any step directories under {sample_dir / 'steps'}"
      )
    return step_dirs

  def build_records(
      self,
      samples: list[SampleDirectoryMetadata],
      *,
      split_name: str,
  ) -> list[TracePairRecord]:
    records: list[TracePairRecord] = []
    for sample in samples:
      all_step_dirs = self.list_step_dirs(sample)
      selected_step_dirs = (
          self.step_selector.select_train_steps(sample, all_step_dirs)
          if split_name == "train"
          else self.step_selector.select_eval_steps(all_step_dirs)
      )
      final_step_dir = all_step_dirs[-1]
      for step_dir in selected_step_dirs:
        step_metadata = self.reader.read_json(step_dir / "metadata.json")
        accepted_iteration_index = int(step_metadata["accepted_iteration_index"])
        is_final_step = step_dir == final_step_dir
        selected_pairs = self.selected_pairs_for_step(
            sample,
            accepted_iteration_index,
            split_name=split_name,
            is_final_step=is_final_step,
        )
        for left_structure_index, right_structure_index in selected_pairs:
            records.append(
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
    return records

  def selected_pairs_for_step(
      self,
      sample: SampleDirectoryMetadata,
      accepted_iteration_index: int,
      *,
      split_name: str,
      is_final_step: bool,
  ) -> list[tuple[int, int]]:
    diagonal_pairs = [
        (structure_index, structure_index)
        for structure_index in range(sample.n_structures)
    ]
    offdiagonal_pairs = [
        (left_structure_index, right_structure_index)
        for left_structure_index in range(sample.n_structures)
        for right_structure_index in range(
            left_structure_index + 1,
            sample.n_structures,
        )
    ]
    if split_name != "train" or is_final_step:
      return diagonal_pairs + offdiagonal_pairs
    if not offdiagonal_pairs:
      return diagonal_pairs

    target_offdiagonal_count = int(
        round(
            len(offdiagonal_pairs) * self.nonfinal_offdiagonal_pair_fraction
        )
    )
    target_offdiagonal_count = max(
        1,
        min(target_offdiagonal_count, len(offdiagonal_pairs)),
    )
    rng = random.Random(
        self.shuffle_seed
        + sample.sample_index * 1_000_003
        + accepted_iteration_index * 10_007
    )
    sampled_offdiagonal_pairs = rng.sample(
        offdiagonal_pairs,
        target_offdiagonal_count,
    )
    return diagonal_pairs + sorted(sampled_offdiagonal_pairs)

  def split_samples(
      self,
      samples: list[SampleDirectoryMetadata],
  ) -> tuple[
      list[SampleDirectoryMetadata],
      list[SampleDirectoryMetadata],
      list[SampleDirectoryMetadata],
  ]:
    if self.valid_fraction < 0.0 or self.test_fraction < 0.0:
      raise ValueError("validation and test fractions must be non-negative")
    if self.valid_fraction + self.test_fraction >= 1.0:
      raise ValueError("validation and test fractions must sum to less than 1")

    shuffled = list(samples)
    random.Random(self.split_seed).shuffle(shuffled)
    n_samples = len(shuffled)
    n_test = min(int(round(n_samples * self.test_fraction)), n_samples)
    n_valid = min(int(round(n_samples * self.valid_fraction)), n_samples - n_test)
    test_samples = shuffled[:n_test]
    valid_samples = shuffled[n_test : n_test + n_valid]
    train_samples = shuffled[n_test + n_valid :]
    if not train_samples:
      raise ValueError("split configuration left the training split empty")
    return train_samples, valid_samples, test_samples

  def shard_records(self, records: list[TracePairRecord]) -> list[TracePairRecord]:
    if self.shard_count <= 0:
      raise ValueError("shard_count must be positive")
    if not (0 <= self.shard_index < self.shard_count):
      raise ValueError(
          f"shard_index must be in [0, {self.shard_count}), got {self.shard_index}"
      )
    if self.shard_count == 1:
      return records
    return records[self.shard_index :: self.shard_count]

  def build_grain_dataset(
      self,
      records: list[TracePairRecord],
      *,
      loader: TracePairLoader,
      shuffle: bool,
      num_epochs: int | None,
  ) -> Any:
    grain = import_grain()
    dataset = grain.MapDataset.source(TracePairDataSource(records))
    if num_epochs is not None:
      dataset = dataset.repeat(num_epochs=num_epochs)
    if shuffle and len(records) > 1:
      dataset = dataset.shuffle(seed=self.shuffle_seed)
    dataset = dataset.map(loader)
    if self.batch_size is not None:
      dataset = dataset.batch(
          batch_size=self.batch_size,
          drop_remainder=self.drop_remainder,
      )
    return dataset

  def create(
      self,
      data_roots: str | Path | list[str | Path] | tuple[str | Path, ...],
  ) -> GrainDatasetBundle:
    resolved_roots = self.resolve_roots(data_roots)
    sample_dirs: list[Path] = []
    for data_root in resolved_roots:
      sample_dirs.extend(self.discover_sample_dirs(data_root))

    samples = self.load_sample_metadata(sample_dirs)
    max_active_orbitals = max(sample.n_active_orbitals for sample in samples)
    max_packed_pairs = max_active_orbitals * (max_active_orbitals + 1) // 2
    train_samples, valid_samples, test_samples = self.split_samples(samples)

    train_records = self.build_records(train_samples, split_name="train")
    valid_records = self.build_records(valid_samples, split_name="valid")
    test_records = self.build_records(test_samples, split_name="test")

    train_records = self.shard_records(train_records)
    valid_records = self.shard_records(valid_records)
    test_records = self.shard_records(test_records)

    loader = TracePairLoader(
        max_active_orbitals=max_active_orbitals,
        feature_dtype=self.feature_dtype,
        target_dtype=self.target_dtype,
    )
    metadata = GrainDatasetMetadata(
        roots=resolved_roots,
        sample_count=len(samples),
        train_record_count=len(train_records),
        valid_record_count=len(valid_records),
        test_record_count=len(test_records),
        max_active_orbitals=max_active_orbitals,
        max_packed_pairs=max_packed_pairs,
        train_step_policy="compressed_input_delta_quantiles",
        eval_step_policy="final_only",
        train_selected_steps=self.train_selected_steps,
        nonfinal_offdiagonal_pair_fraction=self.nonfinal_offdiagonal_pair_fraction,
        pair_policy="upper_triangle_with_diagonal_and_sampled_nonfinal_offdiagonal",
    )
    return GrainDatasetBundle(
        train=self.build_grain_dataset(
            train_records,
            loader=loader,
            shuffle=self.shuffle_train,
            num_epochs=self.train_num_epochs,
        ),
        valid=self.build_grain_dataset(
            valid_records,
            loader=loader,
            shuffle=False,
            num_epochs=self.eval_num_epochs,
        ),
        test=self.build_grain_dataset(
            test_records,
            loader=loader,
            shuffle=False,
            num_epochs=self.eval_num_epochs,
        ),
        metadata=metadata,
    )


def create_grain_pair_datasets(
    data_roots: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    batch_size: int | None = None,
    valid_fraction: float = 0.125,
    test_fraction: float = 0.125,
    split_seed: int = 0,
    shuffle_train: bool = True,
    shuffle_seed: int = 0,
    train_selected_steps: int = 5,
    nonfinal_offdiagonal_pair_fraction: float = 0.5,
    feature_dtype: str = "float32",
    target_dtype: str = "float32",
    train_num_epochs: int | None = None,
    eval_num_epochs: int | None = 1,
    drop_remainder: bool = False,
    shard_count: int = 1,
    shard_index: int = 0,
) -> GrainDatasetBundle:
  """Creates compressed-train and final-eval pairwise `H_{IJ}` datasets."""

  factory = GrainPairDatasetFactory(
      batch_size=batch_size,
      valid_fraction=valid_fraction,
      test_fraction=test_fraction,
      split_seed=split_seed,
      shuffle_train=shuffle_train,
      shuffle_seed=shuffle_seed,
      train_selected_steps=train_selected_steps,
      nonfinal_offdiagonal_pair_fraction=nonfinal_offdiagonal_pair_fraction,
      feature_dtype=feature_dtype,
      target_dtype=target_dtype,
      train_num_epochs=train_num_epochs,
      eval_num_epochs=eval_num_epochs,
      drop_remainder=drop_remainder,
      shard_count=shard_count,
      shard_index=shard_index,
  )
  return factory.create(data_roots)
