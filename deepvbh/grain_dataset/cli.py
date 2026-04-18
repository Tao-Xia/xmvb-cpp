"""Command-line smoke test for the DeepVBH Grain dataset package."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import numpy as np

from .factory import create_grain_pair_datasets


def main() -> None:
  """Runs a small smoke test over the dataset loader."""

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
  args = parser.parse_args()

  datasets = create_grain_pair_datasets(
      args.data_root,
      batch_size=args.batch_size,
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
