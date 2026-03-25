"""Pair-biased Transformer components and Grain datasets for DeepVBH."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "EdgeBias",
    "GrainDatasetBundle",
    "GrainDatasetMetadata",
    "NodeInit",
    "PairBiasedAttention",
    "PairBiasedTransformerBlock",
    "TransformerBlock",
    "VBHamiltonianPredictor",
    "create_grain_pair_datasets",
]

if TYPE_CHECKING:
  from .grain_dataset import (
      GrainDatasetBundle,
      GrainDatasetMetadata,
      create_grain_pair_datasets,
  )
  from .model import (
      EdgeBias,
      NodeInit,
      PairBiasedAttention,
      PairBiasedTransformerBlock,
      TransformerBlock,
      VBHamiltonianPredictor,
  )


def __getattr__(name: str):
  if name in {
      "GrainDatasetBundle",
      "GrainDatasetMetadata",
      "create_grain_pair_datasets",
  }:
    from . import grain_dataset as _grain_dataset

    return getattr(_grain_dataset, name)
  if name in {
      "EdgeBias",
      "NodeInit",
      "PairBiasedAttention",
      "PairBiasedTransformerBlock",
      "TransformerBlock",
      "VBHamiltonianPredictor",
  }:
    from . import model as _pair_biased_transformer

    return getattr(_pair_biased_transformer, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
