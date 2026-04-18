"""Readout layers for the DeepVBH pair-biased transformer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class OverlapConditionedReadout(nnx.Module):
  """Converts pooled pair-token states and overlap features into a scalar.

  Args:
    d_model: Input token width.
    readout_hidden_dim: Readout hidden width.
    rngs: NNX random state container.
  """

  def __init__(
      self,
      d_model: int,
      readout_hidden_dim: int,
      rngs: nnx.Rngs,
  ):
    self.token_hidden = nnx.Linear(d_model, readout_hidden_dim, rngs=rngs)
    self.overlap_hidden = nnx.Linear(3, readout_hidden_dim, rngs=rngs)
    self.condition_mixer = nnx.Linear(
        2 * readout_hidden_dim, readout_hidden_dim, rngs=rngs
    )
    self.scalar_out = nnx.Linear(readout_hidden_dim, 1, rngs=rngs)

  def __call__(
      self,
      pooled_tokens: jax.Array,
      structure_overlap: jax.Array,
  ) -> jax.Array:
    """Predicts the signed matrix-element magnitude.

    Args:
      pooled_tokens: Pooled pair-token state with shape `[batch, d_model]`.
      structure_overlap: Structure-overlap scalar with shape `[batch]`.

    Returns:
      Signed scalar prediction with shape `[batch, 1]`.
    """

    token_hidden = jax.nn.gelu(self.token_hidden(pooled_tokens))
    overlap_sign = jnp.sign(structure_overlap)
    overlap_features = jnp.stack(
        [
            structure_overlap,
            jnp.abs(structure_overlap),
            overlap_sign,
        ],
        axis=-1,
    )
    overlap_hidden = jax.nn.gelu(self.overlap_hidden(overlap_features))
    conditioned_hidden = jax.nn.gelu(
        self.condition_mixer(
            jnp.concatenate([token_hidden, overlap_hidden], axis=-1)
        )
    )
    magnitude = jax.nn.softplus(self.scalar_out(conditioned_hidden)) + 1.0e-8
    return -overlap_sign[:, None] * magnitude
