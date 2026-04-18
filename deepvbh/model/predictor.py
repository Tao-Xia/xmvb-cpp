"""High-level DeepVBH model composed from the lower-level modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from .layers import PairFeatureProjector, PairRelationBias, PairTokenTransformerBlock
from .readout import OverlapConditionedReadout


class VBHamiltonianPredictor(nnx.Module):
  """Predicts one HLSP Hamiltonian matrix element from pair-space inputs.

  Args:
    d_model: Token embedding width.
    num_heads: Attention head count.
    num_layers: Transformer block count.
    rngs: NNX random state container.
    mlp_hidden_dim: Feed-forward hidden width.
    readout_hidden_dim: Readout hidden width.

  The model consumes pair-token features with shape `[batch, pair_count, 5]`,
  a complete pair-relation matrix with shape `[batch, pair_count, pair_count]`,
  and a pair mask with shape `[batch, pair_count]`. Valid tokens are pooled
  with a mask-normalized mean so the readout does not scale trivially with the
  padded pair count.
  """

  def __init__(
      self,
      d_model: int,
      num_heads: int,
      num_layers: int,
      rngs: nnx.Rngs,
      mlp_hidden_dim: int | None = None,
      readout_hidden_dim: int | None = None,
  ):
    if num_layers <= 0:
      raise ValueError(f"num_layers must be positive, got {num_layers}")
    hidden_dim = d_model if readout_hidden_dim is None else readout_hidden_dim
    self.pair_projector = PairFeatureProjector(
        d_model=d_model,
        feature_dim=5,
        rngs=rngs,
    )
    self.relation_bias = PairRelationBias(num_heads, rngs=rngs)
    self.transformer_blocks = nnx.List(
        [
            PairTokenTransformerBlock(
                d_model,
                num_heads,
                rngs=rngs,
                mlp_hidden_dim=mlp_hidden_dim,
            )
            for _ in range(num_layers)
        ]
    )
    self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)
    self.readout = OverlapConditionedReadout(
        d_model=d_model,
        readout_hidden_dim=hidden_dim,
        rngs=rngs,
    )

  def __call__(
      self,
      pair_features: jax.Array,
      pair_relations: jax.Array,
      structure_overlap: jax.Array,
      pair_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Predicts one matrix element for each batch example.

    Args:
      pair_features: Pair-token features with shape `[batch, pair_count, 5]`.
      pair_relations: Pair-space relation matrix with shape `[batch, pair_count,
        pair_count]`.
      structure_overlap: Structure-overlap scalar with shape `[batch]`.
      pair_mask: Optional boolean mask with shape `[batch, pair_count]`.

    Returns:
      Predicted signed matrix elements with shape `[batch, 1]`.
    """

    attention_mask = None
    if pair_mask is not None:
      attention_mask = pair_mask[:, None, :, None] & pair_mask[:, None, None, :]

    token_states = self.pair_projector(pair_features)
    relation_bias = self.relation_bias(pair_relations)

    for block in self.transformer_blocks:
      token_states = block(
          token_states,
          pair_relations,
          relation_bias,
          attn_mask=attention_mask,
      )

    token_states = self.final_norm(token_states)
    if pair_mask is not None:
      token_states = jnp.where(pair_mask[..., None], token_states, 0.0)
      valid_pair_count = jnp.sum(pair_mask, axis=1, keepdims=True).astype(
          token_states.dtype
      )
      pooled_tokens = jnp.sum(token_states, axis=1) / jnp.maximum(
          valid_pair_count,
          jnp.asarray(1.0, dtype=token_states.dtype),
      )
    else:
      pooled_tokens = jnp.mean(token_states, axis=1)

    return self.readout(pooled_tokens, structure_overlap)
