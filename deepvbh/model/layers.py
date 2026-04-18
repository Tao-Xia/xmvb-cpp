"""Reusable NNX transformer layers for the pair-space DeepVBH model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class PairFeatureProjector(nnx.Module):
  """Projects local pair-token features into the model embedding space.

  Args:
    d_model: Token embedding width.
    feature_dim: Input pair-feature width.
    rngs: NNX random state container.

  The module maps pair features with shape `[batch, pair_count, feature_dim]`
  into token states with shape `[batch, pair_count, d_model]`.
  """

  def __init__(self, d_model: int, feature_dim: int, rngs: nnx.Rngs):
    self.proj = nnx.Linear(feature_dim, d_model, rngs=rngs)

  def __call__(self, pair_features: jax.Array) -> jax.Array:
    """Projects pair features into token embeddings.

    Args:
      pair_features: Pair-feature tensor with shape `[batch, pair_count,
        feature_dim]`.

    Returns:
      Token embeddings with shape `[batch, pair_count, d_model]`.
    """

    return jax.nn.gelu(self.proj(pair_features))


class PairRelationBias(nnx.Module):
  """Projects pair-relation scalars into multi-head attention biases.

  Args:
    num_heads: Attention head count.
    rngs: NNX random state container.

  The input relation matrix has shape `[batch, pair_count, pair_count]`. The
  returned bias tensor has shape `[batch, num_heads, pair_count, pair_count]`.
  """

  def __init__(self, num_heads: int, rngs: nnx.Rngs):
    self.proj = nnx.Linear(1, num_heads, rngs=rngs)

  def __call__(self, pair_relations: jax.Array) -> jax.Array:
    """Builds attention biases from the pair-relation matrix.

    Args:
      pair_relations: Pair-relation tensor with shape `[batch, pair_count,
        pair_count]`.

    Returns:
      Bias tensor with shape `[batch, num_heads, pair_count, pair_count]`.
    """

    bias = self.proj(pair_relations[..., None])
    bias = 0.5 * (bias + jnp.swapaxes(bias, -3, -2))
    return jnp.moveaxis(bias, -1, -3)


class PairRelationValue(nnx.Module):
  """Projects pair relations into pair-conditioned attention values.

  Args:
    head_dim: Per-head hidden size.
    rngs: NNX random state container.

  The module maps a relation matrix with shape `[batch, pair_count, pair_count]`
  to pair-conditioned values with shape `[batch, pair_count, pair_count,
  head_dim]`.
  """

  def __init__(self, head_dim: int, rngs: nnx.Rngs):
    self.proj = nnx.Linear(1, head_dim, rngs=rngs)

  def __call__(self, pair_relations: jax.Array) -> jax.Array:
    """Projects pair relations into value features.

    Args:
      pair_relations: Pair-relation tensor with shape `[batch, pair_count,
        pair_count]`.

    Returns:
      Relation value tensor with shape `[batch, pair_count, pair_count,
      head_dim]`.
    """

    return self.proj(pair_relations[..., None])


class PairTokenAttention(nnx.Module):
  """Self-attention over pair tokens with complete pair-space relations.

  Args:
    d_model: Token embedding width.
    num_heads: Attention head count.
    rngs: NNX random state container.

  The module consumes token states with shape `[batch, pair_count, d_model]`
  and relation tensors derived from the complete pair-space ERI.
  """

  def __init__(self, d_model: int, num_heads: int, rngs: nnx.Rngs):
    if d_model % num_heads != 0:
      raise ValueError(
          f"d_model={d_model} must be divisible by num_heads={num_heads}"
      )
    self.d_model = d_model
    self.num_heads = num_heads
    self.head_dim = d_model // num_heads
    self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.k_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.v_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
    self.relation_value = PairRelationValue(self.head_dim, rngs=rngs)

  def split_heads(self, values: jax.Array) -> jax.Array:
    """Reshapes the model dimension into `[num_heads, head_dim]`."""

    return values.reshape(*values.shape[:-1], self.num_heads, self.head_dim)

  def merge_heads(self, values: jax.Array) -> jax.Array:
    """Restores the merged model dimension."""

    return values.reshape(*values.shape[:-2], self.d_model)

  def __call__(
      self,
      token_states: jax.Array,
      pair_relations: jax.Array,
      relation_bias: jax.Array,
      attn_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Applies relation-aware self-attention to pair tokens.

    Args:
      token_states: Token states with shape `[batch, pair_count, d_model]`.
      pair_relations: Relation tensor with shape `[batch, pair_count,
        pair_count]`.
      relation_bias: Bias tensor with shape `[batch, num_heads, pair_count,
        pair_count]`.
      attn_mask: Optional boolean attention mask with shape `[batch, 1,
        pair_count, pair_count]`.

    Returns:
      Updated token states with shape `[batch, pair_count, d_model]`.
    """

    query = self.split_heads(self.q_proj(token_states))
    key = self.split_heads(self.k_proj(token_states))
    value = self.split_heads(self.v_proj(token_states))

    logits = jnp.einsum("bphd,bqhd->bhpq", query, key) + relation_bias
    if attn_mask is not None:
      logits = jnp.where(attn_mask, logits, -1.0e9)

    attn_weights = jax.nn.softmax(logits, axis=-1)
    token_out = jnp.einsum("bhpq,bqhd->bphd", attn_weights, value)

    relation_value = self.relation_value(pair_relations)
    relation_out = jnp.einsum("bhpq,bpqd->bphd", attn_weights, relation_value)
    return self.out_proj(self.merge_heads(token_out + relation_out))


class PairTokenTransformerBlock(nnx.Module):
  """Pre-layer-normalized transformer block for pair-space tokens.

  Args:
    d_model: Token embedding width.
    num_heads: Attention head count.
    rngs: NNX random state container.
    mlp_hidden_dim: Feed-forward hidden width.

  The block preserves the token shape `[batch, pair_count, d_model]`.
  """

  def __init__(
      self,
      d_model: int,
      num_heads: int,
      rngs: nnx.Rngs,
      mlp_hidden_dim: int | None = None,
  ):
    hidden_dim = 4 * d_model if mlp_hidden_dim is None else mlp_hidden_dim
    self.attn_norm = nnx.LayerNorm(d_model, rngs=rngs)
    self.attention = PairTokenAttention(d_model, num_heads, rngs=rngs)
    self.mlp_norm = nnx.LayerNorm(d_model, rngs=rngs)
    self.mlp_gate = nnx.Linear(d_model, hidden_dim, rngs=rngs)
    self.mlp_value = nnx.Linear(d_model, hidden_dim, rngs=rngs)
    self.mlp_out = nnx.Linear(hidden_dim, d_model, rngs=rngs)

  def __call__(
      self,
      token_states: jax.Array,
      pair_relations: jax.Array,
      relation_bias: jax.Array,
      attn_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Applies one transformer block to pair tokens.

    Args:
      token_states: Token states with shape `[batch, pair_count, d_model]`.
      pair_relations: Relation tensor with shape `[batch, pair_count,
        pair_count]`.
      relation_bias: Bias tensor with shape `[batch, num_heads, pair_count,
        pair_count]`.
      attn_mask: Optional boolean attention mask with shape `[batch, 1,
        pair_count, pair_count]`.

    Returns:
      Updated token states with shape `[batch, pair_count, d_model]`.
    """

    token_states = token_states + self.attention(
        self.attn_norm(token_states),
        pair_relations,
        relation_bias,
        attn_mask=attn_mask,
    )

    normed_states = self.mlp_norm(token_states)
    gated_hidden = jax.nn.gelu(self.mlp_gate(normed_states))
    mixed_hidden = self.mlp_value(normed_states)
    return token_states + self.mlp_out(gated_hidden * mixed_hidden)


NodeInit = PairFeatureProjector
EdgeBias = PairRelationBias
PairBiasedAttention = PairTokenAttention
TransformerBlock = PairTokenTransformerBlock
PairBiasedTransformerBlock = PairTokenTransformerBlock
