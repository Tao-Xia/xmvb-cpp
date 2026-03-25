"""NNX Pair-Biased Transformer for VB Hamiltonian prediction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class NodeInit(nnx.Module):
  """Extract diagonal pair features and project them to node embeddings."""

  def __init__(self, d_model: int, rngs: nnx.Rngs):
    
    self.proj = nnx.Linear(6, d_model, rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:

    diagonal_features = jnp.diagonal(x, axis1=1, axis2=2)
    node_features = jnp.swapaxes(diagonal_features, -1, -2)
    hidden = jax.nn.gelu(self.proj(node_features))

    return hidden


class EdgeBias(nnx.Module):
  """Map pair features to symmetric multi-head attention biases."""

  def __init__(self, num_heads: int, rngs: nnx.Rngs):
    self.num_heads = num_heads
    self.proj = nnx.Linear(6, num_heads, rngs=rngs)

  def __call__(self, x: jax.Array) -> jax.Array:

    bias = self.proj(x)
    bias = 0.5 * (bias + jnp.swapaxes(bias, -3, -2))
    bias = jnp.moveaxis(bias, -1, -3)

    return bias


class PairBiasedAttention(nnx.Module):
  """Self-attention layer with an externally supplied pair bias."""

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
    self.edge_v_proj = nnx.Linear(6, self.head_dim, rngs=rngs)

  def _split_heads(self, x: jax.Array) -> jax.Array:
    return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

  def _merge_heads(self, x: jax.Array) -> jax.Array:
    return x.reshape(*x.shape[:-2], self.d_model)

  def __call__(self, x: jax.Array, 
               x_edge: jax.Array, 
               edge_bias: jax.Array, 
               attn_mask: jax.Array | None = None) -> jax.Array:
      
      q = self._split_heads(self.q_proj(x))
      k = self._split_heads(self.k_proj(x))
      v = self._split_heads(self.v_proj(x))
      
      logits = jnp.einsum('bnhd,bmhd->bhnm', q, k)
      logits = logits + edge_bias
      if attn_mask is not None:
          logits = jnp.where(attn_mask, logits, -1e9)
      
      attn_weights = jax.nn.softmax(logits, axis=-1) # (batch, heads, N, N)
      
      #  Node Value 
      node_out = jnp.einsum('bhnm,bmhd->bnhd', attn_weights, v)
      
      edge_v = self.edge_v_proj(x_edge) # (batch, N, N, head_dim)
      edge_out = jnp.einsum('bhnm,bnmd->bnhd', attn_weights, edge_v)

      attn_out = self._merge_heads(node_out + edge_out) 
      
      return self.out_proj(attn_out)


class TransformerBlock(nnx.Module):
  """Pre-LayerNorm pair-biased Transformer block."""

  def __init__(
      self,
      d_model: int,
      num_heads: int,
      rngs: nnx.Rngs,
      mlp_hidden_dim: int | None = None,
  ):
    mlp_hidden_dim = mlp_hidden_dim or (4 * d_model)
    self.attn_norm = nnx.LayerNorm(d_model, rngs=rngs)
    self.attention = PairBiasedAttention(d_model, num_heads, rngs=rngs)
    self.mlp_norm = nnx.LayerNorm(d_model, rngs=rngs)
    self.mlp_wi_0 = nnx.Linear(d_model, mlp_hidden_dim, rngs=rngs)
    self.mlp_wi_1 = nnx.Linear(d_model, mlp_hidden_dim, rngs=rngs)
    self.mlp_out = nnx.Linear(mlp_hidden_dim, d_model, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      edge_bias: jax.Array,
      attn_mask: jax.Array | None = None,
  ) -> jax.Array:

    x = x + self.attention(self.attn_norm(x), edge_bias, attn_mask=attn_mask)

    normed_x = self.mlp_norm(x)
    gate = jax.nn.gelu(self.mlp_wi_0(normed_x))
    transform = self.mlp_wi_1(normed_x)
    x = x + self.mlp_out(gate * transform)

    return x


PairBiasedTransformerBlock = TransformerBlock


class VBHamiltonianPredictor(nnx.Module):
  """Permutation-invariant VB Hamiltonian predictor."""

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
    readout_hidden_dim = readout_hidden_dim or d_model
    self.node_init = NodeInit(d_model, rngs=rngs)
    self.edge_bias = EdgeBias(num_heads, rngs=rngs)
    self.transformer_blocks = nnx.List(
        [
            TransformerBlock(
                d_model,
                num_heads,
                rngs=rngs,
                mlp_hidden_dim=mlp_hidden_dim,
            )
            for _ in range(num_layers)
        ]
    )
    self.final_norm = nnx.LayerNorm(d_model, rngs=rngs)
    self.readout_hidden = nnx.Linear(d_model, readout_hidden_dim, rngs=rngs)
    self.readout_out = nnx.Linear(readout_hidden_dim, 1, rngs=rngs)
    self.overlap_proj = nnx.Linear(3, readout_hidden_dim, rngs=rngs)
    self.condition_hidden = nnx.Linear(
        2 * readout_hidden_dim, readout_hidden_dim, rngs=rngs
    )
    self.condition_out = nnx.Linear(readout_hidden_dim, 1, rngs=rngs)

  def __call__(
      self,
      x: jax.Array,
      structure_overlap: jax.Array,
      node_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Predict a single VB Hamiltonian element for each batch item."""
    attention_mask = None
    if node_mask is not None:
      attention_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]

    hidden = self.node_init(x)
    edge_bias = self.edge_bias(x)

    for block in self.transformer_blocks:
      hidden = block(hidden, edge_bias, attn_mask=attention_mask)

    hidden = self.final_norm(hidden)

    if node_mask is not None:
      hidden = jnp.where(node_mask[..., None], hidden, 0.0)

    pooled = jnp.sum(hidden, axis=1)
    
    readout_hidden = jax.nn.gelu(self.readout_hidden(pooled))

    overlap_sign = jnp.sign(structure_overlap)

    overlap_features = jnp.stack(
        [
            structure_overlap,
            jnp.abs(structure_overlap),
            overlap_sign,
        ],
        axis=-1,
    )

    overlap_hidden = jax.nn.gelu(self.overlap_proj(overlap_features))

    conditioned_hidden = jax.nn.gelu(
        self.condition_hidden(jnp.concatenate([readout_hidden, overlap_hidden], axis=-1))
    )

    magnitude = jax.nn.softplus(self.condition_out(conditioned_hidden)) + 1e-8
    output = -overlap_sign[:, None] * magnitude

    return output
