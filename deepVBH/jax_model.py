from __future__ import annotations

from typing import Any

import flax
import flax.linen as nn
import jax.numpy as jnp

from .jax_energy import total_energy_from_parts


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int
    zero_last: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for _ in range(self.depth - 1):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.silu(x)
        kernel_init = nn.initializers.zeros_init() if self.zero_last else nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros_init()
        x = nn.Dense(
            self.output_dim,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )(x)
        return x


@flax.struct.dataclass
class DeepVBHOutput:
    orbital_features: jnp.ndarray
    structure_features: jnp.ndarray
    two_electron_hamiltonian: jnp.ndarray
    orbital_residual: jnp.ndarray
    exact_reference_energy: jnp.ndarray
    reference_energy_residual: jnp.ndarray
    total_energy: jnp.ndarray
    structure_hamiltonian: jnp.ndarray
    dense_orbital_residual: jnp.ndarray | None = None


class DeepVBHBaseline(nn.Module):
    max_atomic_number: int
    max_angular_momentum: int
    max_ao_local_index: int
    hidden_dim: int = 128
    pair_hidden_dim: int = 128
    orbital_mlp_depth: int = 2
    pair_mlp_depth: int = 3
    predict_reference_residual: bool = True

    def setup(self) -> None:
        self.atomic_embedding = nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=self.hidden_dim // 2,
        )
        self.angular_embedding = nn.Embed(
            num_embeddings=self.max_angular_momentum + 1,
            features=self.hidden_dim // 4,
        )
        self.local_index_embedding = nn.Embed(
            num_embeddings=self.max_ao_local_index + 1,
            features=self.hidden_dim // 4,
        )
        ao_input_dim = self.hidden_dim + 3 + 3
        self.ao_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=self.orbital_mlp_depth,
        )
        self.orbital_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=self.orbital_mlp_depth,
        )
        self.structure_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=self.orbital_mlp_depth,
        )
        self.pair_head = MLP(
            hidden_dim=self.pair_hidden_dim,
            output_dim=1,
            depth=self.pair_mlp_depth,
        )
        self.orbital_residual_head = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=1,
            depth=self.pair_mlp_depth,
            zero_last=False,
        )
        if self.predict_reference_residual:
            self.reference_residual_head = MLP(
                hidden_dim=self.hidden_dim,
                output_dim=1,
                depth=self.pair_mlp_depth,
            )
        self.orbital_residual_output_scale = 1.0e-2

    def __call__(
        self,
        sample: dict[str, Any],
        orbital_value_table: jnp.ndarray | None = None,
    ) -> DeepVBHOutput:
        coefficients = (
            sample["orbital_value_table"]
            if orbital_value_table is None
            else orbital_value_table
        )
        ao_features = self._encode_ao_features(sample)
        orbital_features = self._encode_orbitals(sample, coefficients, ao_features)
        structure_features = self._encode_structures(sample, orbital_features)
        two_electron_hamiltonian = self._predict_structure_matrix(structure_features)
        orbital_residual = self._predict_orbital_residual(
            sample,
            orbital_features,
            ao_features,
            coefficients,
        )
        exact_reference_energy = sample["one_electron_reference_energy"]
        reference_energy_residual = self._predict_reference_energy_residual(
            orbital_features,
            structure_features,
        )
        structure_hamiltonian = (
            sample["one_electron_hamiltonian_matrix"] + two_electron_hamiltonian
        )
        total_energy = total_energy_from_parts(
            exact_reference_energy=exact_reference_energy,
            reference_energy_residual=reference_energy_residual,
            structure_hamiltonian=structure_hamiltonian,
            overlap_matrix=sample["overlap_matrix"],
            nuclear_repulsion_energy=sample["nuclear_repulsion_energy"],
        )
        return DeepVBHOutput(
            orbital_features=orbital_features,
            structure_features=structure_features,
            two_electron_hamiltonian=two_electron_hamiltonian,
            orbital_residual=orbital_residual,
            dense_orbital_residual=None,
            exact_reference_energy=exact_reference_energy,
            reference_energy_residual=reference_energy_residual,
            total_energy=total_energy,
            structure_hamiltonian=structure_hamiltonian,
        )

    def _encode_ao_features(self, sample: dict[str, Any]) -> jnp.ndarray:
        atomic_embedding = self.atomic_embedding(sample["atomic_numbers"])
        ao_atomic_embedding = jnp.take(atomic_embedding, sample["ao_to_atom"], axis=0)
        ao_angular_embedding = self.angular_embedding(sample["ao_angular_momenta"])
        ao_local_embedding = self.local_index_embedding(sample["ao_shell_local_indices"])
        ao_coordinates = jnp.take(sample["atomic_coordinates"], sample["ao_to_atom"], axis=0)
        ao_cartesian_exponents = sample["ao_cartesian_exponents"].astype(
            sample["atomic_coordinates"].dtype
        )
        ao_input = jnp.concatenate(
            [
                ao_atomic_embedding,
                ao_angular_embedding,
                ao_local_embedding,
                ao_coordinates,
                ao_cartesian_exponents,
            ],
            axis=-1,
        )
        return self.ao_encoder(ao_input)

    def _encode_orbitals(
        self,
        sample: dict[str, Any],
        coefficients: jnp.ndarray,
        ao_features: jnp.ndarray,
    ) -> jnp.ndarray:
        absolute_coefficients = jnp.abs(coefficients)
        squared_coefficients = jnp.square(coefficients)

        signed_context = coefficients @ ao_features
        absolute_context = absolute_coefficients @ ao_features
        squared_context = squared_coefficients @ ao_features

        orbital_scalars = jnp.stack(
            [
                jnp.sum(squared_coefficients, axis=-1),
                jnp.sum(absolute_coefficients, axis=-1),
                sample["orbital_basis_counts"].astype(coefficients.dtype),
            ],
            axis=-1,
        )
        orbital_input = jnp.concatenate(
            [
                signed_context,
                absolute_context,
                squared_context,
                orbital_scalars,
            ],
            axis=-1,
        )
        return self.orbital_encoder(orbital_input)

    def _encode_structures(
        self,
        sample: dict[str, Any],
        orbital_features: jnp.ndarray,
    ) -> jnp.ndarray:
        occupancy = sample["structure_occupancy"].astype(orbital_features.dtype)
        normalized_occupancy = occupancy / jnp.clip(
            jnp.sum(occupancy, axis=-1, keepdims=True),
            min=1.0,
        )
        structure_context = normalized_occupancy @ orbital_features
        structure_scalars = jnp.stack(
            [
                jnp.sum(occupancy, axis=-1),
                jnp.sqrt(jnp.sum(jnp.square(occupancy), axis=-1)),
            ],
            axis=-1,
        )
        structure_input = jnp.concatenate(
            [structure_context, structure_scalars],
            axis=-1,
        )
        return self.structure_encoder(structure_input)

    def _predict_structure_matrix(self, structure_features: jnp.ndarray) -> jnp.ndarray:
        left = structure_features[:, None, :]
        right = structure_features[None, :, :]
        pair_input = jnp.concatenate(
            [
                left + right,
                jnp.abs(left - right),
                left * right,
            ],
            axis=-1,
        )
        pair_scores = self.pair_head(pair_input).squeeze(-1)
        return 0.5 * (pair_scores + pair_scores.T)

    def _predict_reference_energy_residual(
        self,
        orbital_features: jnp.ndarray,
        structure_features: jnp.ndarray,
    ) -> jnp.ndarray:
        if not self.predict_reference_residual:
            return jnp.zeros((), dtype=orbital_features.dtype)
        global_input = jnp.concatenate(
            [
                jnp.mean(orbital_features, axis=0),
                jnp.mean(structure_features, axis=0),
            ],
            axis=-1,
        )
        return self.reference_residual_head(global_input).squeeze(-1)

    def _predict_orbital_residual(
        self,
        sample: dict[str, Any],
        orbital_features: jnp.ndarray,
        ao_features: jnp.ndarray,
        coefficients: jnp.ndarray,
    ) -> jnp.ndarray:
        orbital_context = jnp.broadcast_to(
            orbital_features[:, None, :],
            (orbital_features.shape[0], ao_features.shape[0], orbital_features.shape[-1]),
        )
        ao_context = jnp.broadcast_to(
            ao_features[None, :, :],
            (orbital_features.shape[0], ao_features.shape[0], ao_features.shape[-1]),
        )
        pair_input = jnp.concatenate(
            [
                orbital_context + ao_context,
                orbital_context * ao_context,
                coefficients[..., None],
                jnp.abs(coefficients)[..., None],
            ],
            axis=-1,
        )
        residual = self.orbital_residual_head(pair_input).squeeze(-1)
        residual = self.orbital_residual_output_scale * residual
        mask = sample["orbital_basis_mask"].astype(residual.dtype)
        return residual * mask
