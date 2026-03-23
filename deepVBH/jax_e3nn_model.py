from __future__ import annotations

from typing import Any

import e3nn_jax as e3nn
import flax.linen as nn
import jax.numpy as jnp

from .jax_energy import total_energy_from_parts
from .jax_model import DeepVBHOutput, MLP
from .jax_shell_ops import (
    cartesian_to_irreps_matrix,
    dense_to_sparse_ao_coefficients,
    irreps_for_cartesian_shell,
    sparse_to_dense_ao_coefficients,
)


def _masked_mean(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    if values.shape[1] == 0:
        return jnp.zeros((values.shape[0], values.shape[-1]), dtype=values.dtype)
    weights = mask.astype(values.dtype)[..., None]
    return jnp.sum(values * weights, axis=1) / jnp.clip(
        jnp.sum(weights, axis=1),
        min=1.0,
    )


def _build_full_edge_data(
    positions: jnp.ndarray,
    cutoff: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    displacement = positions[:, None, :] - positions[None, :, :]
    distances = jnp.linalg.norm(displacement, axis=-1)
    mask = (distances > 1.0e-8) & (distances <= cutoff)
    return displacement, distances, mask


def _gaussian_radial_basis(
    distances: jnp.ndarray,
    *,
    num_basis: int,
    cutoff: float,
) -> jnp.ndarray:
    centers = jnp.linspace(
        0.0,
        cutoff,
        num_basis,
        dtype=distances.dtype,
    )
    spacing = cutoff / max(num_basis - 1, 1)
    widths = max(spacing, 1.0e-6)
    basis = jnp.exp(-jnp.square((distances[..., None] - centers) / widths))
    envelope = 0.5 * (
        jnp.cos(jnp.pi * jnp.minimum(distances, cutoff) / cutoff) + 1.0
    )
    envelope = jnp.where(distances <= cutoff, envelope, 0.0)
    return basis * envelope[..., None]


def _hidden_irreps(channels: int, lmax: int) -> e3nn.Irreps:
    parts: list[str] = []
    for degree in range(lmax + 1):
        parity = "e" if degree % 2 == 0 else "o"
        parts.append(f"{channels}x{degree}{parity}")
    return e3nn.Irreps(" + ".join(parts))


def _irrep_slices(irreps: e3nn.Irreps) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    offset = 0
    for multiplicity, irrep in irreps:
        block_width = multiplicity * irrep.dim
        slices.append((offset, offset + block_width))
        offset += block_width
    return slices


def _expand_block_scalars(
    block_scalars: jnp.ndarray,
    irrep_slices: list[tuple[int, int]],
) -> jnp.ndarray:
    if not irrep_slices:
        return jnp.zeros((block_scalars.shape[0], 0), dtype=block_scalars.dtype)
    return jnp.concatenate(
        [
            jnp.repeat(block_scalars[:, index : index + 1], stop - start, axis=-1)
            for index, (start, stop) in enumerate(irrep_slices)
        ],
        axis=-1,
    )


class _EquivariantMessageLayer(nn.Module):
    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps
    lmax: int
    num_radial: int
    radial_hidden_dim: int
    cutoff: float

    def setup(self) -> None:
        self.irreps_sh = e3nn.Irreps.spherical_harmonics(self.lmax)
        self.self_connection = e3nn.flax.Linear(e3nn.Irreps(self.irreps_out))
        self.edge_projection = e3nn.flax.Linear(e3nn.Irreps(self.irreps_out))
        self.radial_network = MLP(
            hidden_dim=self.radial_hidden_dim,
            output_dim=self.radial_hidden_dim,
            depth=2,
        )

    def __call__(
        self,
        node_features: e3nn.IrrepsArray,
        positions: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        irreps_out = e3nn.Irreps(self.irreps_out)
        updated = self.self_connection(node_features)
        if updated.irreps != irreps_out:
            padded = jnp.zeros(
                (updated.array.shape[0], irreps_out.dim),
                dtype=updated.array.dtype,
            )
            padded = padded.at[:, : updated.irreps.dim].set(updated.array)
            updated = e3nn.IrrepsArray(irreps_out, padded)

        edge_vectors, edge_distances, edge_mask = _build_full_edge_data(
            positions,
            self.cutoff,
        )
        spherical_harmonics = e3nn.spherical_harmonics(
            self.irreps_sh,
            edge_vectors,
            normalize=True,
            normalization="component",
        )
        source_features = e3nn.IrrepsArray(
            node_features.irreps,
            jnp.broadcast_to(
                node_features.array[None, :, :],
                (
                    node_features.array.shape[0],
                    node_features.array.shape[0],
                    node_features.irreps.dim,
                ),
            ),
        )
        edge_features = e3nn.tensor_product(source_features, spherical_harmonics)
        radial_features = self.radial_network(
            _gaussian_radial_basis(
                edge_distances,
                num_basis=self.num_radial,
                cutoff=self.cutoff,
            )
        )
        messages = self.edge_projection(radial_features, edge_features)
        message_array = messages.array * edge_mask[..., None].astype(messages.array.dtype)
        aggregated_array = jnp.sum(message_array, axis=1)
        degrees = jnp.sum(edge_mask.astype(message_array.dtype), axis=1)
        aggregated_array = aggregated_array / jnp.clip(degrees[:, None], min=1.0)
        aggregated = e3nn.IrrepsArray(irreps_out, aggregated_array)
        return updated + aggregated


class _CartesianShellEncoder(nn.Module):
    angular_momentum: int
    atom_irreps: e3nn.Irreps
    hidden_dim: int

    def setup(self) -> None:
        coefficient_irreps = e3nn.Irreps(
            irreps_for_cartesian_shell(self.angular_momentum)
        )
        self.coefficient_irreps = coefficient_irreps
        self.coefficient_irreps_dim = coefficient_irreps.dim
        self.projection_matrix = cartesian_to_irreps_matrix(
            self.angular_momentum,
            dtype=jnp.float32,
        )
        self.irrep_slices = _irrep_slices(self.coefficient_irreps)
        self.context_projection = e3nn.flax.Linear(self.coefficient_irreps)
        self.interaction_projection = e3nn.flax.Linear(f"{self.hidden_dim}x0e")
        self.post_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=2,
        )

    def __call__(
        self,
        shell_coefficients: jnp.ndarray,
        atom_context: e3nn.IrrepsArray,
    ) -> jnp.ndarray:
        projection_matrix = self.projection_matrix.astype(shell_coefficients.dtype)
        coefficient_irreps = e3nn.IrrepsArray(
            self.coefficient_irreps,
            shell_coefficients @ projection_matrix.T,
        )
        projected_context = self.context_projection(atom_context)
        interaction = self.interaction_projection(
            e3nn.tensor_product(coefficient_irreps, projected_context)
        ).array
        invariant_block_norms = [
            jnp.sum(
                jnp.square(coefficient_irreps.array[:, start:stop]),
                axis=-1,
                keepdims=True,
            )
            for start, stop in self.irrep_slices
        ]
        invariant_input = jnp.concatenate(
            [
                interaction,
                *invariant_block_norms,
                jnp.sum(
                    jnp.square(coefficient_irreps.array),
                    axis=-1,
                    keepdims=True,
                ),
            ],
            axis=-1,
        )
        return self.post_encoder(invariant_input)


class _CartesianShellResidualDecoder(nn.Module):
    angular_momentum: int
    atom_irreps: e3nn.Irreps
    hidden_dim: int

    def setup(self) -> None:
        coefficient_irreps = e3nn.Irreps(
            irreps_for_cartesian_shell(self.angular_momentum)
        )
        self.coefficient_irreps = coefficient_irreps
        self.coefficient_irreps_dim = coefficient_irreps.dim
        self.projection_matrix = cartesian_to_irreps_matrix(
            self.angular_momentum,
            dtype=jnp.float32,
        )
        self.irrep_slices = _irrep_slices(self.coefficient_irreps)
        self.num_irrep_blocks = len(self.irrep_slices)
        self.coefficient_projection = e3nn.flax.Linear(self.coefficient_irreps)
        self.context_projection = e3nn.flax.Linear(self.coefficient_irreps)
        self.gate_mlp = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=2 * self.num_irrep_blocks,
            depth=2,
            zero_last=False,
        )

    def __call__(
        self,
        shell_coefficients: jnp.ndarray,
        atom_context: e3nn.IrrepsArray,
        orbital_features: jnp.ndarray,
        shell_features: jnp.ndarray,
    ) -> jnp.ndarray:
        projection_matrix = self.projection_matrix.astype(shell_coefficients.dtype)
        coefficient_irreps = e3nn.IrrepsArray(
            self.coefficient_irreps,
            shell_coefficients @ projection_matrix.T,
        )
        projected_coefficients = self.coefficient_projection(coefficient_irreps).array
        projected_context = self.context_projection(atom_context).array
        invariant_block_norms = [
            jnp.sum(
                jnp.square(coefficient_irreps.array[:, start:stop]),
                axis=-1,
                keepdims=True,
            )
            for start, stop in self.irrep_slices
        ]
        invariant_input = jnp.concatenate(
            [
                orbital_features,
                shell_features,
                *invariant_block_norms,
                jnp.sum(
                    jnp.square(coefficient_irreps.array),
                    axis=-1,
                    keepdims=True,
                ),
            ],
            axis=-1,
        )
        coefficient_gates, context_gates = jnp.split(
            jnp.tanh(self.gate_mlp(invariant_input)),
            2,
            axis=-1,
        )
        coefficient_gates = _expand_block_scalars(
            coefficient_gates,
            self.irrep_slices,
        )
        context_gates = _expand_block_scalars(
            context_gates,
            self.irrep_slices,
        )
        residual_irreps = (
            projected_coefficients * coefficient_gates
            + projected_context * context_gates
        )
        return residual_irreps @ projection_matrix


class DeepVBHE3Model(nn.Module):
    max_atomic_number: int
    max_angular_momentum: int
    max_ao_local_index: int
    hidden_dim: int = 128
    pair_hidden_dim: int = 128
    atom_channels: int | None = None
    ao_scalar_channels: int | None = None
    lmax: int = 2
    num_radial: int = 8
    cutoff: float = 8.0
    predict_reference_residual: bool = True

    def setup(self) -> None:
        if self.lmax < self.max_angular_momentum:
            raise ValueError(
                "lmax must cover the highest AO angular momentum for strict shell-equivariant encoding"
            )
        self.pair_output_scale = 1.0e-2
        self.reference_output_scale = 1.0e-2

        atom_channels = self.atom_channels or max(self.hidden_dim // 2, 8)
        self.atom_input_irreps = e3nn.Irreps(f"{atom_channels}x0e")
        hidden_channels = max(self.hidden_dim // (self.lmax + 1), 4)
        self.atom_hidden_irreps = _hidden_irreps(hidden_channels, self.lmax)

        self.atomic_embedding = nn.Embed(
            num_embeddings=self.max_atomic_number + 1,
            features=atom_channels,
        )
        self.message_passing_layers = [
            _EquivariantMessageLayer(
                irreps_in=self.atom_input_irreps,
                irreps_out=self.atom_hidden_irreps,
                lmax=self.lmax,
                num_radial=self.num_radial,
                radial_hidden_dim=self.hidden_dim,
                cutoff=self.cutoff,
                name="message_layer_0",
            ),
            _EquivariantMessageLayer(
                irreps_in=self.atom_hidden_irreps,
                irreps_out=self.atom_hidden_irreps,
                lmax=self.lmax,
                num_radial=self.num_radial,
                radial_hidden_dim=self.hidden_dim,
                cutoff=self.cutoff,
                name="message_layer_1",
            ),
        ]
        self.shell_encoders = {
            str(angular_momentum): _CartesianShellEncoder(
                angular_momentum=angular_momentum,
                atom_irreps=self.atom_hidden_irreps,
                hidden_dim=self.hidden_dim,
                name=f"shell_encoder_{angular_momentum}",
            )
            for angular_momentum in range(self.max_angular_momentum + 1)
        }
        self.shell_residual_decoders = {
            str(angular_momentum): _CartesianShellResidualDecoder(
                angular_momentum=angular_momentum,
                atom_irreps=self.atom_hidden_irreps,
                hidden_dim=self.hidden_dim,
                name=f"shell_residual_decoder_{angular_momentum}",
            )
            for angular_momentum in range(self.max_angular_momentum + 1)
        }
        self.orbital_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=2,
        )
        self.rumer_pair_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=2,
        )
        self.structure_encoder = MLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            depth=2,
        )
        self.pair_head = MLP(
            hidden_dim=self.pair_hidden_dim,
            output_dim=1,
            depth=3,
            zero_last=True,
        )
        if self.predict_reference_residual:
            self.reference_residual_head = MLP(
                hidden_dim=self.hidden_dim,
                output_dim=1,
                depth=3,
                zero_last=True,
            )
        self.orbital_residual_output_scale = 1.0e-2

    def __call__(
        self,
        sample: dict[str, Any],
        orbital_value_table: jnp.ndarray | None = None,
        layout: dict[str, Any] | None = None,
    ) -> DeepVBHOutput:
        sparse_coefficients = (
            sample["orbital_value_table"]
            if orbital_value_table is None
            else orbital_value_table
        )
        dense_coefficients = sparse_to_dense_ao_coefficients(sample, sparse_coefficients)
        return self.forward_dense(
            sample,
            dense_coefficients,
            layout=layout,
        )

    def forward_dense(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: jnp.ndarray,
        *,
        layout: dict[str, Any] | None = None,
    ) -> DeepVBHOutput:
        shell_layout = self._layout_from_sample(sample) if layout is None else layout
        atom_features = self._encode_atoms(sample)
        orbital_features, shell_features = self._encode_orbitals_dense(
            sample,
            dense_orbital_coefficients,
            atom_features,
            shell_layout,
        )
        structure_features = self._encode_structures(sample, orbital_features)
        two_electron_hamiltonian = self._predict_structure_matrix(sample, structure_features)
        dense_orbital_residual = self._predict_dense_orbital_residual(
            sample,
            dense_orbital_coefficients,
            atom_features,
            orbital_features,
            shell_features,
            shell_layout,
        )
        orbital_residual = self._dense_to_sparse_orbital_residual(
            sample,
            dense_orbital_residual,
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
            exact_reference_energy=exact_reference_energy,
            reference_energy_residual=reference_energy_residual,
            total_energy=total_energy,
            structure_hamiltonian=structure_hamiltonian,
            dense_orbital_residual=dense_orbital_residual,
        )

    def forward_dense_deployment(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: jnp.ndarray,
        *,
        layout: dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        shell_layout = self._layout_from_sample(sample) if layout is None else layout
        atom_features = self._encode_atoms(sample)
        orbital_features, shell_features = self._encode_orbitals_dense(
            sample,
            dense_orbital_coefficients,
            atom_features,
            shell_layout,
        )
        structure_features = self._encode_structures(sample, orbital_features)
        two_electron_hamiltonian = self._predict_structure_matrix(sample, structure_features)
        dense_orbital_residual = self._predict_dense_orbital_residual(
            sample,
            dense_orbital_coefficients,
            atom_features,
            orbital_features,
            shell_features,
            shell_layout,
        )
        masked_dense_orbital_residual = dense_orbital_residual
        if "orbital_shell_dense_mask" in sample:
            masked_dense_orbital_residual = dense_orbital_residual * sample[
                "orbital_shell_dense_mask"
            ].astype(dense_orbital_residual.dtype)
        reference_energy_residual = self._predict_reference_energy_residual(
            orbital_features,
            structure_features,
        )
        return (
            two_electron_hamiltonian,
            reference_energy_residual,
            masked_dense_orbital_residual,
        )

    def _encode_atoms(self, sample: dict[str, Any]) -> e3nn.IrrepsArray:
        atom_embedding = self.atomic_embedding(sample["atomic_numbers"])
        atom_features = e3nn.IrrepsArray(self.atom_input_irreps, atom_embedding)
        for layer in self.message_passing_layers:
            atom_features = layer(atom_features, sample["atomic_coordinates"])
        return atom_features

    def _encode_orbitals_dense(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: jnp.ndarray,
        atom_features: e3nn.IrrepsArray,
        layout: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        n_orbitals = dense_orbital_coefficients.shape[0]
        shell_feature_list: list[jnp.ndarray] = []
        for angular_momentum, shell_start, shell_count, atom_index in zip(
            layout["shell_angular_momenta"],
            layout["shell_ao_starts"],
            layout["shell_ao_counts"],
            layout["shell_to_atom"],
            strict=True,
        ):
            shell_encoder = self.shell_encoders[str(angular_momentum)]
            shell_coefficients = dense_orbital_coefficients[
                :,
                shell_start : shell_start + shell_count,
            ]
            atom_context = e3nn.IrrepsArray(
                atom_features.irreps,
                jnp.broadcast_to(
                    atom_features.array[atom_index],
                    (n_orbitals, atom_features.irreps.dim),
                ),
            )
            shell_feature_list.append(shell_encoder(shell_coefficients, atom_context))

        shell_features = jnp.stack(shell_feature_list, axis=1)
        summed_shell_features = jnp.sum(shell_features, axis=1)
        orbital_input = jnp.concatenate(
            [
                summed_shell_features,
                sample["orbital_basis_counts"].astype(dense_orbital_coefficients.dtype)[
                    :,
                    None,
                ],
            ],
            axis=-1,
        )
        return self.orbital_encoder(orbital_input), shell_features

    def _encode_structures(
        self,
        sample: dict[str, Any],
        orbital_features: jnp.ndarray,
    ) -> jnp.ndarray:
        occupancy = sample["structure_occupancy"].astype(orbital_features.dtype)
        occupancy_context = (
            occupancy / jnp.clip(jnp.sum(occupancy, axis=-1, keepdims=True), min=1.0)
        ) @ orbital_features
        pair_context = self._pool_rumer_pairs(sample, orbital_features)
        open_shell_context = self._pool_open_shell_orbitals(sample, orbital_features)
        structure_scalars = jnp.stack(
            [
                jnp.sum(occupancy, axis=-1),
                jnp.sqrt(jnp.sum(jnp.square(occupancy), axis=-1)),
                jnp.sum(sample["structure_pair_mask"].astype(orbital_features.dtype), axis=-1),
                jnp.sum(
                    sample["structure_open_shell_mask"].astype(orbital_features.dtype),
                    axis=-1,
                ),
            ],
            axis=-1,
        )
        structure_input = jnp.concatenate(
            [
                occupancy_context,
                pair_context,
                open_shell_context,
                structure_scalars,
            ],
            axis=-1,
        )
        return self.structure_encoder(structure_input)

    def _pool_rumer_pairs(
        self,
        sample: dict[str, Any],
        orbital_features: jnp.ndarray,
    ) -> jnp.ndarray:
        pair_indices = sample["structure_pair_orbital_indices"]
        pair_mask = sample["structure_pair_mask"]
        if pair_indices.shape[1] == 0:
            return jnp.zeros((pair_indices.shape[0], self.hidden_dim), dtype=orbital_features.dtype)

        left = orbital_features[pair_indices[..., 0]]
        right = orbital_features[pair_indices[..., 1]]
        pair_input = jnp.concatenate(
            [
                left + right,
                left - right,
                left * right,
            ],
            axis=-1,
        )
        pair_features = self.rumer_pair_encoder(pair_input)
        return _masked_mean(pair_features, pair_mask)

    def _pool_open_shell_orbitals(
        self,
        sample: dict[str, Any],
        orbital_features: jnp.ndarray,
    ) -> jnp.ndarray:
        open_shell_orbitals = sample["structure_open_shell_orbitals"]
        open_shell_mask = sample["structure_open_shell_mask"]
        if open_shell_orbitals.shape[1] == 0:
            return jnp.zeros(
                (open_shell_orbitals.shape[0], self.hidden_dim),
                dtype=orbital_features.dtype,
            )
        open_shell_features = orbital_features[open_shell_orbitals]
        return _masked_mean(open_shell_features, open_shell_mask)

    def _predict_structure_matrix(
        self,
        sample: dict[str, Any],
        structure_features: jnp.ndarray,
    ) -> jnp.ndarray:
        left = structure_features[:, None, :]
        right = structure_features[None, :, :]
        overlap_matrix = sample["overlap_matrix"].astype(structure_features.dtype)
        pair_input = jnp.concatenate(
            [
                left + right,
                left - right,
                left * right,
                overlap_matrix[..., None],
                jnp.square(overlap_matrix)[..., None],
            ],
            axis=-1,
        )
        pair_scores = self.pair_head(pair_input).squeeze(-1)
        pair_scores = self.pair_output_scale * pair_scores
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
        return self.reference_output_scale * self.reference_residual_head(global_input).squeeze(-1)

    def _predict_dense_orbital_residual(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: jnp.ndarray,
        atom_features: e3nn.IrrepsArray,
        orbital_features: jnp.ndarray,
        shell_features: jnp.ndarray,
        layout: dict[str, Any],
    ) -> jnp.ndarray:
        n_orbitals = dense_orbital_coefficients.shape[0]
        dense_residual = jnp.zeros_like(dense_orbital_coefficients)
        for shell_index, (
            angular_momentum,
            shell_start,
            shell_count,
            atom_index,
        ) in enumerate(
            zip(
                layout["shell_angular_momenta"],
                layout["shell_ao_starts"],
                layout["shell_ao_counts"],
                layout["shell_to_atom"],
                strict=True,
            )
        ):
            shell_decoder = self.shell_residual_decoders[str(angular_momentum)]
            shell_coefficients = dense_orbital_coefficients[
                :,
                shell_start : shell_start + shell_count,
            ]
            atom_context = e3nn.IrrepsArray(
                atom_features.irreps,
                jnp.broadcast_to(
                    atom_features.array[atom_index],
                    (n_orbitals, atom_features.irreps.dim),
                ),
            )
            shell_residual = shell_decoder(
                shell_coefficients,
                atom_context,
                orbital_features,
                shell_features[:, shell_index, :],
            )
            dense_residual = dense_residual.at[
                :,
                shell_start : shell_start + shell_count,
            ].set(shell_residual)

        return self.orbital_residual_output_scale * dense_residual

    def _dense_to_sparse_orbital_residual(
        self,
        sample: dict[str, Any],
        dense_residual: jnp.ndarray,
    ) -> jnp.ndarray:
        sparse_residual = dense_to_sparse_ao_coefficients(sample, dense_residual)
        mask = sample["orbital_basis_mask"].astype(sparse_residual.dtype)
        return sparse_residual * mask

    def _predict_orbital_residual(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: jnp.ndarray,
        atom_features: e3nn.IrrepsArray,
        orbital_features: jnp.ndarray,
        shell_features: jnp.ndarray,
        layout: dict[str, Any],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dense_residual = self._predict_dense_orbital_residual(
            sample,
            dense_orbital_coefficients,
            atom_features,
            orbital_features,
            shell_features,
            layout,
        )
        sparse_residual = self._dense_to_sparse_orbital_residual(sample, dense_residual)
        return sparse_residual, dense_residual

    def _layout_from_sample(self, sample: dict[str, Any]) -> dict[str, tuple[int, ...]]:
        return {
            "shell_to_atom": tuple(int(value) for value in sample["shell_to_atom"].tolist()),
            "shell_angular_momenta": tuple(
                int(value) for value in sample["shell_angular_momenta"].tolist()
            ),
            "shell_ao_starts": tuple(int(value) for value in sample["shell_ao_starts"].tolist()),
            "shell_ao_counts": tuple(int(value) for value in sample["shell_ao_counts"].tolist()),
        }
