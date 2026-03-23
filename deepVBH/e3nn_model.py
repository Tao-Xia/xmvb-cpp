from __future__ import annotations

from typing import Any

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from torch import nn

from .energy import total_energy_from_parts
from .model import DeepVBHOutput, _build_mlp
from .shell_ops import (
    cartesian_to_irreps_matrix,
    irreps_for_cartesian_shell,
    sparse_to_dense_ao_coefficients,
    validate_cartesian_shell_layout,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 0:
        return values.new_zeros((values.shape[0], values.shape[-1]))
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _build_radius_graph(
    positions: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    displacement = positions[:, None, :] - positions[None, :, :]
    distances = torch.linalg.norm(displacement, dim=-1)
    mask = (distances > 1.0e-8) & (distances <= cutoff)
    edge_dst, edge_src = mask.nonzero(as_tuple=True)
    edge_vectors = displacement[edge_dst, edge_src]
    edge_distances = distances[edge_dst, edge_src]
    return edge_src, edge_dst, edge_vectors, edge_distances


def _gaussian_radial_basis(
    distances: torch.Tensor,
    *,
    num_basis: int,
    cutoff: float,
) -> torch.Tensor:
    centers = torch.linspace(
        0.0,
        cutoff,
        num_basis,
        device=distances.device,
        dtype=distances.dtype,
    )
    spacing = cutoff / max(num_basis - 1, 1)
    widths = max(spacing, 1.0e-6)
    basis = torch.exp(-((distances[:, None] - centers[None, :]) / widths) ** 2)
    envelope = 0.5 * (torch.cos(torch.pi * distances.clamp(max=cutoff) / cutoff) + 1.0)
    envelope = torch.where(distances <= cutoff, envelope, torch.zeros_like(envelope))
    return basis * envelope[:, None]


def _hidden_irreps(channels: int, lmax: int) -> o3.Irreps:
    parts: list[str] = []
    for degree in range(lmax + 1):
        parity = "e" if degree % 2 == 0 else "o"
        parts.append(f"{channels}x{degree}{parity}")
    return o3.Irreps(" + ".join(parts))


def _zero_last_linear(module: nn.Sequential) -> None:
    last_layer = module[-1]
    nn.init.zeros_(last_layer.weight)
    nn.init.zeros_(last_layer.bias)


def _irrep_slices(irreps: o3.Irreps) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    offset = 0
    for multiplicity, irrep in irreps:
        block_width = multiplicity * irrep.dim
        slices.append((offset, offset + block_width))
        offset += block_width
    return slices


class _EquivariantMessageLayer(nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        *,
        lmax: int,
        num_radial: int,
        radial_hidden_dim: int,
        cutoff: float,
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.num_radial = num_radial
        self.cutoff = cutoff

        self.self_connection = o3.Linear(self.irreps_in, self.irreps_out)
        self.tensor_product = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.radial_network = FullyConnectedNet(
            [num_radial, radial_hidden_dim, self.tensor_product.weight_numel],
            act=torch.nn.functional.silu,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_vectors: torch.Tensor,
        edge_distances: torch.Tensor,
    ) -> torch.Tensor:
        updated = self.self_connection(node_features)
        if edge_src.numel() == 0:
            return updated

        radial_features = _gaussian_radial_basis(
            edge_distances,
            num_basis=self.num_radial,
            cutoff=self.cutoff,
        )
        spherical_harmonics = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vectors,
            normalize=True,
            normalization="component",
        )
        weights = self.radial_network(radial_features)
        messages = self.tensor_product(
            node_features.index_select(0, edge_src),
            spherical_harmonics,
            weights,
        )
        aggregated = node_features.new_zeros((node_features.shape[0], self.irreps_out.dim))
        aggregated.index_add_(0, edge_dst, messages)

        degrees = edge_distances.new_zeros(node_features.shape[0])
        degrees.index_add_(0, edge_dst, torch.ones_like(edge_distances))
        aggregated = aggregated / degrees.clamp_min(1.0).unsqueeze(-1)
        return updated + aggregated


class _CartesianShellEncoder(nn.Module):
    def __init__(
        self,
        angular_momentum: int,
        atom_irreps: o3.Irreps,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.angular_momentum = angular_momentum
        self.coefficient_irreps = irreps_for_cartesian_shell(angular_momentum)
        self.register_buffer(
            "projection_matrix",
            cartesian_to_irreps_matrix(angular_momentum, dtype=torch.float64),
        )
        self.irrep_slices = _irrep_slices(self.coefficient_irreps)
        self.context_projection = o3.Linear(atom_irreps, self.coefficient_irreps)
        self.interaction = o3.FullyConnectedTensorProduct(
            self.coefficient_irreps,
            self.coefficient_irreps,
            o3.Irreps(f"{hidden_dim}x0e"),
        )
        self.post_encoder = _build_mlp(
            hidden_dim + self.coefficient_irreps.num_irreps + 1,
            hidden_dim,
            hidden_dim,
            depth=2,
        )

    def forward(
        self,
        shell_coefficients: torch.Tensor,
        atom_context: torch.Tensor,
    ) -> torch.Tensor:
        coefficient_irreps = shell_coefficients @ self.projection_matrix.transpose(0, 1)
        projected_context = self.context_projection(atom_context)
        interaction = self.interaction(coefficient_irreps, projected_context)
        invariant_block_norms = [
            coefficient_irreps[:, start:stop].square().sum(dim=-1, keepdim=True)
            for start, stop in self.irrep_slices
        ]
        invariant_input = torch.cat(
            [
                interaction,
                *invariant_block_norms,
                coefficient_irreps.square().sum(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return self.post_encoder(invariant_input)


class DeepVBHE3Model(nn.Module):
    def __init__(
        self,
        *,
        max_atomic_number: int,
        max_angular_momentum: int,
        max_ao_local_index: int,
        hidden_dim: int = 128,
        pair_hidden_dim: int = 128,
        atom_channels: int | None = None,
        ao_scalar_channels: int | None = None,
        lmax: int = 2,
        num_radial: int = 8,
        cutoff: float = 8.0,
    ) -> None:
        del max_ao_local_index
        del ao_scalar_channels
        super().__init__()
        if lmax < max_angular_momentum:
            raise ValueError(
                "lmax must cover the highest AO angular momentum for strict shell-equivariant encoding"
            )
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.num_radial = num_radial
        self.pair_output_scale = 1.0e-2
        self.reference_output_scale = 1.0e-2
        self._validated_layout_sample_names: set[str] = set()

        atom_channels = atom_channels or max(hidden_dim // 2, 8)
        self.atom_input_irreps = o3.Irreps(f"{atom_channels}x0e")
        self.atom_hidden_irreps = _hidden_irreps(max(hidden_dim // (lmax + 1), 4), lmax)

        self.atomic_embedding = nn.Embedding(max_atomic_number + 1, atom_channels)
        self.message_passing_layers = nn.ModuleList(
            [
                _EquivariantMessageLayer(
                    self.atom_input_irreps,
                    self.atom_hidden_irreps,
                    lmax=lmax,
                    num_radial=num_radial,
                    radial_hidden_dim=hidden_dim,
                    cutoff=cutoff,
                ),
                _EquivariantMessageLayer(
                    self.atom_hidden_irreps,
                    self.atom_hidden_irreps,
                    lmax=lmax,
                    num_radial=num_radial,
                    radial_hidden_dim=hidden_dim,
                    cutoff=cutoff,
                ),
            ]
        )
        self.shell_encoders = nn.ModuleDict(
            {
                str(angular_momentum): _CartesianShellEncoder(
                    angular_momentum,
                    self.atom_hidden_irreps,
                    hidden_dim,
                )
                for angular_momentum in range(max_angular_momentum + 1)
            }
        )
        self.orbital_encoder = _build_mlp(
            hidden_dim + 1,
            hidden_dim,
            hidden_dim,
            depth=2,
        )
        self.rumer_pair_encoder = _build_mlp(
            hidden_dim * 3,
            hidden_dim,
            hidden_dim,
            depth=2,
        )
        self.structure_encoder = _build_mlp(
            hidden_dim * 3 + 4,
            hidden_dim,
            hidden_dim,
            depth=2,
        )
        self.pair_head = _build_mlp(
            hidden_dim * 3 + 2,
            pair_hidden_dim,
            1,
            depth=3,
        )
        self.reference_residual_head = _build_mlp(
            hidden_dim * 2,
            hidden_dim,
            1,
            depth=3,
        )
        _zero_last_linear(self.pair_head)
        _zero_last_linear(self.reference_residual_head)

    def forward(
        self,
        sample: dict[str, Any],
        orbital_value_table: torch.Tensor | None = None,
    ) -> DeepVBHOutput:
        sparse_coefficients = (
            orbital_value_table
            if orbital_value_table is not None
            else sample["orbital_value_table"]
        )
        dense_coefficients = sparse_to_dense_ao_coefficients(sample, sparse_coefficients)
        return self.forward_dense(sample, dense_coefficients)

    def forward_dense(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: torch.Tensor,
    ) -> DeepVBHOutput:
        sample_name = str(sample.get("sample_name", ""))
        if sample_name not in self._validated_layout_sample_names:
            validate_cartesian_shell_layout(sample)
            self._validated_layout_sample_names.add(sample_name)
        atom_features = self._encode_atoms(sample)
        orbital_features = self._encode_orbitals_dense(
            sample,
            dense_orbital_coefficients,
            atom_features,
        )
        structure_features = self._encode_structures(sample, orbital_features)
        two_electron_hamiltonian = self._predict_structure_matrix(sample, structure_features)
        exact_reference_energy = sample.get(
            "one_electron_reference_energy",
            dense_orbital_coefficients.new_zeros(()),
        )
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
            exact_reference_energy=exact_reference_energy,
            reference_energy_residual=reference_energy_residual,
            total_energy=total_energy,
            structure_hamiltonian=structure_hamiltonian,
        )

    def _encode_atoms(self, sample: dict[str, Any]) -> torch.Tensor:
        positions = sample["atomic_coordinates"]
        edge_src, edge_dst, edge_vectors, edge_distances = _build_radius_graph(
            positions,
            self.cutoff,
        )
        atom_features = self.atomic_embedding(sample["atomic_numbers"])
        for layer in self.message_passing_layers:
            atom_features = layer(
                atom_features,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_vectors=edge_vectors,
                edge_distances=edge_distances,
            )
        return atom_features

    def _encode_orbitals_dense(
        self,
        sample: dict[str, Any],
        dense_orbital_coefficients: torch.Tensor,
        atom_features: torch.Tensor,
    ) -> torch.Tensor:
        n_orbitals = dense_orbital_coefficients.shape[0]
        shell_feature_list: list[torch.Tensor] = []
        n_shells = int(sample["shell_to_atom"].shape[0])
        for shell_index in range(n_shells):
            angular_momentum = int(sample["shell_angular_momenta"][shell_index].item())
            shell_start = int(sample["shell_ao_starts"][shell_index].item())
            shell_count = int(sample["shell_ao_counts"][shell_index].item())
            atom_index = int(sample["shell_to_atom"][shell_index].item())
            shell_encoder = self.shell_encoders[str(angular_momentum)]
            shell_coefficients = dense_orbital_coefficients[
                :,
                shell_start : shell_start + shell_count,
            ]
            atom_context = atom_features[atom_index].expand(n_orbitals, -1)
            shell_feature_list.append(shell_encoder(shell_coefficients, atom_context))

        summed_shell_features = torch.stack(shell_feature_list, dim=0).sum(dim=0)
        orbital_input = torch.cat(
            [
                summed_shell_features,
                sample["orbital_basis_counts"].to(dense_orbital_coefficients.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.orbital_encoder(orbital_input)

    def _encode_structures(
        self,
        sample: dict[str, Any],
        orbital_features: torch.Tensor,
    ) -> torch.Tensor:
        occupancy = sample["structure_occupancy"].to(orbital_features.dtype)
        occupancy_context = (
            occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ) @ orbital_features
        pair_context = self._pool_rumer_pairs(sample, orbital_features)
        open_shell_context = self._pool_open_shell_orbitals(sample, orbital_features)
        structure_scalars = torch.stack(
            [
                occupancy.sum(dim=-1),
                occupancy.square().sum(dim=-1).sqrt(),
                sample["structure_pair_mask"].to(orbital_features.dtype).sum(dim=-1),
                sample["structure_open_shell_mask"].to(orbital_features.dtype).sum(dim=-1),
            ],
            dim=-1,
        )
        structure_input = torch.cat(
            [
                occupancy_context,
                pair_context,
                open_shell_context,
                structure_scalars,
            ],
            dim=-1,
        )
        return self.structure_encoder(structure_input)

    def _pool_rumer_pairs(
        self,
        sample: dict[str, Any],
        orbital_features: torch.Tensor,
    ) -> torch.Tensor:
        pair_indices = sample["structure_pair_orbital_indices"]
        pair_mask = sample["structure_pair_mask"]
        if pair_indices.shape[1] == 0:
            return orbital_features.new_zeros((pair_indices.shape[0], self.hidden_dim))

        left = orbital_features.index_select(0, pair_indices[..., 0].reshape(-1)).reshape(
            pair_indices.shape[0],
            pair_indices.shape[1],
            self.hidden_dim,
        )
        right = orbital_features.index_select(0, pair_indices[..., 1].reshape(-1)).reshape(
            pair_indices.shape[0],
            pair_indices.shape[1],
            self.hidden_dim,
        )
        pair_input = torch.cat(
            [
                left + right,
                left - right,
                left * right,
            ],
            dim=-1,
        )
        pair_features = self.rumer_pair_encoder(pair_input)
        return _masked_mean(pair_features, pair_mask)

    def _pool_open_shell_orbitals(
        self,
        sample: dict[str, Any],
        orbital_features: torch.Tensor,
    ) -> torch.Tensor:
        open_shell_orbitals = sample["structure_open_shell_orbitals"]
        open_shell_mask = sample["structure_open_shell_mask"]
        if open_shell_orbitals.shape[1] == 0:
            return orbital_features.new_zeros((open_shell_orbitals.shape[0], self.hidden_dim))

        open_shell_features = orbital_features.index_select(
            0,
            open_shell_orbitals.reshape(-1),
        ).reshape(
            open_shell_orbitals.shape[0],
            open_shell_orbitals.shape[1],
            self.hidden_dim,
        )
        return _masked_mean(open_shell_features, open_shell_mask)

    def _predict_structure_matrix(
        self,
        sample: dict[str, Any],
        structure_features: torch.Tensor,
    ) -> torch.Tensor:
        left = structure_features[:, None, :]
        right = structure_features[None, :, :]
        overlap_matrix = sample["overlap_matrix"].to(structure_features.dtype)
        pair_input = torch.cat(
            [
                left + right,
                left - right,
                left * right,
                overlap_matrix.unsqueeze(-1),
                overlap_matrix.square().unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_scores = self.pair_head(pair_input).squeeze(-1)
        pair_scores = self.pair_output_scale * pair_scores
        return 0.5 * (pair_scores + pair_scores.transpose(0, 1))

    def _predict_reference_energy_residual(
        self,
        orbital_features: torch.Tensor,
        structure_features: torch.Tensor,
    ) -> torch.Tensor:
        global_input = torch.cat(
            [
                orbital_features.mean(dim=0),
                structure_features.mean(dim=0),
            ],
            dim=-1,
        )
        return (
            self.reference_output_scale
            * self.reference_residual_head(global_input).squeeze(-1)
        )
