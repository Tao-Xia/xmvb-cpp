from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .energy import total_energy_from_parts


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    depth: int,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be positive")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.SiLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class DeepVBHOutput:
    orbital_features: torch.Tensor
    structure_features: torch.Tensor
    two_electron_hamiltonian: torch.Tensor
    exact_reference_energy: torch.Tensor
    reference_energy_residual: torch.Tensor
    total_energy: torch.Tensor
    structure_hamiltonian: torch.Tensor


class DeepVBHBaseline(nn.Module):
    def __init__(
        self,
        *,
        max_atomic_number: int,
        max_angular_momentum: int,
        max_ao_local_index: int,
        hidden_dim: int = 128,
        pair_hidden_dim: int = 128,
        orbital_mlp_depth: int = 2,
        pair_mlp_depth: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atomic_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim // 2)
        self.angular_embedding = nn.Embedding(max_angular_momentum + 1, hidden_dim // 4)
        self.local_index_embedding = nn.Embedding(max_ao_local_index + 1, hidden_dim // 4)

        ao_input_dim = hidden_dim + 3 + 3
        self.ao_encoder = _build_mlp(
            ao_input_dim,
            hidden_dim,
            hidden_dim,
            depth=orbital_mlp_depth,
        )
        orbital_input_dim = hidden_dim * 3 + 3
        self.orbital_encoder = _build_mlp(
            orbital_input_dim,
            hidden_dim,
            hidden_dim,
            depth=orbital_mlp_depth,
        )
        self.structure_encoder = _build_mlp(
            hidden_dim + 2,
            hidden_dim,
            hidden_dim,
            depth=orbital_mlp_depth,
        )
        self.pair_head = _build_mlp(
            hidden_dim * 3,
            pair_hidden_dim,
            1,
            depth=pair_mlp_depth,
        )
        self.reference_residual_head = _build_mlp(
            hidden_dim * 2,
            hidden_dim,
            1,
            depth=pair_mlp_depth,
        )

    def forward(
        self,
        sample: dict[str, Any],
        orbital_value_table: torch.Tensor | None = None,
    ) -> DeepVBHOutput:
        coefficients = (
            orbital_value_table
            if orbital_value_table is not None
            else sample["orbital_value_table"]
        )
        ao_features = self._encode_ao_features(sample)
        orbital_features = self._encode_orbitals(sample, coefficients, ao_features)
        structure_features = self._encode_structures(sample, orbital_features)
        two_electron_hamiltonian = self._predict_structure_matrix(structure_features)
        exact_reference_energy = sample.get(
            "one_electron_reference_energy",
            coefficients.new_zeros(()),
        )
        # Research prototype only: this residual closes the gap above exact E11.
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

    def _encode_ao_features(self, sample: dict[str, Any]) -> torch.Tensor:
        atomic_embedding = self.atomic_embedding(sample["atomic_numbers"])
        ao_atomic_embedding = atomic_embedding.index_select(0, sample["ao_to_atom"])
        ao_angular_embedding = self.angular_embedding(sample["ao_angular_momenta"])
        ao_local_embedding = self.local_index_embedding(sample["ao_shell_local_indices"])
        ao_coordinates = sample["atomic_coordinates"].index_select(0, sample["ao_to_atom"])
        ao_cartesian_exponents = sample["ao_cartesian_exponents"].to(sample["atomic_coordinates"].dtype)
        ao_input = torch.cat(
            [
                ao_atomic_embedding,
                ao_angular_embedding,
                ao_local_embedding,
                ao_coordinates,
                ao_cartesian_exponents,
            ],
            dim=-1,
        )
        return self.ao_encoder(ao_input)

    def _encode_orbitals(
        self,
        sample: dict[str, Any],
        coefficients: torch.Tensor,
        ao_features: torch.Tensor,
    ) -> torch.Tensor:
        absolute_coefficients = coefficients.abs()
        squared_coefficients = coefficients.square()

        signed_context = coefficients @ ao_features
        absolute_context = absolute_coefficients @ ao_features
        squared_context = squared_coefficients @ ao_features

        orbital_scalars = torch.stack(
            [
                squared_coefficients.sum(dim=-1),
                absolute_coefficients.sum(dim=-1),
                sample["orbital_basis_counts"].to(coefficients.dtype),
            ],
            dim=-1,
        )
        orbital_input = torch.cat(
            [signed_context, absolute_context, squared_context, orbital_scalars],
            dim=-1,
        )
        return self.orbital_encoder(orbital_input)

    def _encode_structures(
        self,
        sample: dict[str, Any],
        orbital_features: torch.Tensor,
    ) -> torch.Tensor:
        occupancy = sample["structure_occupancy"].to(orbital_features.dtype)
        normalized_occupancy = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1.0)
        structure_context = normalized_occupancy @ orbital_features
        structure_scalars = torch.stack(
            [
                occupancy.sum(dim=-1),
                occupancy.square().sum(dim=-1).sqrt(),
            ],
            dim=-1,
        )
        structure_input = torch.cat([structure_context, structure_scalars], dim=-1)
        return self.structure_encoder(structure_input)

    def _predict_structure_matrix(self, structure_features: torch.Tensor) -> torch.Tensor:
        left = structure_features[:, None, :]
        right = structure_features[None, :, :]
        pair_input = torch.cat(
            [
                left + right,
                (left - right).abs(),
                left * right,
            ],
            dim=-1,
        )
        pair_scores = self.pair_head(pair_input).squeeze(-1)
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
        return self.reference_residual_head(global_input).squeeze(-1)
