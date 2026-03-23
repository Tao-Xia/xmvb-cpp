from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .model import DeepVBHOutput


def _lower_triangle(matrix: torch.Tensor) -> torch.Tensor:
    row, column = torch.tril_indices(
        matrix.shape[-2],
        matrix.shape[-1],
        device=matrix.device,
    )
    return matrix[row, column]


def _masked_gradient_view(
    gradient: torch.Tensor,
    differentiable_parameter_indices: torch.Tensor,
) -> torch.Tensor:
    return gradient.reshape(-1).index_select(0, differentiable_parameter_indices)


@dataclass
class DeepVBHLossWeights:
    hamiltonian: float = 1.0
    energy: float = 0.1
    gradient_mse: float = 1.0
    gradient_direction: float = 0.1


@dataclass
class DeepVBHLossScales:
    hamiltonian: float = 1.0
    energy: float = 1.0
    gradient: float = 1.0


def compute_losses(
    sample: dict[str, Any],
    output: DeepVBHOutput,
    predicted_gradient: torch.Tensor | None,
    *,
    weights: DeepVBHLossWeights,
    scales: DeepVBHLossScales | None = None,
    gradient_target: str = "total",
    epsilon: float = 1.0e-12,
) -> dict[str, torch.Tensor]:
    if scales is None:
        scales = DeepVBHLossScales()
    h2e_target = sample["two_electron_hamiltonian_matrix"]
    h2e_pred = output.two_electron_hamiltonian
    hamiltonian_error = _lower_triangle(h2e_pred) - _lower_triangle(h2e_target)
    hamiltonian_loss = torch.mean(hamiltonian_error.square())
    hamiltonian_mae = torch.mean(hamiltonian_error.abs())
    hamiltonian_rmse = torch.sqrt(hamiltonian_loss + epsilon)

    energy_error = output.total_energy - sample["target_total_energy"]
    energy_loss = energy_error.square()
    energy_mae = energy_error.abs()
    energy_rmse = torch.sqrt(energy_loss + epsilon)

    hamiltonian_scale_sq = max(float(scales.hamiltonian), epsilon) ** 2
    energy_scale_sq = max(float(scales.energy), epsilon) ** 2

    if predicted_gradient is None or (
        weights.gradient_mse == 0.0 and weights.gradient_direction == 0.0
    ):
        gradient_mse_loss = h2e_pred.new_zeros(())
        gradient_direction_loss = h2e_pred.new_zeros(())
        scaled_gradient_mse_loss = h2e_pred.new_zeros(())
        gradient_mae = h2e_pred.new_zeros(())
        gradient_rmse = h2e_pred.new_zeros(())
        cosine_similarity = h2e_pred.new_zeros(())
    else:
        differentiable_indices = sample["differentiable_parameter_indices"]
        gradient_target_map = {
            "total": "sparse_orbital_energy_gradient",
            "reference": "sparse_orbital_reference_energy_gradient",
            "residual": "sparse_orbital_residual_energy_gradient",
        }
        try:
            target_gradient_key = gradient_target_map[gradient_target]
        except KeyError as error:
            raise ValueError(f"unsupported gradient target: {gradient_target}") from error
        target_gradient = _masked_gradient_view(
            sample[target_gradient_key],
            differentiable_indices,
        )
        model_gradient = _masked_gradient_view(predicted_gradient, differentiable_indices)
        gradient_error = model_gradient - target_gradient
        gradient_mse_loss = torch.mean(gradient_error.square())
        gradient_mae = torch.mean(gradient_error.abs())
        gradient_rmse = torch.sqrt(gradient_mse_loss + epsilon)
        gradient_scale_sq = max(float(scales.gradient), epsilon) ** 2
        scaled_gradient_mse_loss = gradient_mse_loss / gradient_scale_sq

        model_norm = torch.linalg.norm(model_gradient)
        target_norm = torch.linalg.norm(target_gradient)
        cosine_similarity = torch.sum(model_gradient * target_gradient) / (
            model_norm * target_norm + epsilon
        )
        gradient_direction_loss = 1.0 - cosine_similarity

    scaled_hamiltonian_loss = hamiltonian_loss / hamiltonian_scale_sq
    scaled_energy_loss = energy_loss / energy_scale_sq

    total_loss = (
        weights.hamiltonian * scaled_hamiltonian_loss
        + weights.energy * scaled_energy_loss
        + weights.gradient_mse * scaled_gradient_mse_loss
        + weights.gradient_direction * gradient_direction_loss
    )
    display_loss = (
        weights.hamiltonian * hamiltonian_rmse
        + weights.energy * energy_rmse
        + weights.gradient_mse * gradient_rmse
        + weights.gradient_direction * gradient_direction_loss
    )
    return {
        "display_loss": display_loss,
        "loss": total_loss,
        "hamiltonian_loss": hamiltonian_loss,
        "hamiltonian_mae": hamiltonian_mae,
        "hamiltonian_rmse": hamiltonian_rmse,
        "energy_loss": energy_loss,
        "energy_mae": energy_mae,
        "energy_rmse": energy_rmse,
        "gradient_mse_loss": gradient_mse_loss,
        "gradient_mae": gradient_mae,
        "gradient_rmse": gradient_rmse,
        "gradient_direction_loss": gradient_direction_loss,
        "gradient_cosine_similarity": cosine_similarity,
        "scaled_hamiltonian_loss": scaled_hamiltonian_loss,
        "scaled_energy_loss": scaled_energy_loss,
        "scaled_gradient_mse_loss": scaled_gradient_mse_loss,
    }
