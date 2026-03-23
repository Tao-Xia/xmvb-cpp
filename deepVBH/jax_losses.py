from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp


def _lower_triangle(matrix: jnp.ndarray) -> jnp.ndarray:
    row, column = jnp.tril_indices(
        matrix.shape[-2],
        k=0,
        m=matrix.shape[-1],
    )
    return matrix[row, column]


def _masked_gradient_view(
    gradient: jnp.ndarray,
    differentiable_parameter_indices: jnp.ndarray,
) -> jnp.ndarray:
    return gradient.reshape(-1)[differentiable_parameter_indices]


def _masked_rollout_view(
    rollout: jnp.ndarray,
    differentiable_parameter_indices: jnp.ndarray,
) -> jnp.ndarray:
    flattened = rollout.reshape(rollout.shape[0], -1)
    return flattened[:, differentiable_parameter_indices]


def _rollout_step_weights(
    rollout_step_mask: jnp.ndarray,
    *,
    dtype: jnp.dtype,
    mode: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    step_mask = rollout_step_mask.astype(dtype)
    if mode == "all":
        return step_mask, jnp.clip(jnp.sum(step_mask), a_min=1.0)
    if mode == "terminal":
        valid_count = jnp.sum(rollout_step_mask.astype(jnp.int32))
        last_index = jnp.maximum(valid_count - 1, 0)
        step_weights = jax.nn.one_hot(
            last_index,
            rollout_step_mask.shape[0],
            dtype=dtype,
        )
        step_weights = jnp.where(valid_count > 0, step_weights, jnp.zeros_like(step_weights))
        return step_weights, jnp.clip(jnp.sum(step_weights), a_min=1.0)
    raise ValueError(f"unsupported rollout_loss_mode: {mode}")


def _masked_tensor_metrics(
    predicted: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mask_values = mask.astype(predicted.dtype)
    masked_predicted = predicted * mask_values
    masked_target = target * mask_values
    error = masked_predicted - masked_target
    denominator = jnp.clip(jnp.sum(mask_values), a_min=1.0)
    mse = jnp.sum(jnp.square(error)) / denominator
    mae = jnp.sum(jnp.abs(error)) / denominator
    rmse = jnp.sqrt(mse + epsilon)
    cosine_similarity, direction_loss = _safe_direction_metrics(
        masked_predicted.reshape(-1),
        masked_target.reshape(-1),
        epsilon=epsilon,
    )
    return mse, mae, rmse, cosine_similarity, direction_loss


def _safe_direction_metrics(
    predicted: jnp.ndarray,
    target: jnp.ndarray,
    *,
    epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    predicted_norm = jnp.linalg.norm(predicted)
    target_norm = jnp.linalg.norm(target)
    target_has_direction = target_norm > epsilon
    cosine_similarity = jnp.where(
        target_has_direction,
        jnp.sum(predicted * target) / (predicted_norm * target_norm + epsilon),
        jnp.ones((), dtype=predicted.dtype),
    )
    direction_loss = jnp.where(
        target_has_direction,
        1.0 - cosine_similarity,
        jnp.zeros((), dtype=predicted.dtype),
    )
    return cosine_similarity, direction_loss


@flax.struct.dataclass
class DeepVBHLossWeights:
    hamiltonian: float = 1.0
    energy: float = 0.1
    orbital_residual_mse: float = 0.0
    orbital_residual_direction: float = 0.0
    rollout_mse: float = 0.0
    rollout_direction: float = 0.0
    gradient_mse: float = 1.0
    gradient_direction: float = 0.1


@flax.struct.dataclass
class DeepVBHLossScales:
    hamiltonian: float = 1.0
    energy: float = 1.0
    orbital_residual: float = 1.0
    rollout_orbital: float = 1.0
    gradient: float = 1.0


def compute_losses(
    sample: dict[str, Any],
    output: Any,
    predicted_orbital_residual: jnp.ndarray,
    predicted_gradient: jnp.ndarray | None,
    predicted_rollout_deltas: jnp.ndarray | None = None,
    *,
    weights: DeepVBHLossWeights,
    scales: DeepVBHLossScales,
    rollout_loss_mode: str = "all",
    epsilon: float = 1.0e-12,
) -> dict[str, jnp.ndarray]:
    h2e_target = sample["two_electron_hamiltonian_matrix"]
    h2e_pred = output.two_electron_hamiltonian
    hamiltonian_error = _lower_triangle(h2e_pred) - _lower_triangle(h2e_target)
    hamiltonian_loss = jnp.mean(jnp.square(hamiltonian_error))
    hamiltonian_mae = jnp.mean(jnp.abs(hamiltonian_error))
    hamiltonian_rmse = jnp.sqrt(hamiltonian_loss + epsilon)

    energy_error = output.total_energy - sample["target_total_energy"]
    energy_loss = jnp.square(energy_error)
    energy_mae = jnp.abs(energy_error)
    energy_rmse = jnp.sqrt(energy_loss + epsilon)

    hamiltonian_scale_sq = max(float(scales.hamiltonian), epsilon) ** 2
    energy_scale_sq = max(float(scales.energy), epsilon) ** 2

    differentiable_indices = sample["differentiable_parameter_indices"]
    (
        orbital_residual_mse_loss,
        orbital_residual_mae,
        orbital_residual_rmse,
        orbital_residual_cosine_similarity,
        orbital_residual_direction_loss,
    ) = _masked_tensor_metrics(
        predicted_orbital_residual,
        sample["target_orbital_residual"],
        sample["orbital_update_mask"],
        epsilon=epsilon,
    )
    orbital_residual_scale_sq = max(float(scales.orbital_residual), epsilon) ** 2
    scaled_orbital_residual_mse_loss = orbital_residual_mse_loss / orbital_residual_scale_sq

    if predicted_rollout_deltas is None or (
        weights.rollout_mse == 0.0 and weights.rollout_direction == 0.0
    ):
        rollout_mse_loss = jnp.zeros((), dtype=h2e_pred.dtype)
        scaled_rollout_mse_loss = jnp.zeros((), dtype=h2e_pred.dtype)
        rollout_mae = jnp.zeros((), dtype=h2e_pred.dtype)
        rollout_rmse = jnp.zeros((), dtype=h2e_pred.dtype)
        rollout_direction_loss = jnp.zeros((), dtype=h2e_pred.dtype)
        rollout_cosine_similarity = jnp.zeros((), dtype=h2e_pred.dtype)
    else:
        target_rollout_deltas = sample["target_rollout_orbital_deltas"]
        model_rollout_deltas = predicted_rollout_deltas
        rollout_step_weights, valid_rollout_steps = _rollout_step_weights(
            sample["rollout_step_mask"],
            dtype=h2e_pred.dtype,
            mode=rollout_loss_mode,
        )
        rollout_orbital_mask = sample["orbital_update_mask"].astype(h2e_pred.dtype)[None, ...]
        rollout_error = (model_rollout_deltas - target_rollout_deltas) * rollout_orbital_mask
        reduction_axes = tuple(range(1, rollout_error.ndim))
        rollout_denominator = jnp.clip(
            jnp.sum(rollout_orbital_mask, axis=reduction_axes),
            a_min=1.0,
        )
        per_step_mse = jnp.sum(jnp.square(rollout_error), axis=reduction_axes) / rollout_denominator
        per_step_mae = jnp.sum(jnp.abs(rollout_error), axis=reduction_axes) / rollout_denominator
        per_step_cosine_similarity, per_step_direction_loss = jax.vmap(
            lambda predicted, target: _safe_direction_metrics(
                (predicted * sample["orbital_update_mask"].astype(predicted.dtype)).reshape(-1),
                (target * sample["orbital_update_mask"].astype(target.dtype)).reshape(-1),
                epsilon=epsilon,
            )
        )(model_rollout_deltas, target_rollout_deltas)
        rollout_mse_loss = jnp.sum(per_step_mse * rollout_step_weights) / valid_rollout_steps
        rollout_mae = jnp.sum(per_step_mae * rollout_step_weights) / valid_rollout_steps
        rollout_rmse = jnp.sqrt(rollout_mse_loss + epsilon)
        rollout_cosine_similarity = (
            jnp.sum(per_step_cosine_similarity * rollout_step_weights) / valid_rollout_steps
        )
        rollout_direction_loss = (
            jnp.sum(per_step_direction_loss * rollout_step_weights) / valid_rollout_steps
        )
        rollout_scale_sq = max(float(scales.rollout_orbital), epsilon) ** 2
        scaled_rollout_mse_loss = rollout_mse_loss / rollout_scale_sq

    if predicted_gradient is None or (
        weights.gradient_mse == 0.0 and weights.gradient_direction == 0.0
    ):
        gradient_mse_loss = jnp.zeros((), dtype=h2e_pred.dtype)
        gradient_direction_loss = jnp.zeros((), dtype=h2e_pred.dtype)
        scaled_gradient_mse_loss = jnp.zeros((), dtype=h2e_pred.dtype)
        gradient_mae = jnp.zeros((), dtype=h2e_pred.dtype)
        gradient_rmse = jnp.zeros((), dtype=h2e_pred.dtype)
        cosine_similarity = jnp.zeros((), dtype=h2e_pred.dtype)
    else:
        target_gradient = _masked_gradient_view(
            sample["target_gradient"],
            differentiable_indices,
        )
        model_gradient = _masked_gradient_view(predicted_gradient, differentiable_indices)
        gradient_error = model_gradient - target_gradient
        gradient_mse_loss = jnp.mean(jnp.square(gradient_error))
        gradient_mae = jnp.mean(jnp.abs(gradient_error))
        gradient_rmse = jnp.sqrt(gradient_mse_loss + epsilon)
        gradient_scale_sq = max(float(scales.gradient), epsilon) ** 2
        scaled_gradient_mse_loss = gradient_mse_loss / gradient_scale_sq
        cosine_similarity, gradient_direction_loss = _safe_direction_metrics(
            model_gradient,
            target_gradient,
            epsilon=epsilon,
        )

    scaled_hamiltonian_loss = hamiltonian_loss / hamiltonian_scale_sq
    scaled_energy_loss = energy_loss / energy_scale_sq

    total_loss = (
        weights.hamiltonian * scaled_hamiltonian_loss
        + weights.energy * scaled_energy_loss
        + weights.orbital_residual_mse * scaled_orbital_residual_mse_loss
        + weights.orbital_residual_direction * orbital_residual_direction_loss
        + weights.rollout_mse * scaled_rollout_mse_loss
        + weights.rollout_direction * rollout_direction_loss
        + weights.gradient_mse * scaled_gradient_mse_loss
        + weights.gradient_direction * gradient_direction_loss
    )
    display_loss = (
        weights.hamiltonian * hamiltonian_rmse
        + weights.energy * energy_rmse
        + weights.orbital_residual_mse * orbital_residual_rmse
        + weights.orbital_residual_direction * orbital_residual_direction_loss
        + weights.rollout_mse * rollout_rmse
        + weights.rollout_direction * rollout_direction_loss
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
        "orbital_residual_mse_loss": orbital_residual_mse_loss,
        "orbital_residual_mae": orbital_residual_mae,
        "orbital_residual_rmse": orbital_residual_rmse,
        "orbital_residual_direction_loss": orbital_residual_direction_loss,
        "orbital_residual_cosine_similarity": orbital_residual_cosine_similarity,
        "rollout_mse_loss": rollout_mse_loss,
        "rollout_mae": rollout_mae,
        "rollout_rmse": rollout_rmse,
        "rollout_direction_loss": rollout_direction_loss,
        "rollout_cosine_similarity": rollout_cosine_similarity,
        "gradient_mse_loss": gradient_mse_loss,
        "gradient_mae": gradient_mae,
        "gradient_rmse": gradient_rmse,
        "gradient_direction_loss": gradient_direction_loss,
        "gradient_cosine_similarity": cosine_similarity,
        "scaled_hamiltonian_loss": scaled_hamiltonian_loss,
        "scaled_energy_loss": scaled_energy_loss,
        "scaled_orbital_residual_mse_loss": scaled_orbital_residual_mse_loss,
        "scaled_rollout_mse_loss": scaled_rollout_mse_loss,
        "scaled_gradient_mse_loss": scaled_gradient_mse_loss,
    }
