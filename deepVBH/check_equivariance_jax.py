from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import e3nn_jax as e3nn
import flax
import jax
import jax.numpy as jnp
import torch

from .jax_dataset import JAXStepDataset
from .jax_e3nn_model import DeepVBHE3Model
from .jax_shell_ops import (
    dense_to_sparse_ao_coefficients,
    rotate_dense_ao_coefficients,
    sparse_to_dense_ao_coefficients,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DeepVBH JAX shell equivariance")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--pair-hidden-dim", type=int, default=64)
    parser.add_argument("--atom-channels", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument(
        "--predict-reference-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _torch_dtype_from_arg(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {value}")


def _resolve_device(device_name: str) -> jax.Device:
    platform_map = {
        "cuda": "gpu",
        "gpu": "gpu",
        "cpu": "cpu",
        "tpu": "tpu",
    }
    platform = platform_map.get(device_name, device_name)
    return jax.devices(platform)[0]


def _build_model(
    dataset: JAXStepDataset,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> DeepVBHE3Model:
    checkpoint_args = {} if checkpoint is None else dict(checkpoint.get("args", {}))
    return DeepVBHE3Model(
        max_atomic_number=max(dataset.max_atomic_number(), 1),
        max_angular_momentum=max(dataset.max_angular_momentum(), 1),
        max_ao_local_index=max(dataset.max_ao_local_index(), 1),
        hidden_dim=int(checkpoint_args.get("hidden_dim", args.hidden_dim)),
        pair_hidden_dim=int(checkpoint_args.get("pair_hidden_dim", args.pair_hidden_dim)),
        atom_channels=(
            None
            if int(checkpoint_args.get("atom_channels", args.atom_channels)) == 0
            else int(checkpoint_args.get("atom_channels", args.atom_channels))
        ),
        lmax=int(checkpoint_args.get("lmax", args.lmax)),
        num_radial=int(checkpoint_args.get("num_radial", args.num_radial)),
        cutoff=float(checkpoint_args.get("cutoff", args.cutoff)),
        predict_reference_residual=bool(
            checkpoint_args.get("predict_reference_residual", args.predict_reference_residual)
        ),
    )


def _load_checkpoint_if_present(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.checkpoint is None:
        return None
    with args.checkpoint.open("rb") as handle:
        return pickle.load(handle)


def _merge_param_tree(
    template: Any,
    loaded: Any,
) -> tuple[Any, int, int]:
    if isinstance(template, Mapping):
        loaded_mapping = loaded if isinstance(loaded, Mapping) else {}
        merged: dict[str, Any] = {}
        reused_leaves = 0
        total_leaves = 0
        for key, template_value in template.items():
            merged_value, child_reused, child_total = _merge_param_tree(
                template_value,
                loaded_mapping.get(key),
            )
            merged[key] = merged_value
            reused_leaves += child_reused
            total_leaves += child_total
        return merged, reused_leaves, total_leaves

    total_leaves = 1
    if loaded is None:
        return template, 0, total_leaves

    template_array = jnp.asarray(template)
    loaded_array = jnp.asarray(loaded)
    if template_array.shape != loaded_array.shape:
        return template, 0, total_leaves
    return jnp.asarray(loaded_array, dtype=template_array.dtype), 1, total_leaves


def _rotated_sample(
    sample: dict[str, Any],
    rotation_matrix: jnp.ndarray,
) -> dict[str, Any]:
    rotated = dict(sample)
    rotated["atomic_coordinates"] = sample["atomic_coordinates"] @ rotation_matrix.T
    return rotated


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    device = _resolve_device(args.device)
    checkpoint = _load_checkpoint_if_present(args)
    dataset = JAXStepDataset(
        str(args.data_root),
        dtype=_torch_dtype_from_arg(args.dtype),
    )
    model = _build_model(dataset, args, checkpoint)
    sample = jax.device_put(dataset.example_sample(args.sample_index), device=device)
    layout = dataset.bucket_layout(dataset.bucket_signature_at(args.sample_index))

    variables = model.init(
        jax.random.PRNGKey(args.seed),
        sample,
        sample["orbital_value_table"],
        layout=layout,
    )
    if checkpoint is not None:
        merged_params_tree, _, _ = _merge_param_tree(
            flax.core.unfreeze(variables["params"]),
            flax.core.unfreeze(checkpoint["params"]),
        )
        params = flax.core.freeze(merged_params_tree)
    else:
        params = variables["params"]

    sparse_coefficients = sample["orbital_value_table"]
    dense_coefficients = sparse_to_dense_ao_coefficients(sample, sparse_coefficients)

    def energy_from_dense(local_sample: dict[str, Any], dense_input: jnp.ndarray) -> Any:
        output = model.apply(
            {"params": params},
            local_sample,
            dense_input,
            layout=layout,
            method=model.forward_dense,
        )
        return output.total_energy, output

    (reference_energy, output), gradient = jax.value_and_grad(
        energy_from_dense,
        argnums=1,
        has_aux=True,
    )(sample, dense_coefficients)
    sparse_gradient = dense_to_sparse_ao_coefficients(sample, gradient)
    sparse_residual = output.orbital_residual
    projected_dense_residual = sparse_to_dense_ao_coefficients(sample, sparse_residual)
    raw_dense_residual = (
        projected_dense_residual
        if output.dense_orbital_residual is None
        else output.dense_orbital_residual
    )

    key = jax.random.PRNGKey(args.seed)
    trial_records: list[dict[str, float]] = []
    for trial_index in range(args.num_trials):
        key, subkey = jax.random.split(key)
        rotation_matrix = e3nn.rand_matrix(subkey, dtype=dense_coefficients.dtype)
        rotated_input = rotate_dense_ao_coefficients(
            sample,
            dense_coefficients,
            rotation_matrix,
        )
        rotated_sample = _rotated_sample(sample, rotation_matrix)
        (rotated_energy, rotated_output), rotated_gradient = jax.value_and_grad(
            energy_from_dense,
            argnums=1,
            has_aux=True,
        )(rotated_sample, rotated_input)
        expected_rotated_gradient = rotate_dense_ao_coefficients(
            sample,
            gradient,
            rotation_matrix,
        )
        expected_rotated_sparse_gradient = dense_to_sparse_ao_coefficients(
            sample,
            expected_rotated_gradient,
        )
        rotated_sparse_gradient = dense_to_sparse_ao_coefficients(
            sample,
            rotated_gradient,
        )
        expected_rotated_raw_dense_residual = rotate_dense_ao_coefficients(
            sample,
            raw_dense_residual,
            rotation_matrix,
        )
        rotated_raw_dense_residual = (
            sparse_to_dense_ao_coefficients(sample, rotated_output.orbital_residual)
            if rotated_output.dense_orbital_residual is None
            else rotated_output.dense_orbital_residual
        )
        expected_rotated_projected_dense_residual = rotate_dense_ao_coefficients(
            sample,
            projected_dense_residual,
            rotation_matrix,
        )
        rotated_projected_dense_residual = sparse_to_dense_ao_coefficients(
            sample,
            rotated_output.orbital_residual,
        )
        expected_rotated_sparse_residual = dense_to_sparse_ao_coefficients(
            sample,
            expected_rotated_projected_dense_residual,
        )
        trial_records.append(
            {
                "trial_index": float(trial_index),
                "energy_abs_diff": float(jnp.abs(rotated_energy - reference_energy)),
                "two_electron_h_abs_diff": float(
                    jnp.max(
                        jnp.abs(
                            rotated_output.two_electron_hamiltonian
                            - output.two_electron_hamiltonian
                        )
                    )
                ),
                "dense_gradient_max_abs_diff": float(
                    jnp.max(jnp.abs(rotated_gradient - expected_rotated_gradient))
                ),
                "dense_gradient_l2_diff": float(
                    jnp.linalg.norm(rotated_gradient - expected_rotated_gradient)
                ),
                "sparse_gradient_max_abs_diff": float(
                    jnp.max(
                        jnp.abs(rotated_sparse_gradient - expected_rotated_sparse_gradient)
                    )
                ),
                "sparse_gradient_l2_diff": float(
                    jnp.linalg.norm(
                        rotated_sparse_gradient - expected_rotated_sparse_gradient
                    )
                ),
                "raw_dense_residual_max_abs_diff": float(
                    jnp.max(
                        jnp.abs(
                            rotated_raw_dense_residual
                            - expected_rotated_raw_dense_residual
                        )
                    )
                ),
                "raw_dense_residual_l2_diff": float(
                    jnp.linalg.norm(
                        rotated_raw_dense_residual
                        - expected_rotated_raw_dense_residual
                    )
                ),
                "projected_dense_residual_max_abs_diff": float(
                    jnp.max(
                        jnp.abs(
                            rotated_projected_dense_residual
                            - expected_rotated_projected_dense_residual
                        )
                    )
                ),
                "projected_dense_residual_l2_diff": float(
                    jnp.linalg.norm(
                        rotated_projected_dense_residual
                        - expected_rotated_projected_dense_residual
                    )
                ),
                "sparse_residual_max_abs_diff": float(
                    jnp.max(
                        jnp.abs(
                            rotated_output.orbital_residual - expected_rotated_sparse_residual
                        )
                    )
                ),
                "sparse_residual_l2_diff": float(
                    jnp.linalg.norm(
                        rotated_output.orbital_residual - expected_rotated_sparse_residual
                    )
                ),
            }
        )

    print(
        json.dumps(
            {
                "sample_name": dataset.sample_name_at(args.sample_index),
                "sample_index": args.sample_index,
                "checkpoint": None if checkpoint is None else str(args.checkpoint),
                "num_trials": args.num_trials,
                "max_energy_abs_diff": max(
                    record["energy_abs_diff"] for record in trial_records
                ),
                "max_two_electron_h_abs_diff": max(
                    record["two_electron_h_abs_diff"] for record in trial_records
                ),
                "max_dense_gradient_abs_diff": max(
                    record["dense_gradient_max_abs_diff"] for record in trial_records
                ),
                "max_sparse_gradient_abs_diff": max(
                    record["sparse_gradient_max_abs_diff"] for record in trial_records
                ),
                "max_raw_dense_residual_abs_diff": max(
                    record["raw_dense_residual_max_abs_diff"] for record in trial_records
                ),
                "max_projected_dense_residual_abs_diff": max(
                    record["projected_dense_residual_max_abs_diff"]
                    for record in trial_records
                ),
                "max_sparse_residual_abs_diff": max(
                    record["sparse_residual_max_abs_diff"] for record in trial_records
                ),
                "trials": trial_records,
            },
            ensure_ascii=False,
        )
    )
    print(
        json.dumps(
            {
                "reference_total_energy": float(reference_energy),
                "reference_dense_gradient_norm": float(jnp.linalg.norm(gradient)),
                "reference_sparse_gradient_norm": float(jnp.linalg.norm(sparse_gradient)),
                "reference_raw_dense_residual_norm": float(
                    jnp.linalg.norm(raw_dense_residual)
                ),
                "reference_projected_dense_residual_norm": float(
                    jnp.linalg.norm(projected_dense_residual)
                ),
                "reference_sparse_residual_norm": float(jnp.linalg.norm(sparse_residual)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
