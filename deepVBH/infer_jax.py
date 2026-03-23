from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from .runtime_env import maybe_sanitize_cuda_runtime_env

maybe_sanitize_cuda_runtime_env()

import torch

from .jax_dataset import JAXStepDataset
from .jax_e3nn_model import DeepVBHE3Model
from .jax_model import DeepVBHBaseline
from .jax_shell_ops import sparse_to_dense_ao_coefficients


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepVBH JAX inference for one exported step")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--accepted-iteration-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="",
    )
    parser.add_argument("--orbital-update-scale", type=float, default=1.0)
    return parser.parse_args()


def _torch_dtype_from_name(value: str) -> torch.dtype:
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
    devices = jax.devices(platform)
    if not devices:
        raise ValueError(f"no JAX devices available for platform={platform}")
    return devices[0]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
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


def _build_model(
    dataset: JAXStepDataset,
    checkpoint_args: dict[str, Any],
) -> Any:
    model_type = str(checkpoint_args["model_type"])
    if model_type == "baseline":
        return DeepVBHBaseline(
            max_atomic_number=max(dataset.max_atomic_number(), 1),
            max_angular_momentum=max(dataset.max_angular_momentum(), 1),
            max_ao_local_index=max(dataset.max_ao_local_index(), 1),
            hidden_dim=int(checkpoint_args["hidden_dim"]),
            pair_hidden_dim=int(checkpoint_args["pair_hidden_dim"]),
            predict_reference_residual=bool(
                checkpoint_args.get("predict_reference_residual", True)
            ),
        )
    if model_type == "e3nn":
        atom_channels = int(checkpoint_args["atom_channels"])
        return DeepVBHE3Model(
            max_atomic_number=max(dataset.max_atomic_number(), 1),
            max_angular_momentum=max(dataset.max_angular_momentum(), 1),
            max_ao_local_index=max(dataset.max_ao_local_index(), 1),
            hidden_dim=int(checkpoint_args["hidden_dim"]),
            pair_hidden_dim=int(checkpoint_args["pair_hidden_dim"]),
            atom_channels=None if atom_channels == 0 else atom_channels,
            ao_scalar_channels=None,
            lmax=int(checkpoint_args["lmax"]),
            num_radial=int(checkpoint_args["num_radial"]),
            cutoff=float(checkpoint_args["cutoff"]),
            predict_reference_residual=bool(
                checkpoint_args.get("predict_reference_residual", True)
            ),
        )
    raise ValueError(f"unsupported model type: {model_type}")


def _find_dataset_index(
    dataset: JAXStepDataset,
    sample_name: str,
    accepted_iteration_index: int,
) -> int:
    sample_indices = dataset.indices_by_sample().get(sample_name)
    if not sample_indices:
        raise ValueError(f"sample not found in dataset: {sample_name}")

    for index in sample_indices:
        if dataset.accepted_iteration_index_at(index) == accepted_iteration_index:
            return index
    raise ValueError(
        f"accepted_iteration_index={accepted_iteration_index} not found for sample={sample_name}"
    )


def _write_column_major_matrix(path: Path, matrix: np.ndarray) -> None:
    np.asfortranarray(matrix.astype(np.float64, copy=False)).reshape(-1, order="F").tofile(path)


def _write_scalar(path: Path, value: float) -> None:
    np.asarray([value], dtype=np.float64).tofile(path)


def _write_array(path: Path, array: np.ndarray) -> None:
    np.asarray(array, dtype=np.float64).tofile(path)


def main() -> None:
    args = _parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint_args = dict(checkpoint["args"])

    dtype_name = args.dtype or str(checkpoint_args["dtype"])
    device_name = args.device or str(checkpoint_args["device"])

    jax.config.update("jax_enable_x64", dtype_name == "float64")
    device = _resolve_device(device_name)

    dataset = JAXStepDataset(
        str(args.data_root),
        dtype=_torch_dtype_from_name(dtype_name),
        include_samples={args.sample_name},
    )
    dataset_index = _find_dataset_index(
        dataset,
        args.sample_name,
        args.accepted_iteration_index,
    )
    sample = jax.device_put(dataset.example_sample(dataset_index), device=device)
    model = _build_model(dataset, checkpoint_args)
    model_type = str(checkpoint_args["model_type"])
    orbital_update_space = str(checkpoint_args.get("orbital_update_space", "sparse"))
    layout = (
        dataset.bucket_layout(dataset.bucket_signature_at(dataset_index))
        if model_type == "e3nn"
        else None
    )
    if layout is None:
        initialized_params = model.init(
            jax.random.PRNGKey(0),
            sample,
            sample["orbital_value_table"],
        )["params"]
    elif orbital_update_space == "dense_shell":
        initialized_params = model.init(
            jax.random.PRNGKey(0),
            sample,
            sample["dense_orbital_coefficients"],
            layout=layout,
            method=model.forward_dense,
        )["params"]
    else:
        initialized_params = model.init(
            jax.random.PRNGKey(0),
            sample,
            sample["orbital_value_table"],
            layout=layout,
        )["params"]
    merged_params_tree, reused_leaves, total_leaves = _merge_param_tree(
        flax.core.unfreeze(initialized_params),
        flax.core.unfreeze(checkpoint["params"]),
    )
    params = flax.core.freeze(merged_params_tree)

    if layout is None:
        output = model.apply(
            {"params": params},
            sample,
            sample["orbital_value_table"],
        )
    elif orbital_update_space == "dense_shell":
        output = model.apply(
            {"params": params},
            sample,
            sample["dense_orbital_coefficients"],
            layout=layout,
            method=model.forward_dense,
        )
    else:
        output = model.apply(
            {"params": params},
            sample,
            sample["orbital_value_table"],
            layout=layout,
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    two_electron_hamiltonian = np.asarray(
        jax.device_get(output.two_electron_hamiltonian),
        dtype=np.float64,
    )
    structure_hamiltonian = np.asarray(
        jax.device_get(output.structure_hamiltonian),
        dtype=np.float64,
    )
    orbital_residual = np.asarray(
        jax.device_get(output.orbital_residual),
        dtype=np.float64,
    )
    raw_dense_orbital_residual = np.asarray(
        jax.device_get(
            output.dense_orbital_residual
            if output.dense_orbital_residual is not None
            else sparse_to_dense_ao_coefficients(sample, output.orbital_residual)
        ),
        dtype=np.float64,
    )
    dense_orbital_mask = np.asarray(
        jax.device_get(sample["orbital_shell_dense_mask"]),
        dtype=np.float64,
    )
    dense_orbital_residual = raw_dense_orbital_residual * dense_orbital_mask
    updated_orbital_value_table = np.asarray(
        jax.device_get(sample["orbital_value_table"]),
        dtype=np.float64,
    ) + float(args.orbital_update_scale) * orbital_residual
    updated_dense_orbital_coefficients = np.asarray(
        jax.device_get(sample["dense_orbital_coefficients"]),
        dtype=np.float64,
    ) + float(args.orbital_update_scale) * dense_orbital_residual
    _write_column_major_matrix(
        output_dir / "predicted_two_electron_hamiltonian_f64.bin",
        two_electron_hamiltonian,
    )
    _write_column_major_matrix(
        output_dir / "predicted_structure_hamiltonian_f64.bin",
        structure_hamiltonian,
    )
    _write_scalar(
        output_dir / "predicted_reference_energy_residual_f64.bin",
        float(np.asarray(jax.device_get(output.reference_energy_residual))),
    )
    _write_scalar(
        output_dir / "predicted_total_energy_f64.bin",
        float(np.asarray(jax.device_get(output.total_energy))),
    )
    _write_array(
        output_dir / "predicted_orbital_residual_f64.bin",
        orbital_residual,
    )
    _write_array(
        output_dir / "predicted_dense_orbital_residual_f64.bin",
        dense_orbital_residual,
    )
    _write_array(
        output_dir / "predicted_raw_dense_orbital_residual_f64.bin",
        raw_dense_orbital_residual,
    )
    _write_array(
        output_dir / "predicted_updated_orbital_value_table_f64.bin",
        updated_orbital_value_table,
    )
    _write_array(
        output_dir / "predicted_updated_dense_orbital_coefficients_f64.bin",
        updated_dense_orbital_coefficients,
    )

    metadata = {
        "sample_name": args.sample_name,
        "accepted_iteration_index": int(args.accepted_iteration_index),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": device_name,
        "dtype": dtype_name,
        "model_type": model_type,
        "orbital_update_space": orbital_update_space,
        "predict_reference_residual": bool(
            checkpoint_args.get("predict_reference_residual", True)
        ),
        "reused_param_leaves": reused_leaves,
        "total_param_leaves": total_leaves,
        "predicted_total_energy": float(np.asarray(jax.device_get(output.total_energy))),
        "predicted_reference_energy_residual": float(
            np.asarray(jax.device_get(output.reference_energy_residual))
        ),
        "orbital_update_scale": float(args.orbital_update_scale),
        "predicted_orbital_residual_norm": float(np.linalg.norm(orbital_residual)),
        "predicted_dense_orbital_residual_norm": float(
            np.linalg.norm(dense_orbital_residual)
        ),
        "predicted_raw_dense_orbital_residual_norm": float(
            np.linalg.norm(raw_dense_orbital_residual)
        ),
        "predicted_updated_orbital_value_norm": float(
            np.linalg.norm(updated_orbital_value_table)
        ),
        "predicted_updated_dense_orbital_norm": float(
            np.linalg.norm(updated_dense_orbital_coefficients)
        ),
        "exact_reference_energy": float(np.asarray(sample["one_electron_reference_energy"])),
        "nuclear_repulsion_energy": float(np.asarray(sample["nuclear_repulsion_energy"])),
        "n_structures": int(np.asarray(sample["n_structures"])),
        "n_orbitals": int(np.asarray(sample["n_orbitals"])),
        "n_basis_functions": int(np.asarray(sample["n_basis_functions"])),
    }
    (output_dir / "prediction.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
