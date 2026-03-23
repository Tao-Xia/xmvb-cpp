from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from e3nn import o3

from .dataset import DeepVBHStepDataset, move_sample_to_device
from .e3nn_model import DeepVBHE3Model
from .shell_ops import (
    dense_to_sparse_ao_coefficients,
    rotate_dense_ao_coefficients,
    sparse_to_dense_ao_coefficients,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DeepVBH shell equivariance")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--pair-hidden-dim", type=int, default=64)
    parser.add_argument("--atom-channels", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _dtype_from_arg(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {value}")


def _build_model(
    dataset: DeepVBHStepDataset,
    args: argparse.Namespace,
) -> DeepVBHE3Model:
    return DeepVBHE3Model(
        max_atomic_number=max(dataset.max_atomic_number(), 1),
        max_angular_momentum=max(dataset.max_angular_momentum(), 1),
        max_ao_local_index=max(dataset.max_ao_local_index(), 1),
        hidden_dim=args.hidden_dim,
        pair_hidden_dim=args.pair_hidden_dim,
        atom_channels=args.atom_channels or None,
        lmax=args.lmax,
        num_radial=args.num_radial,
        cutoff=args.cutoff,
    )


def _load_checkpoint_if_present(
    model: DeepVBHE3Model,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.checkpoint is None:
        return None
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _rotated_sample(
    sample: dict[str, Any],
    rotation_matrix: torch.Tensor,
) -> dict[str, Any]:
    rotated = dict(sample)
    rotated["atomic_coordinates"] = sample["atomic_coordinates"] @ rotation_matrix.transpose(0, 1)
    return rotated


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    dtype = _dtype_from_arg(args.dtype)
    device = torch.device(args.device)

    dataset = DeepVBHStepDataset(args.data_root, dtype=dtype)
    model = _build_model(dataset, args).to(device=device, dtype=dtype)
    checkpoint = _load_checkpoint_if_present(model, args)
    model.eval()

    sample = move_sample_to_device(dataset[args.sample_index], device)
    sparse_coefficients = sample["orbital_value_table"].clone().detach()
    dense_coefficients = sparse_to_dense_ao_coefficients(sample, sparse_coefficients)

    dense_input = dense_coefficients.clone().detach().requires_grad_(True)
    output = model.forward_dense(sample, dense_input)
    gradient = torch.autograd.grad(output.total_energy, dense_input)[0].detach()
    sparse_gradient = dense_to_sparse_ao_coefficients(sample, gradient)

    trial_records: list[dict[str, float]] = []
    for trial_index in range(args.num_trials):
        rotation_matrix = o3.rand_matrix().to(device=device, dtype=dtype)
        rotated_input = rotate_dense_ao_coefficients(
            sample,
            dense_coefficients,
            rotation_matrix,
        ).clone().detach().requires_grad_(True)
        rotated_output = model.forward_dense(
            _rotated_sample(sample, rotation_matrix),
            rotated_input,
        )
        rotated_gradient = torch.autograd.grad(
            rotated_output.total_energy,
            rotated_input,
        )[0].detach()
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
        trial_records.append(
            {
                "trial_index": float(trial_index),
                "energy_abs_diff": float(
                    (rotated_output.total_energy - output.total_energy).abs().item()
                ),
                "two_electron_h_abs_diff": float(
                    (
                        rotated_output.two_electron_hamiltonian
                        - output.two_electron_hamiltonian
                    )
                    .abs()
                    .max()
                    .item()
                ),
                "dense_gradient_max_abs_diff": float(
                    (rotated_gradient - expected_rotated_gradient).abs().max().item()
                ),
                "dense_gradient_l2_diff": float(
                    torch.linalg.norm(rotated_gradient - expected_rotated_gradient).item()
                ),
                "sparse_gradient_max_abs_diff": float(
                    (rotated_sparse_gradient - expected_rotated_sparse_gradient)
                    .abs()
                    .max()
                    .item()
                ),
                "sparse_gradient_l2_diff": float(
                    torch.linalg.norm(
                        rotated_sparse_gradient - expected_rotated_sparse_gradient
                    ).item()
                ),
            }
        )

    summary = {
        "sample_name": sample["sample_name"],
        "sample_index": args.sample_index,
        "checkpoint": None if checkpoint is None else str(args.checkpoint),
        "num_trials": args.num_trials,
        "max_energy_abs_diff": max(record["energy_abs_diff"] for record in trial_records),
        "max_two_electron_h_abs_diff": max(
            record["two_electron_h_abs_diff"] for record in trial_records
        ),
        "max_dense_gradient_abs_diff": max(
            record["dense_gradient_max_abs_diff"] for record in trial_records
        ),
        "max_sparse_gradient_abs_diff": max(
            record["sparse_gradient_max_abs_diff"] for record in trial_records
        ),
        "trials": trial_records,
    }
    print(json.dumps(summary, ensure_ascii=False))
    print(
        json.dumps(
            {
                "reference_total_energy": float(output.total_energy.item()),
                "reference_dense_gradient_norm": float(torch.linalg.norm(gradient).item()),
                "reference_sparse_gradient_norm": float(
                    torch.linalg.norm(sparse_gradient).item()
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
