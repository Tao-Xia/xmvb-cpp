from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, checker, helper

from .runtime_env import maybe_sanitize_cuda_runtime_env

maybe_sanitize_cuda_runtime_env()

import torch

from .dataset import DeepVBHStepDataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a dummy DeepVBH dense-shell ONNX model for C++ backend validation.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--accepted-iteration-index", type=int, default=0)
    parser.add_argument(
        "--float-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating-point tensor dtype used for the dummy model inputs and outputs.",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported torch dtype: {name}")


def _onnx_float_dtype(name: str) -> int:
    if name == "float32":
        return TensorProto.FLOAT
    if name == "float64":
        return TensorProto.DOUBLE
    raise ValueError(f"unsupported ONNX float dtype: {name}")


def _find_sample_index(
    dataset: DeepVBHStepDataset,
    sample_name: str,
    accepted_iteration_index: int,
) -> int:
    for index in range(len(dataset)):
        if (
            dataset.sample_name_at(index) == sample_name
            and dataset.accepted_iteration_index_at(index) == accepted_iteration_index
        ):
            return index
    raise ValueError(
        f"sample={sample_name} accepted_iteration_index={accepted_iteration_index} not found"
    )


def _shape_of(sample: dict[str, object], key: str) -> list[int]:
    value = sample[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor sample entry for {key}, got {type(value)!r}")
    return [int(dimension) for dimension in value.shape]


def main() -> None:
    args = _parse_args()
    dataset = DeepVBHStepDataset(
        args.data_root,
        dtype=_torch_dtype(args.float_dtype),
        include_samples={args.sample_name},
    )
    sample_index = _find_sample_index(
        dataset,
        args.sample_name,
        args.accepted_iteration_index,
    )
    sample = dataset[sample_index]

    float_dtype = _onnx_float_dtype(args.float_dtype)
    int_dtype = TensorProto.INT64
    bool_dtype = TensorProto.BOOL

    inputs = [
        helper.make_tensor_value_info(
            "atomic_numbers",
            int_dtype,
            _shape_of(sample, "atomic_numbers"),
        ),
        helper.make_tensor_value_info(
            "atomic_coordinates",
            float_dtype,
            _shape_of(sample, "atomic_coordinates"),
        ),
        helper.make_tensor_value_info(
            "orbital_basis_counts",
            int_dtype,
            _shape_of(sample, "orbital_basis_counts"),
        ),
        helper.make_tensor_value_info(
            "structure_occupancy",
            float_dtype,
            _shape_of(sample, "structure_occupancy"),
        ),
        helper.make_tensor_value_info(
            "structure_pair_orbital_indices",
            int_dtype,
            _shape_of(sample, "structure_pair_orbital_indices"),
        ),
        helper.make_tensor_value_info(
            "structure_pair_mask",
            bool_dtype,
            _shape_of(sample, "structure_pair_mask"),
        ),
        helper.make_tensor_value_info(
            "structure_open_shell_orbitals",
            int_dtype,
            _shape_of(sample, "structure_open_shell_orbitals"),
        ),
        helper.make_tensor_value_info(
            "structure_open_shell_mask",
            bool_dtype,
            _shape_of(sample, "structure_open_shell_mask"),
        ),
        helper.make_tensor_value_info(
            "overlap_matrix",
            float_dtype,
            _shape_of(sample, "overlap_matrix"),
        ),
        helper.make_tensor_value_info(
            "one_electron_reference_energy",
            float_dtype,
            _shape_of(sample, "one_electron_reference_energy"),
        ),
        helper.make_tensor_value_info(
            "one_electron_hamiltonian_matrix",
            float_dtype,
            _shape_of(sample, "one_electron_hamiltonian_matrix"),
        ),
        helper.make_tensor_value_info(
            "nuclear_repulsion_energy",
            float_dtype,
            _shape_of(sample, "nuclear_repulsion_energy"),
        ),
        helper.make_tensor_value_info(
            "dense_orbital_coefficients",
            float_dtype,
            _shape_of(sample, "dense_orbital_coefficients"),
        ),
        helper.make_tensor_value_info(
            "orbital_shell_dense_mask",
            bool_dtype,
            _shape_of(sample, "orbital_shell_dense_mask"),
        ),
        helper.make_tensor_value_info(
            "orbital_basis_mask",
            bool_dtype,
            _shape_of(sample, "orbital_basis_mask"),
        ),
        helper.make_tensor_value_info(
            "orbital_basis_index_table",
            int_dtype,
            _shape_of(sample, "orbital_basis_index_table"),
        ),
    ]

    outputs = [
        helper.make_tensor_value_info(
            "two_electron_hamiltonian",
            float_dtype,
            _shape_of(sample, "two_electron_hamiltonian_matrix"),
        ),
        helper.make_tensor_value_info(
            "reference_energy_residual",
            float_dtype,
            _shape_of(sample, "one_electron_reference_energy"),
        ),
        helper.make_tensor_value_info(
            "total_energy",
            float_dtype,
            _shape_of(sample, "one_electron_reference_energy"),
        ),
        helper.make_tensor_value_info(
            "dense_orbital_residual",
            float_dtype,
            _shape_of(sample, "dense_orbital_coefficients"),
        ),
    ]

    zero_scalar = helper.make_tensor(
        "zero_scalar_value",
        float_dtype,
        dims=[1],
        vals=[0.0],
    )

    nodes = [
        helper.make_node(
            "Shape",
            ["one_electron_hamiltonian_matrix"],
            ["two_electron_hamiltonian_shape"],
        ),
        helper.make_node(
            "ConstantOfShape",
            ["two_electron_hamiltonian_shape"],
            ["two_electron_hamiltonian"],
            value=zero_scalar,
        ),
        helper.make_node(
            "Shape",
            ["dense_orbital_coefficients"],
            ["dense_orbital_residual_shape"],
        ),
        helper.make_node(
            "ConstantOfShape",
            ["dense_orbital_residual_shape"],
            ["dense_orbital_residual"],
            value=zero_scalar,
        ),
        helper.make_node(
            "Mul",
            ["one_electron_reference_energy", "zero_reference_scale"],
            ["reference_energy_residual"],
        ),
        helper.make_node(
            "Add",
            ["one_electron_reference_energy", "nuclear_repulsion_energy"],
            ["total_energy"],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        name="deepvbh_dense_shell_dummy",
        inputs=inputs,
        outputs=outputs,
        initializer=[
            helper.make_tensor(
                "zero_reference_scale",
                float_dtype,
                dims=[],
                vals=[0.0],
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="deepVBH.export_dummy_onnx",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
