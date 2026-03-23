from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from .runtime_env import maybe_sanitize_cuda_runtime_env

maybe_sanitize_cuda_runtime_env()

from .infer_jax import (
    _build_model,
    _find_dataset_index,
    _load_checkpoint,
    _merge_param_tree,
    _resolve_device,
    _torch_dtype_from_name,
)
from .jax_dataset import JAXStepDataset
from .jax_shell_ops import sparse_to_dense_ao_coefficients

_ONNX_INT64_INPUT_KEYS = {
    "atomic_numbers",
    "orbital_basis_counts",
    "structure_pair_orbital_indices",
    "structure_open_shell_orbitals",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a DeepVBH deployment package and ONNX skeleton metadata",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--accepted-iteration-index", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to generate example tensors and reference outputs.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64", ""],
        default="",
        help="Optional override for the checkpoint dtype.",
    )
    parser.add_argument(
        "--export-format",
        choices=["manifest", "saved_model", "onnx"],
        default="manifest",
        help="`manifest` writes the deployment package contract only; `saved_model` writes the TensorFlow intermediate; `onnx` attempts final ONNX export.",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=18,
        help="ONNX opset used for the deployment export path.",
    )
    return parser.parse_args()


def _layout_to_jsonable(layout: dict[str, Any] | None) -> dict[str, list[int]] | None:
    if layout is None:
        return None
    return {
        key: [int(value) for value in values]
        for key, values in layout.items()
    }


def _array_summary(name: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(dimension) for dimension in array.shape],
        "dtype": str(array.dtype),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_numpy_tree(sample: dict[str, Any], keys: list[str]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(jax.device_get(sample[key]))
        for key in keys
    }


def _to_onnx_numpy_tree(sample: dict[str, Any], keys: list[str]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key in keys:
        value = np.asarray(jax.device_get(sample[key]))
        if key in _ONNX_INT64_INPUT_KEYS:
            value = value.astype(np.int64, copy=False)
        result[key] = value
    return result


def _model_runtime_input_keys(
    checkpoint_args: dict[str, Any],
) -> tuple[list[str], list[str]]:
    model_type = str(checkpoint_args["model_type"])
    orbital_update_space = str(checkpoint_args.get("orbital_update_space", "sparse"))
    if model_type == "baseline":
        model_keys = [
            "atomic_numbers",
            "atomic_coordinates",
            "ao_to_atom",
            "ao_angular_momenta",
            "ao_shell_local_indices",
            "ao_cartesian_exponents",
            "structure_occupancy",
            "overlap_matrix",
            "one_electron_reference_energy",
            "one_electron_hamiltonian_matrix",
            "nuclear_repulsion_energy",
            "orbital_value_table",
        ]
        host_keys = ["orbital_basis_mask"]
        return model_keys, host_keys

    if orbital_update_space == "dense_shell":
        model_keys = [
            "atomic_numbers",
            "atomic_coordinates",
            "orbital_basis_counts",
            "structure_occupancy",
            "structure_pair_orbital_indices",
            "structure_pair_mask",
            "structure_open_shell_orbitals",
            "structure_open_shell_mask",
            "overlap_matrix",
            "one_electron_reference_energy",
            "one_electron_hamiltonian_matrix",
            "nuclear_repulsion_energy",
            "dense_orbital_coefficients",
        ]
        host_keys = [
            "orbital_shell_dense_mask",
            "orbital_basis_mask",
            "orbital_basis_index_table",
        ]
        return model_keys, host_keys

    model_keys = [
        "atomic_numbers",
        "atomic_coordinates",
        "orbital_basis_counts",
        "structure_occupancy",
        "structure_pair_orbital_indices",
        "structure_pair_mask",
        "structure_open_shell_orbitals",
        "structure_open_shell_mask",
        "overlap_matrix",
        "one_electron_reference_energy",
        "one_electron_hamiltonian_matrix",
        "nuclear_repulsion_energy",
        "orbital_value_table",
    ]
    host_keys = ["orbital_basis_mask"]
    return model_keys, host_keys


def _deployment_onnx_input_keys(
    checkpoint_args: dict[str, Any],
) -> list[str]:
    model_type = str(checkpoint_args["model_type"])
    orbital_update_space = str(checkpoint_args.get("orbital_update_space", "sparse"))
    if model_type != "e3nn" or orbital_update_space != "dense_shell":
        raise RuntimeError(
            "JAX -> ONNX deployment export currently supports only `e3nn` checkpoints "
            "with `orbital_update_space=dense_shell`."
        )
    return [
        "atomic_numbers",
        "atomic_coordinates",
        "orbital_basis_counts",
        "structure_occupancy",
        "structure_pair_orbital_indices",
        "structure_pair_mask",
        "structure_open_shell_orbitals",
        "structure_open_shell_mask",
        "overlap_matrix",
        "dense_orbital_coefficients",
        "orbital_shell_dense_mask",
    ]


def _checkpoint_dtype_name(checkpoint_args: dict[str, Any], override: str) -> str:
    if override:
        return override
    return str(checkpoint_args["dtype"])


def _example_outputs(
    sample: dict[str, Any],
    output: Any,
) -> dict[str, np.ndarray]:
    sparse_residual = np.asarray(jax.device_get(output.orbital_residual), dtype=np.float64)
    raw_dense_residual = np.asarray(
        jax.device_get(
            output.dense_orbital_residual
            if output.dense_orbital_residual is not None
            else sparse_to_dense_ao_coefficients(sample, output.orbital_residual)
        ),
        dtype=np.float64,
    )
    dense_mask = np.asarray(jax.device_get(sample["orbital_shell_dense_mask"]), dtype=np.float64)
    dense_residual = raw_dense_residual * dense_mask
    return {
        "two_electron_hamiltonian": np.asarray(
            jax.device_get(output.two_electron_hamiltonian),
            dtype=np.float64,
        ),
        "structure_hamiltonian": np.asarray(
            jax.device_get(output.structure_hamiltonian),
            dtype=np.float64,
        ),
        "reference_energy_residual": np.asarray(
            jax.device_get(output.reference_energy_residual),
            dtype=np.float64,
        ),
        "total_energy": np.asarray(
            jax.device_get(output.total_energy),
            dtype=np.float64,
        ),
        "sparse_orbital_residual": sparse_residual,
        "dense_orbital_residual": dense_residual,
        "raw_dense_orbital_residual": raw_dense_residual,
    }


def _write_readme(
    path: Path,
    *,
    checkpoint_path: Path,
    export_format: str,
    model_type: str,
    orbital_update_space: str,
) -> None:
    lines = [
        "# DeepVBH Deployment Package",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Export format: `{export_format}`",
        f"- Model type: `{model_type}`",
        f"- Orbital update space: `{orbital_update_space}`",
        "",
        "Files:",
        "- `manifest.json`: deployment contract, input/output summaries, and baked shell layout.",
        "- `checkpoint_args.json`: serialized checkpoint arguments.",
        "- `example_inputs.npz`: reference runtime tensors for one exported sample.",
        "- `example_outputs.npz`: reference outputs produced by the current JAX model.",
        "- `saved_model/`: optional TensorFlow SavedModel intermediate used for JAX -> ONNX export attempts.",
        "- `model.onnx`: final ONNX graph when export succeeds.",
        "",
        "Notes:",
        "- The current `dense_shell` deployment path assumes the shell layout is baked into one exported model package.",
        "- `predicted_dense_orbital_residual` uses the shell-complete locality mask and matches the training target semantics.",
        "- The deployment ONNX graph omits `total_energy`; C++ recomputes the final scalar from the predicted two-electron Hamiltonian, exact overlap, and exact generalized eigensolve.",
        "- The deployment ONNX graph only exports the dense-shell path used by the current C++ runtime.",
        "- ONNX export uses `jax2onnx` because the JAX 0.9.x -> TensorFlow -> tf2onnx path emits `XlaCallModule`, which is not convertible today.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_saved_model_dependencies() -> None:
    required_modules = [
        "tensorflow",
    ]
    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    try:
        from jax.experimental import jax2tf  # noqa: F401
    except Exception:
        missing.append("jax.experimental.jax2tf")
    if missing:
        raise RuntimeError(
            "full ONNX export is not available in the current environment; missing dependencies: "
            + ", ".join(missing)
        )
    return


def _ensure_onnx_dependencies() -> None:
    required_modules = [
        "onnx",
        "onnxruntime",
        "jax2onnx",
    ]
    missing = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "JAX -> ONNX export is not available in the current environment; missing dependencies: "
            + ", ".join(missing)
        )


def _build_deployment_onnx_function(
    *,
    model: Any,
    params: Any,
    layout: dict[str, Any] | None,
    checkpoint_args: dict[str, Any],
    input_keys: list[str],
):
    model_type = str(checkpoint_args["model_type"])
    orbital_update_space = str(checkpoint_args.get("orbital_update_space", "sparse"))
    if model_type != "e3nn" or orbital_update_space != "dense_shell":
        raise RuntimeError(
            "deployment ONNX export currently supports only `e3nn` dense-shell checkpoints"
        )
    if layout is None:
        raise RuntimeError("dense-shell deployment ONNX export requires a baked shell layout")

    def apply_flat(*flat_args):
        data: dict[str, Any] = {}
        for key, value in zip(input_keys, flat_args, strict=True):
            if key in _ONNX_INT64_INPUT_KEYS:
                data[key] = jnp.asarray(value, dtype=jnp.int32)
            else:
                data[key] = value
        return model.apply(
            {"params": params},
            data,
            data["dense_orbital_coefficients"],
            layout=layout,
            method=model.forward_dense_deployment,
        )

    return apply_flat


@contextmanager
def _patched_jax2onnx_concatenate() -> Any:
    from jax import core
    from jax2onnx.plugins._patching import AssignSpec, MonkeyPatchSpec
    from jax2onnx.plugins.jax.numpy._common import get_orig_impl
    from jax2onnx.plugins.jax.numpy.concatenate import (
        JnpConcatenatePlugin,
        _ORIGINAL_JNP_CONCATENATE,
    )

    original_binding_specs = JnpConcatenatePlugin.binding_specs

    def patched_binding_specs(cls):
        storage_slot = f"__orig_impl__{cls._FUNC_NAME}"

        def _make_value(orig):
            if orig is None:
                raise RuntimeError("Original jnp.concatenate not found")
            setattr(cls._PRIM, storage_slot, orig)

            def _patched(*args, **kwargs):
                arrays, axis, dtype = cls._canonicalize_call(*args, **kwargs)
                axis_int = int(axis)
                arrays_for_bind = arrays
                if dtype is not None:
                    target_dtype = np.dtype(dtype)
                    arrays_for_bind = tuple(
                        jnp.asarray(arr, dtype=target_dtype) for arr in arrays
                    )
                if not any(isinstance(arr, core.Tracer) for arr in arrays_for_bind):
                    try:
                        original = get_orig_impl(cls._PRIM, cls._FUNC_NAME)
                    except RuntimeError:
                        original = _ORIGINAL_JNP_CONCATENATE
                    return original(arrays_for_bind, axis=axis_int)
                rank = jnp.asarray(arrays_for_bind[0]).ndim
                norm_axis = axis_int % rank if axis_int < 0 else axis_int
                return cls._PRIM.bind(*arrays_for_bind, dimension=norm_axis)

            return _patched

        return [
            AssignSpec(
                "jax.numpy", f"{cls._FUNC_NAME}_p", cls._PRIM, delete_if_missing=True
            ),
            MonkeyPatchSpec(
                target="jax.numpy",
                attr=cls._FUNC_NAME,
                make_value=_make_value,
                delete_if_missing=False,
            ),
        ]

    JnpConcatenatePlugin.binding_specs = classmethod(patched_binding_specs)
    try:
        yield
    finally:
        JnpConcatenatePlugin.binding_specs = original_binding_specs


@contextmanager
def _patched_jax2onnx_dot_general() -> Any:
    from jax2onnx.plugins.jax.lax.dot_general import DotGeneralPlugin

    original_compute_einsum_labels = DotGeneralPlugin._compute_einsum_labels

    def patched_compute_einsum_labels(
        self,
        lhs_shape: tuple[int, ...],
        rhs_shape: tuple[int, ...],
        lhs_contract: tuple[int, ...],
        rhs_contract: tuple[int, ...],
        lhs_batch: tuple[int, ...],
        rhs_batch: tuple[int, ...],
    ) -> tuple[str, str, str]:
        lhs_rank = len(lhs_shape)
        rhs_rank = len(rhs_shape)

        lhs_batch_set = set(lhs_batch)
        rhs_batch_set = set(rhs_batch)
        lhs_contract_set = set(lhs_contract)
        rhs_contract_set = set(rhs_contract)

        lhs_free = [
            axis
            for axis in range(lhs_rank)
            if axis not in lhs_batch_set | lhs_contract_set
        ]
        rhs_free = [
            axis
            for axis in range(rhs_rank)
            if axis not in rhs_batch_set | rhs_contract_set
        ]

        main_gen = self._label_stream("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        batch_gen = self._label_stream("zyxwvutsrqponmlkjihgfedcbaZYXWVUTSRQPONMLKJIHGFEDCBA")

        lhs_lbl: list[str | None] = [None] * lhs_rank
        rhs_lbl: list[str | None] = [None] * rhs_rank

        batch_pairs = list(zip(lhs_batch, rhs_batch))
        contract_pairs = list(zip(lhs_contract, rhs_contract))
        lhs_for_rhs_contract = {
            rhs_axis: lhs_axis for lhs_axis, rhs_axis in contract_pairs
        }

        for i, j in batch_pairs:
            label = next(batch_gen)
            lhs_lbl[i] = label
            rhs_lbl[j] = label

        for i in range(lhs_rank):
            if i in lhs_batch_set:
                continue
            if lhs_lbl[i] is None:
                lhs_lbl[i] = next(main_gen)

        for j in range(rhs_rank):
            if j in rhs_batch_set:
                continue
            if j in rhs_contract_set:
                lhs_axis = lhs_for_rhs_contract[j]
                rhs_lbl[j] = lhs_lbl[lhs_axis]
            elif rhs_lbl[j] is None:
                rhs_lbl[j] = next(main_gen)

        out_labels_list: list[str | None] = []
        for axis in lhs_batch:
            out_labels_list.append(lhs_lbl[axis])
        for axis in lhs_free:
            out_labels_list.append(lhs_lbl[axis])
        for axis in rhs_free:
            out_labels_list.append(rhs_lbl[axis])

        if not lhs_free and not rhs_free:
            if lhs_batch:
                out_labels_list = [lhs_lbl[i] for i in lhs_batch]
            else:
                out_labels_list = []

        if any(label is None for label in lhs_lbl):
            raise RuntimeError(f"Unlabeled LHS axes: {lhs_lbl}")
        if any(label is None for label in rhs_lbl):
            raise RuntimeError(f"Unlabeled RHS axes: {rhs_lbl}")

        lhs_labels = "".join(label for label in lhs_lbl if label is not None)
        rhs_labels = "".join(label for label in rhs_lbl if label is not None)
        out_labels = "".join(label for label in out_labels_list if label is not None)
        return lhs_labels, rhs_labels, out_labels

    DotGeneralPlugin._compute_einsum_labels = patched_compute_einsum_labels
    try:
        yield
    finally:
        DotGeneralPlugin._compute_einsum_labels = original_compute_einsum_labels


def _deployment_output_summaries(
    runtime_outputs: dict[str, np.ndarray],
    checkpoint_args: dict[str, Any],
    *,
    include_total_energy: bool,
) -> list[dict[str, Any]]:
    orbital_update_space = str(checkpoint_args.get("orbital_update_space", "sparse"))
    if orbital_update_space == "dense_shell":
        keys = [
            "two_electron_hamiltonian",
            "reference_energy_residual",
            "dense_orbital_residual",
        ]
    else:
        keys = [
            "two_electron_hamiltonian",
            "reference_energy_residual",
            "sparse_orbital_residual",
        ]
    if include_total_energy:
        keys.insert(2, "total_energy")
    return [_array_summary(key, runtime_outputs[key]) for key in keys]


def _build_saved_model_signature(
    *,
    model: Any,
    params: Any,
    checkpoint_args: dict[str, Any],
    sample: dict[str, Any],
    layout: dict[str, Any] | None,
    model_input_keys: list[str],
    host_input_keys: list[str],
):
    import tensorflow as tf
    from jax.experimental import jax2tf

    orbital_update_space = str(checkpoint_args.get("orbital_update_space", "sparse"))
    all_keys = model_input_keys + host_input_keys

    def apply_flat(*flat_args):
        data = {key: value for key, value in zip(all_keys, flat_args, strict=True)}
        if layout is None:
            output = model.apply(
                {"params": params},
                data,
                data["orbital_value_table"],
            )
        elif orbital_update_space == "dense_shell":
            output = model.apply(
                {"params": params},
                data,
                data["dense_orbital_coefficients"],
                layout=layout,
                method=model.forward_dense,
            )
        else:
            output = model.apply(
                {"params": params},
                data,
                data["orbital_value_table"],
                layout=layout,
            )
        dense_residual = output.dense_orbital_residual
        if "orbital_shell_dense_mask" in data:
            dense_residual = dense_residual * data["orbital_shell_dense_mask"]
        result = {
            "two_electron_hamiltonian": output.two_electron_hamiltonian,
            "reference_energy_residual": output.reference_energy_residual,
            "total_energy": output.total_energy,
        }
        if orbital_update_space == "dense_shell":
            result["dense_orbital_residual"] = dense_residual
        else:
            result["sparse_orbital_residual"] = output.orbital_residual
        return result

    specs = [
        tf.TensorSpec(
            tuple(int(dimension) for dimension in sample[key].shape),
            tf.as_dtype(sample[key].dtype.name),
            name=key,
        )
        for key in all_keys
    ]
    converted = jax2tf.convert(
        apply_flat,
        with_gradient=False,
        enable_xla=False,
        native_serialization=False,
    )

    @tf.function(input_signature=specs, autograph=False)
    def serving_default(*args):
        return converted(*args)

    return serving_default, specs


def _export_saved_model(
    *,
    output_dir: Path,
    model: Any,
    params: Any,
    checkpoint_args: dict[str, Any],
    sample: dict[str, Any],
    layout: dict[str, Any] | None,
    model_input_keys: list[str],
    host_input_keys: list[str],
) -> Path:
    import tensorflow as tf

    serving_default, specs = _build_saved_model_signature(
        model=model,
        params=params,
        checkpoint_args=checkpoint_args,
        sample=sample,
        layout=layout,
        model_input_keys=model_input_keys,
        host_input_keys=host_input_keys,
    )
    saved_model_dir = output_dir / "saved_model"
    if saved_model_dir.exists():
        tf.io.gfile.rmtree(str(saved_model_dir))
    module = tf.Module()
    module.serving_default = serving_default
    tf.saved_model.save(
        module,
        str(saved_model_dir),
        signatures={"serving_default": serving_default.get_concrete_function(*specs)},
    )
    return saved_model_dir


def _export_onnx(
    *,
    output_dir: Path,
    model: Any,
    params: Any,
    checkpoint_args: dict[str, Any],
    sample: dict[str, Any],
    layout: dict[str, Any] | None,
    input_keys: list[str],
    opset: int,
) -> Path:
    import onnx
    from onnx import helper
    from jax2onnx import to_onnx

    apply_flat = _build_deployment_onnx_function(
        model=model,
        params=params,
        checkpoint_args=checkpoint_args,
        layout=layout,
        input_keys=input_keys,
    )
    input_specs = []
    for key in input_keys:
        if key in _ONNX_INT64_INPUT_KEYS:
            input_specs.append(jax.ShapeDtypeStruct(sample[key].shape, jnp.int64))
        else:
            input_specs.append(sample[key])
    onnx_path = output_dir / "model.onnx"
    with _patched_jax2onnx_concatenate(), _patched_jax2onnx_dot_general():
        to_onnx(
            apply_flat,
            inputs=input_specs,
            model_name="deepvbh_dense_shell_deployment",
            opset=opset,
            enable_double_precision=sample[input_keys[1]].dtype == jnp.float64,
            return_mode="file",
            output_path=str(onnx_path),
            input_names=input_keys,
            output_names=[
                "two_electron_hamiltonian",
                "reference_energy_residual",
                "dense_orbital_residual",
            ],
        )
    model = onnx.load(str(onnx_path))

    tensor_elem_types: dict[str, int] = {}
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type != 0:
            tensor_elem_types[value_info.name] = int(tensor_type.elem_type)
    for initializer in model.graph.initializer:
        tensor_elem_types.setdefault(initializer.name, int(initializer.data_type))

    castlike_replacements = 0
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "CastLike" or len(node.input) < 2:
            continue
        target_type = tensor_elem_types.get(node.input[1])
        if target_type is None:
            continue
        replacement = helper.make_node(
            "Cast",
            inputs=[node.input[0]],
            outputs=list(node.output),
            name=node.name,
            to=int(target_type),
        )
        model.graph.node[node_index].CopyFrom(replacement)
        for output_name in node.output:
            tensor_elem_types[output_name] = target_type
        castlike_replacements += 1

    stripped_value_info_count = len(model.graph.value_info)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))
    return onnx_path, {
        "castlike_replacements": castlike_replacements,
        "stripped_value_info_count": stripped_value_info_count,
    }


def main() -> None:
    args = _parse_args()
    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint_args = dict(checkpoint["args"])

    dtype_name = _checkpoint_dtype_name(checkpoint_args, args.dtype)
    device = _resolve_device(args.device)
    jax.config.update("jax_enable_x64", dtype_name == "float64")

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
        output = None
    elif orbital_update_space == "dense_shell":
        initialized_params = model.init(
            jax.random.PRNGKey(0),
            sample,
            sample["dense_orbital_coefficients"],
            layout=layout,
            method=model.forward_dense,
        )["params"]
        output = None
    else:
        initialized_params = model.init(
            jax.random.PRNGKey(0),
            sample,
            sample["orbital_value_table"],
            layout=layout,
        )["params"]
        output = None

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

    model_input_keys, host_input_keys = _model_runtime_input_keys(checkpoint_args)
    if args.export_format == "onnx":
        deployment_model_input_keys = _deployment_onnx_input_keys(checkpoint_args)
        deployment_host_input_keys: list[str] = []
        runtime_inputs = _to_onnx_numpy_tree(
            sample,
            deployment_model_input_keys + deployment_host_input_keys,
        )
    else:
        deployment_model_input_keys = model_input_keys
        deployment_host_input_keys = host_input_keys
        runtime_inputs = _to_numpy_tree(
            sample,
            deployment_model_input_keys + deployment_host_input_keys,
        )
    runtime_outputs = _example_outputs(sample, output)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "example_inputs.npz", **runtime_inputs)
    np.savez(output_dir / "example_outputs.npz", **runtime_outputs)

    checkpoint_args_path = output_dir / "checkpoint_args.json"
    checkpoint_args_path.write_text(
        json.dumps(_json_ready(checkpoint_args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "format_version": 1,
        "export_format": args.export_format,
        "checkpoint": str(args.checkpoint.resolve()),
        "model_type": model_type,
        "orbital_update_space": orbital_update_space,
        "sample_name": args.sample_name,
        "accepted_iteration_index": int(args.accepted_iteration_index),
        "device_for_examples": args.device,
        "dtype": dtype_name,
        "onnx_opset": int(args.onnx_opset) if args.export_format == "onnx" else None,
        "reused_param_leaves": reused_leaves,
        "total_param_leaves": total_leaves,
        "layout_baked_into_model": layout is not None,
        "bucket_layout": _layout_to_jsonable(layout),
        "model_runtime_inputs": [
            _array_summary(key, runtime_inputs[key])
            for key in deployment_model_input_keys
        ],
        "host_runtime_inputs": [
            _array_summary(key, runtime_inputs[key])
            for key in deployment_host_input_keys
        ],
        "runtime_outputs": _deployment_output_summaries(
            runtime_outputs,
            checkpoint_args,
            include_total_energy=args.export_format != "onnx",
        ),
        "notes": [
            "This package fixes the deployment contract before the final C++ backend is swapped.",
            "For `dense_shell`, the shell layout is treated as static package metadata.",
            "The exported `dense_orbital_residual` is masked to the shell-complete local support used during training.",
            "The example output archive still includes extra diagnostics such as `sparse_orbital_residual` and `raw_dense_orbital_residual`.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir / "README.md",
        checkpoint_path=args.checkpoint.resolve(),
        export_format=args.export_format,
        model_type=model_type,
        orbital_update_space=orbital_update_space,
    )

    onnx_export_summary: dict[str, Any] | None = None

    if args.export_format == "saved_model":
        _ensure_saved_model_dependencies()
        _export_saved_model(
            output_dir=output_dir,
            model=model,
            params=params,
            checkpoint_args=checkpoint_args,
            sample=sample,
            layout=layout,
            model_input_keys=model_input_keys,
            host_input_keys=host_input_keys,
        )

    if args.export_format == "onnx":
        _ensure_onnx_dependencies()
        _, onnx_export_summary = _export_onnx(
            output_dir=output_dir,
            model=model,
            params=params,
            checkpoint_args=checkpoint_args,
            sample=sample,
            layout=layout,
            input_keys=deployment_model_input_keys,
            opset=args.onnx_opset,
        )
        manifest["onnx_graph_sanitization"] = onnx_export_summary
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
