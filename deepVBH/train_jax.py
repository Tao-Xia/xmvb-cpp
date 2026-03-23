from __future__ import annotations

import argparse
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .runtime_env import maybe_sanitize_cuda_runtime_env

maybe_sanitize_cuda_runtime_env()

import torch
import yaml

from .jax_dataset import JAXStepDataset
from .jax_e3nn_model import DeepVBHE3Model
from .jax_losses import DeepVBHLossScales, DeepVBHLossWeights, compute_losses
from .jax_model import DeepVBHBaseline
from .jax_shell_ops import validate_cartesian_shell_layout


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. CLI flags override YAML values.",
    )
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train DeepVBH JAX/Flax models",
        parents=[config_parser],
    )
    parser.add_argument("--data-root", type=Path, default=Path("deepvbscf_data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument(
        "--model-type",
        choices=["baseline", "e3nn"],
        default="baseline",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pair-hidden-dim", type=int, default=128)
    parser.add_argument("--atom-channels", type=int, default=0)
    parser.add_argument("--ao-scalar-channels", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument(
        "--predict-reference-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/deepVBH_jax.pkl"),
    )
    parser.add_argument("--best-checkpoint", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--include-samples", type=str, default="")
    parser.add_argument("--exclude-samples", type=str, default="")
    parser.add_argument("--valid-samples", type=str, default="")
    parser.add_argument("--test-samples", type=str, default="")
    parser.add_argument("--valid-fraction", type=float, default=0.0)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--max-steps-per-sample", type=int, default=0)
    parser.add_argument(
        "--step-selection",
        choices=["all", "final", "uniform", "first"],
        default="final",
    )
    parser.add_argument("--gradient-loss-interval", type=int, default=1)
    parser.add_argument(
        "--cache-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preload-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cache-device-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument(
        "--normalize-losses",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "default"],
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hamiltonian-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--orbital-residual-mse-weight", type=float, default=0.0)
    parser.add_argument("--orbital-residual-direction-weight", type=float, default=0.0)
    parser.add_argument(
        "--orbital-residual-target",
        choices=["none", "final", "next"],
        default="none",
    )
    parser.add_argument(
        "--orbital-update-space",
        choices=["sparse", "dense_shell"],
        default="sparse",
    )
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument(
        "--rollout-target",
        choices=["trajectory", "final", "none"],
        default="trajectory",
    )
    parser.add_argument(
        "--rollout-loss-mode",
        choices=["all", "terminal"],
        default="all",
    )
    parser.add_argument("--rollout-mse-weight", type=float, default=0.0)
    parser.add_argument("--rollout-direction-weight", type=float, default=0.0)
    parser.add_argument("--gradient-mse-weight", type=float, default=1.0)
    parser.add_argument("--gradient-direction-weight", type=float, default=0.1)
    parser.add_argument(
        "--gradient-target",
        choices=["total", "reference", "residual"],
        default="total",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau"],
        default="none",
    )
    parser.add_argument(
        "--best-metric",
        choices=[
            "objective_loss",
            "hamiltonian_rmse",
            "energy_rmse",
            "orbital_residual_rmse",
            "rollout_rmse",
        ],
        default="objective_loss",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    config_tokens = _config_to_cli_tokens(parser, config_args.config)
    args = parser.parse_args(config_tokens + remaining_argv)
    args.config = config_args.config
    return args


def _split_csv_arg(value: str) -> set[str] | None:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    return set(tokens) if tokens else None


def _load_yaml_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    if "train" in raw:
        train_section = raw["train"]
        if not isinstance(train_section, dict):
            raise ValueError(f"`train` section must be a mapping: {path}")
        return train_section
    return raw


def _config_to_cli_tokens(
    parser: argparse.ArgumentParser,
    config_path: Path | None,
) -> list[str]:
    if config_path is None:
        return []

    config = _load_yaml_config(config_path)
    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.option_strings
    }
    tokens: list[str] = []
    unknown_keys: list[str] = []

    for raw_key, raw_value in config.items():
        dest = str(raw_key).replace("-", "_")
        action = action_by_dest.get(dest)
        if action is None or dest == "help":
            unknown_keys.append(str(raw_key))
            continue
        if raw_value is None:
            continue

        option = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            tokens.append(option if bool(raw_value) else f"--no-{dest.replace('_', '-')}")
            continue

        if isinstance(raw_value, list):
            value = ",".join(str(item) for item in raw_value)
        else:
            value = raw_value

        if isinstance(value, str):
            value = os.path.expandvars(os.path.expanduser(value))
        tokens.extend([option, str(value)])

    if unknown_keys:
        raise ValueError(
            f"unknown config keys in {config_path}: {sorted(unknown_keys)}"
        )
    return tokens


def _uniform_selection_offsets(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count <= 1:
        return [length - 1]

    chosen: list[int] = []
    seen: set[int] = set()
    for position in np.linspace(0, length - 1, num=count).tolist():
        index = int(round(position))
        if index not in seen:
            chosen.append(index)
            seen.add(index)
    if len(chosen) < count:
        for index in range(length):
            if index in seen:
                continue
            chosen.append(index)
            seen.add(index)
            if len(chosen) == count:
                break
    return sorted(chosen[:count])


def _resolve_split_sample_names(
    dataset: JAXStepDataset,
    args: argparse.Namespace,
) -> dict[str, list[str]]:
    valid_fraction = float(args.valid_fraction)
    test_fraction = float(args.test_fraction)
    if valid_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("split fractions must be non-negative")
    if valid_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than 1")

    all_sample_names = dataset.sample_names
    all_sample_name_set = set(all_sample_names)
    valid_samples = _split_csv_arg(args.valid_samples) or set()
    test_samples = _split_csv_arg(args.test_samples) or set()

    unknown_valid = valid_samples - all_sample_name_set
    unknown_test = test_samples - all_sample_name_set
    if unknown_valid:
        raise ValueError(f"unknown validation samples: {sorted(unknown_valid)}")
    if unknown_test:
        raise ValueError(f"unknown test samples: {sorted(unknown_test)}")
    overlap = valid_samples & test_samples
    if overlap:
        raise ValueError(
            f"samples cannot appear in both validation and test: {sorted(overlap)}"
        )

    remaining_samples = [
        sample_name
        for sample_name in all_sample_names
        if sample_name not in valid_samples and sample_name not in test_samples
    ]
    rng = np.random.default_rng(args.split_seed)
    shuffled = remaining_samples.copy()
    rng.shuffle(shuffled)

    n_valid = int(round(valid_fraction * len(shuffled)))
    n_test = int(round(test_fraction * len(shuffled)))

    valid_auto = shuffled[:n_valid]
    test_auto = shuffled[n_valid : n_valid + n_test]
    train_samples = shuffled[n_valid + n_test :]

    valid_sample_names = sorted(valid_samples | set(valid_auto))
    test_sample_names = sorted(test_samples | set(test_auto))
    train_sample_names = sorted(train_samples)

    if not train_sample_names:
        raise ValueError("train split is empty after applying geometry filters")

    return {
        "train": train_sample_names,
        "valid": valid_sample_names,
        "test": test_sample_names,
    }


def _select_step_indices(
    dataset: JAXStepDataset,
    sample_names: list[str],
    *,
    max_steps_per_sample: int,
    step_selection: str,
) -> list[int]:
    indices_by_sample = dataset.indices_by_sample()
    selected: list[int] = []
    for sample_name in sample_names:
        sample_indices = list(indices_by_sample[sample_name])
        if (
            max_steps_per_sample <= 0
            or step_selection == "all"
            or len(sample_indices) <= max_steps_per_sample
        ):
            selected.extend(sample_indices)
            continue

        if step_selection == "final":
            chosen = sample_indices[-max_steps_per_sample:]
        elif step_selection == "first":
            chosen = sample_indices[:max_steps_per_sample]
        elif step_selection == "uniform":
            offsets = _uniform_selection_offsets(len(sample_indices), max_steps_per_sample)
            chosen = [sample_indices[offset] for offset in offsets]
        else:
            raise ValueError(f"unsupported step selection mode: {step_selection}")
        selected.extend(chosen)
    return selected


def _torch_dtype_from_arg(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {value}")


def _build_model(dataset: JAXStepDataset, args: argparse.Namespace) -> Any:
    common_kwargs = {
        "max_atomic_number": max(dataset.max_atomic_number(), 1),
        "max_angular_momentum": max(dataset.max_angular_momentum(), 1),
        "max_ao_local_index": max(dataset.max_ao_local_index(), 1),
        "hidden_dim": args.hidden_dim,
        "pair_hidden_dim": args.pair_hidden_dim,
        "predict_reference_residual": args.predict_reference_residual,
    }
    if args.model_type == "baseline":
        return DeepVBHBaseline(**common_kwargs)
    if args.model_type == "e3nn":
        return DeepVBHE3Model(
            **common_kwargs,
            atom_channels=args.atom_channels or None,
            ao_scalar_channels=args.ao_scalar_channels or None,
            lmax=args.lmax,
            num_radial=args.num_radial,
            cutoff=args.cutoff,
        )
    raise ValueError(f"unsupported model type: {args.model_type}")


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


def _default_best_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}.best{path.suffix}")


def _resolve_best_metric_name(
    args: argparse.Namespace,
    *,
    has_validation: bool,
) -> str:
    prefix = "valid" if has_validation else "train"
    return f"{prefix}_{args.best_metric}"


def _make_gradient_disabled_weights(weights: DeepVBHLossWeights) -> DeepVBHLossWeights:
    return DeepVBHLossWeights(
        hamiltonian=weights.hamiltonian,
        energy=weights.energy,
        orbital_residual_mse=weights.orbital_residual_mse,
        orbital_residual_direction=weights.orbital_residual_direction,
        rollout_mse=weights.rollout_mse,
        rollout_direction=weights.rollout_direction,
        gradient_mse=0.0,
        gradient_direction=0.0,
    )


def _weighted_mean(values: Any, batch_mask: Any) -> Any:
    return jnp.sum(values * batch_mask) / jnp.clip(jnp.sum(batch_mask), a_min=1.0)


def _build_optimizer(args: argparse.Namespace) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if args.grad_clip_norm > 0.0:
        transforms.append(optax.clip_by_global_norm(args.grad_clip_norm))
    transforms.append(optax.scale_by_adam())
    if args.weight_decay > 0.0:
        transforms.append(optax.add_decayed_weights(args.weight_decay))
    return optax.chain(*transforms)


def _sanitize_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: jnp.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0),
        tree,
    )


def _aggregate_metrics(
    metrics: dict[str, Any],
    count: int,
    totals: dict[str, float] | None = None,
) -> dict[str, float]:
    result = {} if totals is None else totals
    for key, value in metrics.items():
        result[key] = result.get(key, 0.0) + float(np.asarray(value)) * count
    return result


def _finalize_metrics(
    prefix: str,
    metric_totals: dict[str, float],
    total_count: int,
) -> dict[str, float]:
    if total_count == 0:
        return {}
    return {
        f"{prefix}_loss": metric_totals["display_loss"] / total_count,
        f"{prefix}_objective_loss": metric_totals["loss"] / total_count,
        f"{prefix}_hamiltonian_mae": metric_totals["hamiltonian_mae"] / total_count,
        f"{prefix}_hamiltonian_rmse": metric_totals["hamiltonian_rmse"] / total_count,
        f"{prefix}_energy_mae": metric_totals["energy_mae"] / total_count,
        f"{prefix}_energy_rmse": metric_totals["energy_rmse"] / total_count,
        f"{prefix}_orbital_residual_mae": metric_totals["orbital_residual_mae"] / total_count,
        f"{prefix}_orbital_residual_rmse": metric_totals["orbital_residual_rmse"] / total_count,
        f"{prefix}_orbital_residual_cosine_similarity": (
            metric_totals["orbital_residual_cosine_similarity"] / total_count
        ),
        f"{prefix}_orbital_residual_direction_loss": (
            metric_totals["orbital_residual_direction_loss"] / total_count
        ),
        f"{prefix}_rollout_mae": metric_totals["rollout_mae"] / total_count,
        f"{prefix}_rollout_rmse": metric_totals["rollout_rmse"] / total_count,
        f"{prefix}_rollout_cosine_similarity": (
            metric_totals["rollout_cosine_similarity"] / total_count
        ),
        f"{prefix}_rollout_direction_loss": (
            metric_totals["rollout_direction_loss"] / total_count
        ),
        f"{prefix}_gradient_mae": metric_totals["gradient_mae"] / total_count,
        f"{prefix}_gradient_rmse": metric_totals["gradient_rmse"] / total_count,
        f"{prefix}_gradient_cosine_similarity": (
            metric_totals["gradient_cosine_similarity"] / total_count
        ),
        f"{prefix}_gradient_direction_loss": (
            metric_totals["gradient_direction_loss"] / total_count
        ),
    }


def _save_checkpoint(
    path: Path,
    params: Any,
    opt_state: Any,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    *,
    split_summary: dict[str, Any],
    lr: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "args": vars(args),
        "history": history,
        "split_summary": split_summary,
        "lr": lr,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


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


def _load_init_params(
    initialized_params: Any,
    checkpoint_path: Path,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = _load_checkpoint(checkpoint_path)
    template_tree = flax.core.unfreeze(initialized_params)
    loaded_tree = flax.core.unfreeze(checkpoint["params"])
    merged_tree, reused_leaves, total_leaves = _merge_param_tree(
        template_tree,
        loaded_tree,
    )
    return flax.core.freeze(merged_tree), {
        "init_checkpoint": str(checkpoint_path),
        "reused_param_leaves": reused_leaves,
        "total_param_leaves": total_leaves,
    }


def _build_step_functions(
    model: Any,
    optimizer: optax.GradientTransformation,
    weights: DeepVBHLossWeights,
    loss_scales: DeepVBHLossScales,
    *,
    orbital_update_space: str,
    rollout_steps: int,
    rollout_loss_mode: str,
    layout: dict[str, Any] | None = None,
) -> dict[str, Callable[..., Any]]:
    gradient_disabled_weights = _make_gradient_disabled_weights(weights)

    if orbital_update_space == "dense_shell" and layout is None:
        raise ValueError("dense_shell orbital updates require the e3nn forward_dense path")

    def _apply_model(params: Any, sample: dict[str, Any], orbital_state: Any) -> Any:
        if orbital_update_space == "dense_shell":
            return model.apply(
                {"params": params},
                sample,
                orbital_state,
                layout=layout,
                method=model.forward_dense,
            )
        if layout is None:
            return model.apply({"params": params}, sample, orbital_state)
        return model.apply(
            {"params": params},
            sample,
            orbital_state,
            layout=layout,
        )

    def _initial_orbital_state(sample: dict[str, Any]) -> Any:
        return sample["input_orbital_state"]

    def _predicted_orbital_update(sample: dict[str, Any], output: Any) -> Any:
        if orbital_update_space == "dense_shell":
            if output.dense_orbital_residual is None:
                raise ValueError("dense_shell orbital updates require dense_orbital_residual")
            mask = sample["orbital_update_mask"].astype(output.dense_orbital_residual.dtype)
            return output.dense_orbital_residual * mask
        return output.orbital_residual

    def _predict_rollout_deltas(
        params: Any,
        sample: dict[str, Any],
        initial_orbital_state: Any,
        *,
        first_output: Any | None = None,
    ) -> Any | None:
        if rollout_steps <= 0 or (
            weights.rollout_mse == 0.0 and weights.rollout_direction == 0.0
        ):
            return None
        current_orbital_state = initial_orbital_state
        rollout_deltas: list[Any] = []
        for rollout_index in range(rollout_steps):
            if rollout_index == 0 and first_output is not None:
                output = first_output
            else:
                output = _apply_model(params, sample, current_orbital_state)
            current_orbital_state = current_orbital_state + _predicted_orbital_update(sample, output)
            rollout_deltas.append(current_orbital_state - initial_orbital_state)
        return jnp.stack(rollout_deltas, axis=0)

    def _single_loss_no_grad(params: Any, sample: dict[str, Any]) -> dict[str, Any]:
        orbital_state = _initial_orbital_state(sample)
        output = _apply_model(params, sample, orbital_state)
        predicted_orbital_residual = _predicted_orbital_update(sample, output)
        rollout_deltas = _predict_rollout_deltas(
            params,
            sample,
            orbital_state,
            first_output=output,
        )
        return compute_losses(
            sample,
            output,
            predicted_orbital_residual,
            None,
            rollout_deltas,
            weights=gradient_disabled_weights,
            scales=loss_scales,
            rollout_loss_mode=rollout_loss_mode,
        )

    def _single_loss_with_grad(params: Any, sample: dict[str, Any]) -> dict[str, Any]:
        orbital_state = _initial_orbital_state(sample)

        def _energy_with_aux(orbital_values: Any) -> tuple[Any, Any]:
            output = _apply_model(params, sample, orbital_values)
            return output.total_energy, output

        (total_energy, output), predicted_gradient = jax.value_and_grad(
            _energy_with_aux,
            has_aux=True,
        )(orbital_state)
        del total_energy
        predicted_orbital_residual = _predicted_orbital_update(sample, output)
        rollout_deltas = _predict_rollout_deltas(
            params,
            sample,
            orbital_state,
            first_output=output,
        )
        return compute_losses(
            sample,
            output,
            predicted_orbital_residual,
            predicted_gradient,
            rollout_deltas,
            weights=weights,
            scales=loss_scales,
            rollout_loss_mode=rollout_loss_mode,
        )

    def _batch_metrics(
        params: Any,
        batch: dict[str, Any],
        *,
        with_gradient: bool,
    ) -> dict[str, Any]:
        single_loss = _single_loss_with_grad if with_gradient else _single_loss_no_grad
        per_sample_losses = jax.vmap(single_loss, in_axes=(None, 0))(params, batch)
        batch_mask = batch["batch_mask"]
        return {
            key: _weighted_mean(value, batch_mask)
            for key, value in per_sample_losses.items()
        }

    def _objective(
        params: Any,
        batch: dict[str, Any],
        *,
        with_gradient: bool,
    ) -> tuple[Any, dict[str, Any]]:
        metrics = _batch_metrics(params, batch, with_gradient=with_gradient)
        return metrics["loss"], metrics

    @jax.jit
    def train_step_no_grad(
        params: Any,
        opt_state: Any,
        batch: dict[str, Any],
        learning_rate: float,
    ) -> tuple[Any, Any, dict[str, Any]]:
        (_, metrics), grads = jax.value_and_grad(
            lambda current_params: _objective(
                current_params,
                batch,
                with_gradient=False,
            ),
            has_aux=True,
        )(params)
        grads = _sanitize_tree(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        scaled_updates = jax.tree_util.tree_map(
            lambda value: -learning_rate * value,
            updates,
        )
        params = optax.apply_updates(params, scaled_updates)
        return params, opt_state, metrics

    @jax.jit
    def train_step_with_grad(
        params: Any,
        opt_state: Any,
        batch: dict[str, Any],
        learning_rate: float,
    ) -> tuple[Any, Any, dict[str, Any]]:
        (_, metrics), grads = jax.value_and_grad(
            lambda current_params: _objective(
                current_params,
                batch,
                with_gradient=True,
            ),
            has_aux=True,
        )(params)
        grads = _sanitize_tree(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        scaled_updates = jax.tree_util.tree_map(
            lambda value: -learning_rate * value,
            updates,
        )
        params = optax.apply_updates(params, scaled_updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_step_no_grad(
        params: Any,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        return _batch_metrics(params, batch, with_gradient=False)

    @jax.jit
    def eval_step_with_grad(
        params: Any,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        return _batch_metrics(params, batch, with_gradient=True)

    return {
        "train_no_grad": train_step_no_grad,
        "train_with_grad": train_step_with_grad,
        "eval_no_grad": eval_step_no_grad,
        "eval_with_grad": eval_step_with_grad,
    }


def _maybe_validate_e3nn_layout(dataset: JAXStepDataset, indices: list[int]) -> None:
    seen_signatures: set[Any] = set()
    for index in indices:
        sample = dataset.numeric_sample(index)
        signature = (
            tuple(np.asarray(sample["shell_angular_momenta"]).tolist()),
            tuple(np.asarray(sample["shell_ao_starts"]).tolist()),
            tuple(np.asarray(sample["shell_ao_counts"]).tolist()),
        )
        if signature in seen_signatures:
            continue
        validate_cartesian_shell_layout(sample)
        seen_signatures.add(signature)


def main() -> None:
    args = _parse_args()
    if args.orbital_update_space == "dense_shell" and args.model_type != "e3nn":
        raise ValueError("dense_shell orbital updates currently require --model-type e3nn")
    if args.orbital_update_space == "dense_shell" and (
        args.gradient_mse_weight != 0.0 or args.gradient_direction_weight != 0.0
    ):
        raise ValueError(
            "dense_shell orbital updates do not yet support gradient supervision; "
            "set gradient weights to zero for this stage"
        )
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    np.random.seed(args.seed)
    device = _resolve_device(args.device)
    include_samples = _split_csv_arg(args.include_samples)
    exclude_samples = _split_csv_arg(args.exclude_samples)

    dataset = JAXStepDataset(
        str(args.data_root),
        dtype=_torch_dtype_from_arg(args.dtype),
        include_samples=include_samples,
        exclude_samples=exclude_samples,
        cache_steps=args.cache_steps,
        preload_steps=args.preload_steps,
        rollout_horizon=args.rollout_steps,
    )
    split_sample_names = _resolve_split_sample_names(dataset, args)
    train_indices = _select_step_indices(
        dataset,
        split_sample_names["train"],
        max_steps_per_sample=args.max_steps_per_sample,
        step_selection=args.step_selection,
    )
    valid_indices = _select_step_indices(
        dataset,
        split_sample_names["valid"],
        max_steps_per_sample=args.max_steps_per_sample,
        step_selection=args.step_selection,
    )
    test_indices = _select_step_indices(
        dataset,
        split_sample_names["test"],
        max_steps_per_sample=args.max_steps_per_sample,
        step_selection=args.step_selection,
    )
    if args.model_type == "e3nn":
        _maybe_validate_e3nn_layout(dataset, train_indices + valid_indices + test_indices)

    loss_scale_values = (
        dataset.compute_loss_scales(
            train_indices,
            args.gradient_target,
            args.orbital_residual_target,
            args.orbital_update_space,
            args.rollout_target,
        )
        if args.normalize_losses
        else {
            "hamiltonian": 1.0,
            "energy": 1.0,
            "orbital_residual": 1.0,
            "rollout_orbital": 1.0,
            "gradient": 1.0,
        }
    )
    loss_scales = DeepVBHLossScales(**loss_scale_values)

    split_summary: dict[str, Any] = {
        "framework": "jax",
        "train_samples": split_sample_names["train"],
        "valid_samples": split_sample_names["valid"],
        "test_samples": split_sample_names["test"],
        "train_steps": len(train_indices),
        "valid_steps": len(valid_indices),
        "test_steps": len(test_indices),
        "max_steps_per_sample": args.max_steps_per_sample,
        "step_selection": args.step_selection,
        "gradient_loss_interval": args.gradient_loss_interval,
        "orbital_residual_target": args.orbital_residual_target,
        "orbital_update_space": args.orbital_update_space,
        "rollout_steps": args.rollout_steps,
        "rollout_target": args.rollout_target,
        "rollout_loss_mode": args.rollout_loss_mode,
        "batch_size": args.batch_size,
        "loss_scales": loss_scale_values,
        "device": f"{device.platform}:{device.id}",
        "dtype": args.dtype,
        "predict_reference_residual": bool(args.predict_reference_residual),
        "best_metric": args.best_metric,
        "config": None if args.config is None else str(args.config),
        "init_checkpoint": None if args.init_checkpoint is None else str(args.init_checkpoint),
        "min_lr": float(args.min_lr),
    }
    print(json.dumps(split_summary, ensure_ascii=False), flush=True)

    model = _build_model(dataset, args)
    example_sample = jax.device_put(dataset.example_sample(train_indices[0]), device=device)
    example_layout = None
    if args.model_type == "e3nn":
        example_layout = dataset.bucket_layout(dataset.bucket_signature_at(train_indices[0]))
    if example_layout is None:
        params = model.init(
            jax.random.PRNGKey(args.seed),
            example_sample,
            example_sample["orbital_value_table"],
        )["params"]
    else:
        params = model.init(
            jax.random.PRNGKey(args.seed),
            example_sample,
            example_sample["orbital_value_table"],
            layout=example_layout,
        )["params"]
    if args.init_checkpoint is not None:
        params, init_summary = _load_init_params(params, args.init_checkpoint)
        print(json.dumps(init_summary, ensure_ascii=False), flush=True)

    optimizer = _build_optimizer(args)
    opt_state = optimizer.init(params)

    weights = DeepVBHLossWeights(
        hamiltonian=args.hamiltonian_weight,
        energy=args.energy_weight,
        orbital_residual_mse=args.orbital_residual_mse_weight,
        orbital_residual_direction=args.orbital_residual_direction_weight,
        rollout_mse=args.rollout_mse_weight,
        rollout_direction=args.rollout_direction_weight,
        gradient_mse=args.gradient_mse_weight,
        gradient_direction=args.gradient_direction_weight,
    )
    step_function_cache: dict[Any, dict[str, Callable[..., Any]]] = {}

    def _step_functions_for(bucket_signature: Any) -> dict[str, Callable[..., Any]]:
        cached = step_function_cache.get(bucket_signature)
        if cached is not None:
            return cached
        layout = dataset.bucket_layout(bucket_signature) if args.model_type == "e3nn" else None
        cached = _build_step_functions(
            model,
            optimizer,
            weights,
            loss_scales,
            orbital_update_space=args.orbital_update_space,
            rollout_steps=args.rollout_steps,
            rollout_loss_mode=args.rollout_loss_mode,
            layout=layout,
        )
        step_function_cache[bucket_signature] = cached
        return cached

    best_checkpoint_path = args.best_checkpoint or _default_best_checkpoint(args.checkpoint)
    best_metric_name = _resolve_best_metric_name(
        args,
        has_validation=bool(valid_indices),
    )
    best_metric = float("inf")
    best_saved = False
    current_lr = float(args.lr)
    scheduler_best = float("inf")
    scheduler_bad_epochs = 0
    history: list[dict[str, Any]] = []
    global_batch_index = 0
    gradients_enabled = (
        args.gradient_mse_weight != 0.0 or args.gradient_direction_weight != 0.0
    )

    for epoch in range(1, args.epochs + 1):
        train_totals: dict[str, float] = {}
        train_count = 0
        train_batches = dataset.iter_batches(
            train_indices,
            batch_size=args.batch_size,
            gradient_target=args.gradient_target,
            orbital_residual_target=args.orbital_residual_target,
            orbital_update_space=args.orbital_update_space,
            rollout_target=args.rollout_target,
            shuffle=True,
            seed=args.seed + epoch,
            device=device,
        )
        for batch_record in train_batches:
            step_functions = _step_functions_for(batch_record["bucket_signature"])
            use_gradient = gradients_enabled and (
                global_batch_index % max(args.gradient_loss_interval, 1) == 0
            )
            step_key = "train_with_grad" if use_gradient else "train_no_grad"
            params, opt_state, metrics = step_functions[step_key](
                params,
                opt_state,
                batch_record["batch"],
                current_lr,
            )
            batch_count = int(batch_record["host_batch_size"])
            train_totals = _aggregate_metrics(metrics, batch_count, train_totals)
            train_count += batch_count
            global_batch_index += 1

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "gradient_target": args.gradient_target,
            "lr": current_lr,
        }
        epoch_record.update(_finalize_metrics("train", train_totals, train_count))

        if valid_indices:
            valid_totals: dict[str, float] = {}
            valid_count = 0
            valid_batches = dataset.iter_batches(
                valid_indices,
                batch_size=args.batch_size,
                gradient_target=args.gradient_target,
                orbital_residual_target=args.orbital_residual_target,
                orbital_update_space=args.orbital_update_space,
                rollout_target=args.rollout_target,
                shuffle=False,
                seed=args.seed,
                device=device,
            )
            eval_key = "eval_with_grad" if gradients_enabled else "eval_no_grad"
            for batch_record in valid_batches:
                step_functions = _step_functions_for(batch_record["bucket_signature"])
                metrics = step_functions[eval_key](params, batch_record["batch"])
                batch_count = int(batch_record["host_batch_size"])
                valid_totals = _aggregate_metrics(metrics, batch_count, valid_totals)
                valid_count += batch_count
            epoch_record.update(_finalize_metrics("valid", valid_totals, valid_count))

        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        current_metric = float(epoch_record[best_metric_name])
        if args.scheduler == "plateau":
            if current_metric < scheduler_best:
                scheduler_best = current_metric
                scheduler_bad_epochs = 0
            else:
                scheduler_bad_epochs += 1
                if scheduler_bad_epochs > args.lr_patience:
                    reduced_lr = current_lr * args.lr_factor
                    new_lr = max(float(args.min_lr), reduced_lr)
                    if new_lr < current_lr:
                        current_lr = new_lr
                        scheduler_bad_epochs = 0

        if current_metric < best_metric:
            best_metric = current_metric
            _save_checkpoint(
                best_checkpoint_path,
                params,
                opt_state,
                args,
                history,
                split_summary=split_summary,
                lr=current_lr,
            )
            best_saved = True

    _save_checkpoint(
        args.checkpoint,
        params,
        opt_state,
        args,
        history,
        split_summary=split_summary,
        lr=current_lr,
    )

    if test_indices:
        if best_saved:
            best_state = _load_checkpoint(best_checkpoint_path)
            params = best_state["params"]
        test_totals: dict[str, float] = {}
        test_count = 0
        test_batches = dataset.iter_batches(
            test_indices,
            batch_size=args.batch_size,
            gradient_target=args.gradient_target,
            orbital_residual_target=args.orbital_residual_target,
            orbital_update_space=args.orbital_update_space,
            rollout_target=args.rollout_target,
            shuffle=False,
            seed=args.seed,
            device=device,
        )
        eval_key = "eval_with_grad" if gradients_enabled else "eval_no_grad"
        for batch_record in test_batches:
            step_functions = _step_functions_for(batch_record["bucket_signature"])
            metrics = step_functions[eval_key](params, batch_record["batch"])
            batch_count = int(batch_record["host_batch_size"])
            test_totals = _aggregate_metrics(metrics, batch_count, test_totals)
            test_count += batch_count
        test_record = _finalize_metrics("test", test_totals, test_count)
        print(json.dumps(test_record, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "best_checkpoint": str(best_checkpoint_path) if best_saved else None,
                "epochs": args.epochs,
                "samples": dataset.sample_names,
                "steps": len(dataset),
                "split_summary": split_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
