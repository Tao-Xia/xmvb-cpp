from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .dataset import DeepVBHStepDataset, move_sample_to_device
from .e3nn_model import DeepVBHE3Model
from .losses import DeepVBHLossScales, DeepVBHLossWeights, compute_losses
from .model import DeepVBHBaseline


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
        description="Train DeepVBH Torch models",
        parents=[config_parser],
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("deepvbscf_data"),
        help="Root directory containing dumped DeepVBSCF samples",
    )
    parser.add_argument("--epochs", type=int, default=50)
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/deepVBH_baseline.pt"),
    )
    parser.add_argument(
        "--best-checkpoint",
        type=Path,
        default=None,
        help="Optional path for the best validation checkpoint",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint whose model weights are used to initialize training",
    )
    parser.add_argument(
        "--include-samples",
        type=str,
        default="",
        help="Comma-separated sample names to include before splitting",
    )
    parser.add_argument(
        "--exclude-samples",
        type=str,
        default="",
        help="Comma-separated sample names to exclude before splitting",
    )
    parser.add_argument(
        "--valid-samples",
        type=str,
        default="",
        help="Comma-separated geometry names reserved for validation",
    )
    parser.add_argument(
        "--test-samples",
        type=str,
        default="",
        help="Comma-separated geometry names reserved for test",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.0,
        help="Geometry-level validation fraction sampled from the remaining geometries",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.0,
        help="Geometry-level test fraction sampled from the remaining geometries",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--max-steps-per-sample",
        type=int,
        default=0,
        help="Keep at most this many SCF steps per geometry; 0 keeps all steps",
    )
    parser.add_argument(
        "--step-selection",
        choices=["all", "final", "uniform", "first"],
        default="final",
        help="How to subsample steps when --max-steps-per-sample is active",
    )
    parser.add_argument(
        "--gradient-loss-interval",
        type=int,
        default=1,
        help="Only every Nth training step carries gradient supervision; 1 keeps all steps",
    )
    parser.add_argument(
        "--cache-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache step tensors in memory after their first load",
    )
    parser.add_argument(
        "--preload-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Eagerly load all step tensors into memory at startup",
    )
    parser.add_argument(
        "--cache-device-samples",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache moved samples on the training device across epochs",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap the model with torch.compile when available",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW on CUDA when supported",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Clip gradient norm after backward; 0 disables clipping",
    )
    parser.add_argument(
        "--normalize-losses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize Hamiltonian, energy, and gradient MSE terms by train-set target scales",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/convolution on CUDA for float32 training",
    )
    parser.add_argument("--hamiltonian-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
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
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    config_tokens = _config_to_cli_tokens(parser, config_args.config)
    args = parser.parse_args(config_tokens + remaining_argv)
    args.config = config_args.config
    return args


def _split_csv_arg(value: str) -> set[str] | None:
    tokens = [item.strip() for item in value.split(",") if item.strip()]
    return set(tokens) if tokens else None


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            "YAML config support requires PyYAML. Run `uv sync --extra geom` first."
        ) from error

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
        if action is None or dest in {"help"}:
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


def _mean_metric(metrics: list[dict[str, float]], key: str) -> float:
    if not metrics:
        return 0.0
    return sum(item[key] for item in metrics) / len(metrics)


def _build_model(dataset: DeepVBHStepDataset, args: argparse.Namespace) -> nn.Module:
    common_kwargs = {
        "max_atomic_number": max(dataset.max_atomic_number(), 1),
        "max_angular_momentum": max(dataset.max_angular_momentum(), 1),
        "max_ao_local_index": max(dataset.max_ao_local_index(), 1),
        "hidden_dim": args.hidden_dim,
        "pair_hidden_dim": args.pair_hidden_dim,
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


def _dtype_from_arg(value: str) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {value}")


def _step_metrics(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu().item())
        for key, value in losses.items()
    }


def _prefixed_epoch_metrics(
    prefix: str,
    metrics: list[dict[str, float]],
) -> dict[str, float]:
    if not metrics:
        return {}
    return {
        f"{prefix}_loss": _mean_metric(metrics, "display_loss"),
        f"{prefix}_objective_loss": _mean_metric(metrics, "loss"),
        f"{prefix}_hamiltonian_mae": _mean_metric(metrics, "hamiltonian_mae"),
        f"{prefix}_hamiltonian_rmse": _mean_metric(metrics, "hamiltonian_rmse"),
        f"{prefix}_energy_mae": _mean_metric(metrics, "energy_mae"),
        f"{prefix}_energy_rmse": _mean_metric(metrics, "energy_rmse"),
        f"{prefix}_gradient_mae": _mean_metric(metrics, "gradient_mae"),
        f"{prefix}_gradient_rmse": _mean_metric(metrics, "gradient_rmse"),
        f"{prefix}_gradient_cosine_similarity": _mean_metric(
            metrics,
            "gradient_cosine_similarity",
        ),
        f"{prefix}_gradient_direction_loss": _mean_metric(
            metrics,
            "gradient_direction_loss",
        ),
    }


def _default_best_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}.best{path.suffix}")


def _configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.benchmark = True


def _maybe_compile_model(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch build")
    return torch.compile(model, mode=args.compile_mode)


def _load_initial_model_state(
    model: nn.Module,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _resolve_split_sample_names(
    dataset: DeepVBHStepDataset,
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
        raise ValueError(f"samples cannot appear in both validation and test: {sorted(overlap)}")

    remaining_samples = [
        sample_name
        for sample_name in all_sample_names
        if sample_name not in valid_samples and sample_name not in test_samples
    ]
    if remaining_samples:
        generator = torch.Generator().manual_seed(args.split_seed)
        permutation = torch.randperm(len(remaining_samples), generator=generator).tolist()
        shuffled = [remaining_samples[index] for index in permutation]
    else:
        shuffled = []

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


def _uniform_selection_offsets(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count <= 1:
        return [length - 1]

    chosen: list[int] = []
    seen: set[int] = set()
    for position in torch.linspace(0, length - 1, steps=count).tolist():
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


def _select_step_indices(
    dataset: DeepVBHStepDataset,
    sample_names: list[str],
    *,
    max_steps_per_sample: int,
    step_selection: str,
) -> list[int]:
    indices_by_sample = dataset.indices_by_sample()
    selected: list[int] = []
    for sample_name in sample_names:
        sample_indices = list(indices_by_sample[sample_name])
        if max_steps_per_sample <= 0 or step_selection == "all" or len(sample_indices) <= max_steps_per_sample:
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


def _make_gradient_disabled_weights(weights: DeepVBHLossWeights) -> DeepVBHLossWeights:
    return DeepVBHLossWeights(
        hamiltonian=weights.hamiltonian,
        energy=weights.energy,
        gradient_mse=0.0,
        gradient_direction=0.0,
    )


def _sample_to_device(
    dataset: DeepVBHStepDataset,
    sample_index: int,
    device: torch.device,
    *,
    cache: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if cache is not None:
        cached = cache.get(sample_index)
        if cached is not None:
            return cached
    sample = move_sample_to_device(dataset[sample_index], device)
    if cache is not None:
        cache[sample_index] = sample
    return sample


def _compute_loss_scales(
    dataset: DeepVBHStepDataset,
    indices: list[int],
    gradient_target: str,
    *,
    epsilon: float = 1.0e-8,
) -> DeepVBHLossScales:
    if not indices:
        return DeepVBHLossScales()

    h_terms: list[torch.Tensor] = []
    energy_terms: list[float] = []
    gradient_terms: list[torch.Tensor] = []
    gradient_target_map = {
        "total": "sparse_orbital_energy_gradient",
        "reference": "sparse_orbital_reference_energy_gradient",
        "residual": "sparse_orbital_residual_energy_gradient",
    }
    target_gradient_key = gradient_target_map[gradient_target]

    for sample_index in indices:
        sample = dataset[sample_index]
        hamiltonian = sample["two_electron_hamiltonian_matrix"]
        row, column = torch.tril_indices(
            hamiltonian.shape[0],
            hamiltonian.shape[1],
        )
        h_terms.append(hamiltonian[row, column].reshape(-1))
        energy_terms.append(float(sample["target_total_energy"].item()))
        gradient = sample[target_gradient_key].reshape(-1).index_select(
            0,
            sample["differentiable_parameter_indices"],
        )
        gradient_terms.append(gradient)

    h_scale = torch.cat(h_terms).square().mean().sqrt().item()
    energy_tensor = torch.tensor(energy_terms, dtype=torch.float64)
    if energy_tensor.numel() > 1:
        energy_scale = energy_tensor.std(unbiased=False).item()
    else:
        energy_scale = abs(float(energy_tensor[0].item()))
    gradient_scale = torch.cat(gradient_terms).square().mean().sqrt().item()
    return DeepVBHLossScales(
        hamiltonian=max(h_scale, epsilon),
        energy=max(energy_scale, epsilon),
        gradient=max(gradient_scale, epsilon),
    )


def _run_epoch(
    model: nn.Module,
    dataset: DeepVBHStepDataset,
    indices: list[int],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    weights: DeepVBHLossWeights,
    loss_scales: DeepVBHLossScales,
    gradient_target: str,
    *,
    training: bool,
    gradient_loss_interval: int = 1,
    grad_clip_norm: float = 0.0,
    sample_cache: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, float]]:
    if not indices:
        return []

    metrics: list[dict[str, float]] = []
    if training:
        model.train()
    else:
        model.eval()

    gradient_loss_interval = max(int(gradient_loss_interval), 1)
    weights_without_gradient = _make_gradient_disabled_weights(weights)
    order = (
        [indices[index] for index in torch.randperm(len(indices)).tolist()]
        if training
        else list(indices)
    )

    for order_index, sample_index in enumerate(order):
        sample = _sample_to_device(
            dataset,
            sample_index,
            device,
            cache=sample_cache,
        )
        orbital_value_table = (
            sample["orbital_value_table"].clone().detach().requires_grad_(True)
        )

        use_gradient_supervision = (
            not training or order_index % gradient_loss_interval == 0
        )
        effective_weights = weights if use_gradient_supervision else weights_without_gradient
        need_predicted_gradient = (
            effective_weights.gradient_mse != 0.0
            or effective_weights.gradient_direction != 0.0
        )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            output = model(sample, orbital_value_table)
            predicted_gradient = None
            if need_predicted_gradient:
                predicted_gradient = torch.autograd.grad(
                    output.total_energy,
                    orbital_value_table,
                    create_graph=training,
                )[0]
            losses = compute_losses(
                sample,
                output,
                predicted_gradient,
                weights=effective_weights,
                scales=loss_scales,
                gradient_target=gradient_target,
            )
            if training:
                losses["loss"].backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        metrics.append(_step_metrics(losses))

    return metrics


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    *,
    split_summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "history": history,
            "split_summary": split_summary,
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = _dtype_from_arg(args.dtype)
    device = torch.device(args.device)
    _configure_runtime(args, device)
    include_samples = _split_csv_arg(args.include_samples)
    exclude_samples = _split_csv_arg(args.exclude_samples)

    dataset = DeepVBHStepDataset(
        args.data_root,
        dtype=dtype,
        include_samples=include_samples,
        exclude_samples=exclude_samples,
        cache_steps=args.cache_steps,
        preload_steps=args.preload_steps,
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
    split_summary: dict[str, Any] = {
        "train_samples": split_sample_names["train"],
        "valid_samples": split_sample_names["valid"],
        "test_samples": split_sample_names["test"],
        "train_steps": len(train_indices),
        "valid_steps": len(valid_indices),
        "test_steps": len(test_indices),
        "max_steps_per_sample": args.max_steps_per_sample,
        "step_selection": args.step_selection,
        "gradient_loss_interval": args.gradient_loss_interval,
    }
    loss_scales = (
        _compute_loss_scales(dataset, train_indices, args.gradient_target)
        if args.normalize_losses
        else DeepVBHLossScales()
    )
    compile_enabled = bool(args.compile)
    if compile_enabled:
        print(
            json.dumps(
                {
                    "warning": (
                        "torch.compile is disabled for DeepVBH training because the current "
                        "path relies on generalized eigenvalue solves and, when gradient "
                        "supervision is enabled, double backward. In practice this training "
                        "path hits unsupported torch.compile/aot_autograd or CUDA graph "
                        "capture failures."
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        compile_enabled = False
    split_summary["loss_scales"] = {
        "hamiltonian": loss_scales.hamiltonian,
        "energy": loss_scales.energy,
        "gradient": loss_scales.gradient,
    }
    split_summary["device"] = str(device)
    split_summary["dtype"] = args.dtype
    split_summary["compile"] = compile_enabled
    split_summary["cache_device_samples"] = args.cache_device_samples
    split_summary["config"] = None if args.config is None else str(args.config)
    split_summary["init_checkpoint"] = (
        None if args.init_checkpoint is None else str(args.init_checkpoint)
    )
    print(json.dumps(split_summary, ensure_ascii=False), flush=True)

    model = _build_model(dataset, args).to(device=device, dtype=dtype)
    _load_initial_model_state(model, args.init_checkpoint)
    compile_args = argparse.Namespace(**vars(args))
    compile_args.compile = compile_enabled
    model = _maybe_compile_model(model, compile_args)
    adamw_kwargs: dict[str, Any] = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if (
        args.fused_optimizer
        and device.type == "cuda"
        and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    ):
        adamw_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    weights = DeepVBHLossWeights(
        hamiltonian=args.hamiltonian_weight,
        energy=args.energy_weight,
        gradient_mse=args.gradient_mse_weight,
        gradient_direction=args.gradient_direction_weight,
    )

    best_checkpoint_path = args.best_checkpoint or _default_best_checkpoint(args.checkpoint)
    best_metric = float("inf")
    best_metric_name = "valid_objective_loss" if valid_indices else "train_objective_loss"
    best_saved = False
    sample_cache = (
        {} if args.cache_device_samples and device.type != "cpu" else None
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            dataset,
            train_indices,
            optimizer,
            device,
            weights,
            loss_scales,
            args.gradient_target,
            training=True,
            gradient_loss_interval=args.gradient_loss_interval,
            grad_clip_norm=args.grad_clip_norm,
            sample_cache=sample_cache,
        )
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "gradient_target": args.gradient_target,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        epoch_record.update(_prefixed_epoch_metrics("train", train_metrics))

        if valid_indices:
            valid_metrics = _run_epoch(
                model,
                dataset,
                valid_indices,
                optimizer=None,
                device=device,
                weights=weights,
                loss_scales=loss_scales,
                gradient_target=args.gradient_target,
                training=False,
                sample_cache=sample_cache,
            )
            epoch_record.update(_prefixed_epoch_metrics("valid", valid_metrics))

        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)

        current_metric = float(epoch_record[best_metric_name])
        if scheduler is not None:
            scheduler.step(current_metric)

        if current_metric < best_metric:
            best_metric = current_metric
            _save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                args,
                history,
                split_summary=split_summary,
            )
            best_saved = True

    _save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        args,
        history,
        split_summary=split_summary,
    )

    if test_indices:
        if best_saved:
            best_state = torch.load(
                best_checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(best_state["model_state_dict"])
        test_metrics = _run_epoch(
            model,
            dataset,
            test_indices,
            optimizer=None,
            device=device,
            weights=weights,
            loss_scales=loss_scales,
            gradient_target=args.gradient_target,
            training=False,
            sample_cache=sample_cache,
        )
        test_record = _prefixed_epoch_metrics("test", test_metrics)
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
