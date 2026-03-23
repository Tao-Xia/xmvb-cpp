from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import jax
import numpy as np
import torch

from .dataset import DeepVBHStepDataset


def _numeric_sample(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    numeric: dict[str, np.ndarray] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            numeric[key] = value.detach().cpu().numpy()
        elif isinstance(value, bool):
            numeric[key] = np.asarray(value, dtype=np.bool_)
        elif isinstance(value, int):
            numeric[key] = np.asarray(value, dtype=np.int64)
        elif isinstance(value, float):
            numeric[key] = np.asarray(value, dtype=np.float64)
    return numeric


def _sample_signature(sample: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple(
        (key, tuple(value.shape), value.dtype.str)
        for key, value in sorted(sample.items())
    )


def _layout_signature(sample: dict[str, np.ndarray]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    keys = (
        "shell_to_atom",
        "shell_angular_momenta",
        "shell_ao_starts",
        "shell_ao_counts",
    )
    return tuple(
        (
            key,
            tuple(int(value) for value in sample[key].reshape(-1).tolist()),
        )
        for key in keys
    )


def _static_layout(sample: dict[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    return {
        "shell_to_atom": tuple(int(value) for value in sample["shell_to_atom"].reshape(-1).tolist()),
        "shell_angular_momenta": tuple(
            int(value) for value in sample["shell_angular_momenta"].reshape(-1).tolist()
        ),
        "shell_ao_starts": tuple(int(value) for value in sample["shell_ao_starts"].reshape(-1).tolist()),
        "shell_ao_counts": tuple(int(value) for value in sample["shell_ao_counts"].reshape(-1).tolist()),
    }


def _zero_like_tree(tree: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.zeros_like(value)
        for key, value in tree.items()
    }


def _stack_tree(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = samples[0].keys()
    return {
        key: np.stack([sample[key] for sample in samples], axis=0)
        for key in keys
    }


def _pad_batch(
    batch: dict[str, np.ndarray],
    zero_sample: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, np.ndarray]:
    current_size = next(iter(batch.values())).shape[0]
    if current_size == batch_size:
        result = dict(batch)
        result["batch_mask"] = np.ones((batch_size,), dtype=np.float32)
        return result

    pad_count = batch_size - current_size
    padding = _stack_tree([zero_sample] * pad_count)
    result = {
        key: np.concatenate([value, padding[key]], axis=0)
        for key, value in batch.items()
    }
    result["batch_mask"] = np.concatenate(
        [
            np.ones((current_size,), dtype=np.float32),
            np.zeros((pad_count,), dtype=np.float32),
        ],
        axis=0,
    )
    return result


def _attach_target_gradient(
    batch: dict[str, np.ndarray],
    gradient_target: str,
) -> dict[str, np.ndarray]:
    gradient_target_map = {
        "total": "sparse_orbital_energy_gradient",
        "reference": "sparse_orbital_reference_energy_gradient",
        "residual": "sparse_orbital_residual_energy_gradient",
    }
    target_key = gradient_target_map[gradient_target]
    result = dict(batch)
    result["target_gradient"] = result[target_key]
    return result


def _attach_target_orbital_residual(
    batch: dict[str, np.ndarray],
    orbital_residual_target: str,
    orbital_update_space: str,
    rollout_target: str,
) -> dict[str, np.ndarray]:
    if orbital_update_space == "dense_shell":
        input_key = "dense_orbital_coefficients"
        mask_key = "orbital_shell_dense_mask"
        target_map = {
            "final": "final_dense_orbital_residual",
            "next": "next_dense_orbital_delta",
        }
        rollout_key = "rollout_dense_orbital_deltas"
    else:
        input_key = "orbital_value_table"
        mask_key = "orbital_basis_mask"
        target_map = {
            "final": "final_orbital_residual",
            "next": "next_orbital_delta",
        }
        rollout_key = "rollout_orbital_deltas"

    final_target_key = target_map["final"]
    batch_size = int(batch[input_key].shape[0])
    orbital_shape = tuple(int(size) for size in batch[input_key].shape[1:])
    if orbital_residual_target == "none":
        result = dict(batch)
        result["target_orbital_residual"] = np.zeros_like(result[input_key])
    else:
        target_key = target_map[orbital_residual_target]
        result = dict(batch)
        result["target_orbital_residual"] = result[target_key]

    result["input_orbital_state"] = result[input_key]
    result["orbital_update_mask"] = result[mask_key]
    if rollout_target == "trajectory" and rollout_key in result:
        result["target_rollout_orbital_deltas"] = result[rollout_key]
    elif rollout_target == "final":
        if rollout_key in result:
            rollout_horizon = int(result[rollout_key].shape[1])
        else:
            rollout_horizon = 0
        repeated_final_delta = np.broadcast_to(
            result[final_target_key][:, None, ...],
            (batch_size, rollout_horizon, *result[final_target_key].shape[1:]),
        ).copy()
        result["target_rollout_orbital_deltas"] = repeated_final_delta
    else:
        result["target_rollout_orbital_deltas"] = np.zeros(
            (batch_size, 0, *orbital_shape),
            dtype=result[input_key].dtype,
        )
    if "rollout_step_mask" not in result:
        result["rollout_step_mask"] = np.zeros((batch_size, 0), dtype=np.bool_)
        return result
    return result


@dataclass(frozen=True)
class JAXBucket:
    signature: tuple[Any, ...]
    global_indices: np.ndarray
    sample_names: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    zero_sample: dict[str, np.ndarray]
    layout: dict[str, tuple[int, ...]]


class JAXStepDataset:
    def __init__(
        self,
        dataset_root: str,
        *,
        dtype: torch.dtype = torch.float32,
        include_samples: set[str] | None = None,
        exclude_samples: set[str] | None = None,
        cache_steps: bool = False,
        preload_steps: bool = False,
        rollout_horizon: int = 0,
    ) -> None:
        self.torch_dataset = DeepVBHStepDataset(
            dataset_root,
            dtype=dtype,
            include_samples=include_samples,
            exclude_samples=exclude_samples,
            cache_steps=cache_steps,
            preload_steps=preload_steps,
            rollout_horizon=rollout_horizon,
        )
        self._numeric_samples: list[dict[str, np.ndarray] | None] = [None] * len(self.torch_dataset)
        self._sample_names = [
            self.torch_dataset.sample_name_at(index)
            for index in range(len(self.torch_dataset))
        ]
        self._accepted_iteration_indices = [
            self.torch_dataset.accepted_iteration_index_at(index)
            for index in range(len(self.torch_dataset))
        ]
        self.buckets = self._build_buckets()
        self._bucket_lookup = self._build_bucket_lookup()

    def __len__(self) -> int:
        return len(self.torch_dataset)

    @property
    def sample_names(self) -> list[str]:
        return self.torch_dataset.sample_names

    def sample_name_at(self, index: int) -> str:
        return self._sample_names[index]

    def accepted_iteration_index_at(self, index: int) -> int:
        return self._accepted_iteration_indices[index]

    def indices_by_sample(self) -> dict[str, list[int]]:
        return self.torch_dataset.indices_by_sample()

    def example_sample(self, index: int) -> dict[str, np.ndarray]:
        return self.numeric_sample(index)

    def bucket_signature_at(self, index: int) -> tuple[Any, ...]:
        return self._bucket_lookup[index][0]

    def bucket_layout(self, signature: tuple[Any, ...]) -> dict[str, tuple[int, ...]]:
        return self.buckets[signature].layout

    def max_atomic_number(self) -> int:
        return self.torch_dataset.max_atomic_number()

    def max_angular_momentum(self) -> int:
        return self.torch_dataset.max_angular_momentum()

    def max_ao_local_index(self) -> int:
        return self.torch_dataset.max_ao_local_index()

    def numeric_sample(self, index: int) -> dict[str, np.ndarray]:
        cached = self._numeric_samples[index]
        if cached is not None:
            return cached
        sample = _numeric_sample(self.torch_dataset[index])
        self._numeric_samples[index] = sample
        return sample

    def iter_batches(
        self,
        indices: list[int],
        *,
        batch_size: int,
        gradient_target: str,
        orbital_residual_target: str,
        orbital_update_space: str,
        rollout_target: str,
        shuffle: bool,
        seed: int,
        device: jax.Device | None = None,
    ) -> Iterator[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        batches: list[dict[str, Any]] = []
        positions_by_bucket = self._positions_by_bucket(indices)

        for signature, positions in positions_by_bucket.items():
            bucket = self.buckets[signature]
            ordered_positions = positions.copy()
            if shuffle:
                rng.shuffle(ordered_positions)
            for start in range(0, ordered_positions.size, batch_size):
                batch_positions = ordered_positions[start : start + batch_size]
                host_batch = {
                    key: value[batch_positions]
                    for key, value in bucket.arrays.items()
                }
                host_batch = _pad_batch(host_batch, bucket.zero_sample, batch_size)
                host_batch = _attach_target_gradient(host_batch, gradient_target)
                host_batch = _attach_target_orbital_residual(
                    host_batch,
                    orbital_residual_target,
                    orbital_update_space,
                    rollout_target,
                )
                if device is not None:
                    device_batch = jax.device_put(host_batch, device=device)
                else:
                    device_batch = jax.device_put(host_batch)
                batches.append(
                    {
                        "bucket_signature": signature,
                        "host_batch_size": int(batch_positions.size),
                        "batch": device_batch,
                    }
                )

        if shuffle:
            rng.shuffle(batches)
        yield from batches

    def compute_loss_scales(
        self,
        indices: list[int],
        gradient_target: str,
        orbital_residual_target: str,
        orbital_update_space: str,
        rollout_target: str,
        *,
        epsilon: float = 1.0e-8,
    ) -> dict[str, float]:
        if not indices:
            return {
                "hamiltonian": 1.0,
                "energy": 1.0,
                "gradient": 1.0,
                "orbital_residual": 1.0,
                "rollout_orbital": 1.0,
            }

        gradient_target_map = {
            "total": "sparse_orbital_energy_gradient",
            "reference": "sparse_orbital_reference_energy_gradient",
            "residual": "sparse_orbital_residual_energy_gradient",
        }
        target_key = gradient_target_map[gradient_target]
        if orbital_update_space == "dense_shell":
            orbital_mask_key = "orbital_shell_dense_mask"
            orbital_residual_target_map = {
                "final": "final_dense_orbital_residual",
                "next": "next_dense_orbital_delta",
            }
            rollout_target_key = "rollout_dense_orbital_deltas"
        else:
            orbital_mask_key = "orbital_basis_mask"
            orbital_residual_target_map = {
                "final": "final_orbital_residual",
                "next": "next_orbital_delta",
            }
            rollout_target_key = "rollout_orbital_deltas"
        orbital_target_key = orbital_residual_target_map.get(orbital_residual_target)

        h_terms: list[np.ndarray] = []
        energy_terms: list[float] = []
        gradient_terms: list[np.ndarray] = []
        orbital_residual_terms: list[np.ndarray] = []
        rollout_terms: list[np.ndarray] = []
        for index in indices:
            sample = self.numeric_sample(index)
            hamiltonian = sample["two_electron_hamiltonian_matrix"]
            row, column = np.tril_indices(hamiltonian.shape[0], k=0)
            h_terms.append(hamiltonian[row, column].reshape(-1))
            energy_terms.append(float(sample["target_total_energy"]))
            differentiable_indices = sample["differentiable_parameter_indices"].reshape(-1)
            gradient = sample[target_key].reshape(-1)[differentiable_indices]
            gradient_terms.append(gradient)
            orbital_mask = sample[orbital_mask_key].astype(np.bool_)
            if orbital_target_key is not None:
                orbital_residual = sample[orbital_target_key][orbital_mask]
                orbital_residual_terms.append(orbital_residual)
            rollout_deltas = sample.get(rollout_target_key)
            rollout_step_mask = sample.get("rollout_step_mask")
            if (
                rollout_target == "trajectory"
                and rollout_deltas is not None
                and rollout_step_mask is not None
            ):
                valid_rollout = rollout_deltas[rollout_step_mask.astype(np.bool_)]
                if valid_rollout.size > 0:
                    rollout_terms.append(valid_rollout[:, orbital_mask].reshape(-1))
            elif rollout_target == "final" and rollout_step_mask is not None:
                valid_rollout_count = int(np.sum(rollout_step_mask.astype(np.bool_)))
                if valid_rollout_count > 0:
                    final_residual = sample[orbital_residual_target_map["final"]][orbital_mask]
                    rollout_terms.append(final_residual.reshape(-1))

        h_scale = float(np.sqrt(np.mean(np.square(np.concatenate(h_terms, axis=0)))))
        energy_array = np.asarray(energy_terms, dtype=np.float64)
        if energy_array.size > 1:
            energy_scale = float(np.std(energy_array))
        else:
            energy_scale = float(abs(energy_array[0]))
        gradient_scale = float(
            np.sqrt(np.mean(np.square(np.concatenate(gradient_terms, axis=0))))
        )
        if orbital_residual_terms:
            orbital_residual_scale = float(
                np.sqrt(
                    np.mean(
                        np.square(np.concatenate(orbital_residual_terms, axis=0))
                    )
                )
            )
        else:
            orbital_residual_scale = 1.0
        if rollout_terms:
            rollout_orbital_scale = float(
                np.sqrt(
                    np.mean(
                        np.square(np.concatenate(rollout_terms, axis=0))
                    )
                )
            )
        else:
            rollout_orbital_scale = 1.0
        return {
            "hamiltonian": max(h_scale, epsilon),
            "energy": max(energy_scale, epsilon),
            "gradient": max(gradient_scale, epsilon),
            "orbital_residual": max(orbital_residual_scale, epsilon),
            "rollout_orbital": max(rollout_orbital_scale, epsilon),
        }

    def _build_buckets(self) -> dict[tuple[tuple[str, tuple[int, ...], str], ...], JAXBucket]:
        grouped: dict[tuple[Any, ...], list[tuple[int, str, dict[str, np.ndarray]]]] = {}
        for index in range(len(self)):
            sample = self.numeric_sample(index)
            signature = (_sample_signature(sample), _layout_signature(sample))
            grouped.setdefault(signature, []).append(
                (index, self.sample_name_at(index), sample)
            )

        buckets: dict[tuple[Any, ...], JAXBucket] = {}
        for signature, entries in grouped.items():
            global_indices = np.asarray([entry[0] for entry in entries], dtype=np.int64)
            sample_names = tuple(entry[1] for entry in entries)
            stacked = _stack_tree([entry[2] for entry in entries])
            zero_sample = _zero_like_tree(entries[0][2])
            buckets[signature] = JAXBucket(
                signature=signature,
                global_indices=global_indices,
                sample_names=sample_names,
                arrays=stacked,
                zero_sample=zero_sample,
                layout=_static_layout(entries[0][2]),
            )
        return buckets

    def _build_bucket_lookup(
        self,
    ) -> dict[int, tuple[tuple[Any, ...], int]]:
        lookup: dict[int, tuple[tuple[Any, ...], int]] = {}
        for signature, bucket in self.buckets.items():
            for position, index in enumerate(bucket.global_indices.tolist()):
                lookup[int(index)] = (signature, position)
        return lookup

    def _positions_by_bucket(
        self,
        indices: list[int],
    ) -> dict[tuple[Any, ...], np.ndarray]:
        grouped: dict[tuple[Any, ...], list[int]] = {}
        for index in indices:
            signature, position = self._bucket_lookup[index]
            grouped.setdefault(signature, []).append(position)
        return {
            signature: np.asarray(positions, dtype=np.int64)
            for signature, positions in grouped.items()
        }
