from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_array(path: Path, dtype: np.dtype[Any], shape: tuple[int, ...]) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    expected_size = int(np.prod(shape))
    if array.size != expected_size:
        raise ValueError(
            f"unexpected size for {path}: got {array.size}, expected {expected_size}"
        )
    return array.reshape(shape)


def _read_vector(path: Path, dtype: np.dtype[Any], length: int) -> np.ndarray:
    array = np.fromfile(path, dtype=dtype)
    if array.size != length:
        raise ValueError(
            f"unexpected size for {path}: got {array.size}, expected {length}"
        )
    return array


def _column_major_matrix(path: Path, size: int) -> np.ndarray:
    flat = _read_vector(path, np.float64, size * size)
    matrix = flat.reshape((size, size), order="F")
    return 0.5 * (matrix + matrix.T)


def _read_optional_array(
    path: Path,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if not path.exists():
        return None
    return _read_array(path, dtype, shape)


def _structure_occupancy(
    raw_structure_orbitals: np.ndarray,
    n_orbitals: int,
) -> np.ndarray:
    occupancy = np.zeros((raw_structure_orbitals.shape[0], n_orbitals), dtype=np.float64)
    row_index = np.repeat(
        np.arange(raw_structure_orbitals.shape[0], dtype=np.int64),
        raw_structure_orbitals.shape[1],
    )
    orbital_index = raw_structure_orbitals.reshape(-1) - 1
    np.add.at(occupancy, (row_index, orbital_index), 1.0)
    return occupancy


def _orbital_basis_mask(
    orbital_basis_counts: np.ndarray,
    n_basis_functions: int,
) -> np.ndarray:
    basis_index = np.arange(n_basis_functions, dtype=np.int64)[None, :]
    return basis_index < orbital_basis_counts[:, None]


def _orbital_shell_dense_mask(
    orbital_basis_index_table: np.ndarray,
    orbital_basis_mask: np.ndarray,
    ao_to_shell: np.ndarray,
    shell_ao_starts: np.ndarray,
    shell_ao_counts: np.ndarray,
) -> np.ndarray:
    n_orbitals, n_basis_functions = orbital_basis_index_table.shape
    dense_mask = np.zeros((n_orbitals, n_basis_functions), dtype=np.bool_)
    for orbital_index in range(n_orbitals):
        active_sparse_indices = orbital_basis_index_table[orbital_index][orbital_basis_mask[orbital_index]]
        if active_sparse_indices.size == 0:
            continue
        active_ao_indices = active_sparse_indices.astype(np.int64, copy=False) - 1
        touched_shells = np.unique(ao_to_shell[active_ao_indices])
        for shell_index in touched_shells.tolist():
            shell_start = int(shell_ao_starts[shell_index])
            shell_count = int(shell_ao_counts[shell_index])
            dense_mask[orbital_index, shell_start : shell_start + shell_count] = True
    return dense_mask


def _sparse_to_dense_ao_coefficients(
    sparse_coefficients: torch.Tensor,
    orbital_basis_index_table: torch.Tensor,
    orbital_basis_mask: torch.Tensor,
) -> torch.Tensor:
    n_orbitals, n_basis_functions = sparse_coefficients.shape
    ao_indices = orbital_basis_index_table - 1
    safe_ao_indices = torch.clamp(ao_indices, min=0)
    orbital_indices = torch.arange(
        n_orbitals,
        dtype=torch.int64,
        device=sparse_coefficients.device,
    )[:, None]
    flat_indices = orbital_indices * n_basis_functions + safe_ao_indices
    flat_values = torch.where(
        orbital_basis_mask,
        sparse_coefficients,
        torch.zeros_like(sparse_coefficients),
    ).reshape(-1)
    dense_flat = torch.zeros(
        (n_orbitals * n_basis_functions,),
        dtype=sparse_coefficients.dtype,
        device=sparse_coefficients.device,
    )
    dense_flat = dense_flat.scatter_add(0, flat_indices.reshape(-1), flat_values)
    return dense_flat.reshape(n_orbitals, n_basis_functions)


def _structure_pair_topology(
    raw_structure_orbitals: np.ndarray,
    *,
    n_total_electrons: int,
    n_active_electrons: int,
    spin_multiplicity: int,
) -> dict[str, np.ndarray]:
    n_structures = raw_structure_orbitals.shape[0]
    n_inactive_doubly_occupied_orbitals = (n_total_electrons - n_active_electrons) // 2
    n_open_shell_electrons = spin_multiplicity - 1
    n_active_beta_electrons = (n_active_electrons - n_open_shell_electrons) // 2
    active_start = 2 * n_inactive_doubly_occupied_orbitals
    active_stop = active_start + n_active_electrons
    active_structure_orbitals = raw_structure_orbitals[:, active_start:active_stop] - 1

    # The current runtime exports raw VB structures in legacy electron-slot order.
    # We interpret the paired active part as consecutive spin-paired slots.
    if n_active_beta_electrons > 0:
        paired_orbitals = active_structure_orbitals[:, : 2 * n_active_beta_electrons]
        structure_pair_orbital_indices = paired_orbitals.reshape(n_structures, n_active_beta_electrons, 2)
        structure_pair_mask = np.ones((n_structures, n_active_beta_electrons), dtype=np.bool_)
    else:
        structure_pair_orbital_indices = np.zeros((n_structures, 0, 2), dtype=np.int32)
        structure_pair_mask = np.zeros((n_structures, 0), dtype=np.bool_)

    structure_open_shell_orbitals = active_structure_orbitals[:, 2 * n_active_beta_electrons :]
    if n_open_shell_electrons > 0:
        structure_open_shell_mask = np.ones(
            (n_structures, n_open_shell_electrons),
            dtype=np.bool_,
        )
    else:
        structure_open_shell_mask = np.zeros((n_structures, 0), dtype=np.bool_)

    return {
        "active_structure_orbitals": active_structure_orbitals.astype(np.int32, copy=False),
        "structure_pair_orbital_indices": structure_pair_orbital_indices.astype(np.int32, copy=False),
        "structure_pair_mask": structure_pair_mask,
        "structure_open_shell_orbitals": structure_open_shell_orbitals.astype(np.int32, copy=False),
        "structure_open_shell_mask": structure_open_shell_mask,
    }


def _to_tensor_map(data: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"i", "u"}:
                converted[key] = torch.from_numpy(value.astype(np.int64, copy=False))
            elif value.dtype.kind == "b":
                converted[key] = torch.from_numpy(value.astype(np.bool_, copy=False))
            else:
                converted[key] = torch.from_numpy(value.astype(np.float64, copy=False)).to(dtype)
        elif isinstance(value, float):
            converted[key] = torch.tensor(value, dtype=dtype)
        elif isinstance(value, int):
            converted[key] = torch.tensor(value, dtype=torch.int64)
        else:
            converted[key] = value
    return converted


@dataclass(frozen=True)
class _StaticSampleCache:
    data: dict[str, Any]


@dataclass(frozen=True)
class _DatasetEntry:
    sample_dir: Path
    step_dir: Path
    sample_metadata: dict[str, Any]
    step_metadata: dict[str, Any]


def _align_orbital_gauge(
    reference_orbital_value_table: torch.Tensor,
    target_orbital_value_table: torch.Tensor,
    orbital_basis_mask: torch.Tensor,
) -> torch.Tensor:
    masked_reference = torch.where(
        orbital_basis_mask,
        reference_orbital_value_table,
        torch.zeros_like(reference_orbital_value_table),
    )
    masked_target = torch.where(
        orbital_basis_mask,
        target_orbital_value_table,
        torch.zeros_like(target_orbital_value_table),
    )
    overlaps = torch.sum(masked_reference * masked_target, dim=-1, keepdim=True)
    signs = torch.where(
        overlaps < 0.0,
        -torch.ones_like(overlaps),
        torch.ones_like(overlaps),
    )
    return target_orbital_value_table * signs


class DeepVBHStepDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_root: str | Path,
        *,
        dtype: torch.dtype = torch.float64,
        include_samples: set[str] | None = None,
        exclude_samples: set[str] | None = None,
        cache_steps: bool = False,
        preload_steps: bool = False,
        rollout_horizon: int = 0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.dtype = dtype
        self.cache_steps = cache_steps or preload_steps
        self.rollout_horizon = max(int(rollout_horizon), 0)
        self._entries: list[_DatasetEntry] = []
        self._static_cache: dict[Path, _StaticSampleCache] = {}
        self._step_cache: dict[Path, dict[str, Any]] = {}
        self._next_index_by_entry_index: list[int | None] = []
        self._final_index_by_entry_index: list[int] = []

        sample_dirs = sorted(
            path
            for path in self.dataset_root.iterdir()
            if path.is_dir() and path.joinpath("metadata.json").exists()
        )
        for sample_dir in sample_dirs:
            sample_metadata = _read_json(sample_dir / "metadata.json")
            sample_name = str(sample_metadata["sample_name"])
            if include_samples is not None and sample_name not in include_samples:
                continue
            if exclude_samples is not None and sample_name in exclude_samples:
                continue

            step_dirs = sorted(
                path
                for path in sample_dir.joinpath("steps").iterdir()
                if path.is_dir() and path.joinpath("metadata.json").exists()
            )
            for step_dir in step_dirs:
                self._entries.append(
                    _DatasetEntry(
                        sample_dir=sample_dir,
                        step_dir=step_dir,
                        sample_metadata=sample_metadata,
                        step_metadata=_read_json(step_dir / "metadata.json"),
                    )
                )

        if not self._entries:
            raise ValueError(f"no DeepVBH samples found under {self.dataset_root}")

        self._build_step_links()

        if preload_steps:
            for entry in self._entries:
                self._load_step(
                    entry.step_dir,
                    entry.sample_metadata,
                    entry.step_metadata,
                )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self._entries[index]
        static_data = self._load_static(entry.sample_dir, entry.sample_metadata)
        step_data = self._load_step(entry.step_dir, entry.sample_metadata, entry.step_metadata)
        result = dict(static_data)
        result.update(step_data)
        orbital_basis_mask = result["orbital_basis_mask"]
        orbital_shell_dense_mask = result["orbital_shell_dense_mask"]
        for key in (
            "orbital_value_table",
            "sparse_orbital_energy_gradient",
            "sparse_orbital_reference_energy_gradient",
            "sparse_orbital_residual_energy_gradient",
        ):
            result[key] = torch.where(
                orbital_basis_mask,
                result[key],
                torch.zeros_like(result[key]),
            )
        dense_orbital_coefficients = _sparse_to_dense_ao_coefficients(
            result["orbital_value_table"],
            result["orbital_basis_index_table"],
            orbital_basis_mask,
        )
        result["dense_orbital_coefficients"] = torch.where(
            orbital_shell_dense_mask,
            dense_orbital_coefficients,
            torch.zeros_like(dense_orbital_coefficients),
        )
        final_step_data = self._load_linked_step(self._final_index_by_entry_index[index])
        aligned_final_orbital_value_table = _align_orbital_gauge(
            result["orbital_value_table"],
            final_step_data["orbital_value_table"],
            orbital_basis_mask,
        )
        result["aligned_final_orbital_value_table"] = torch.where(
            orbital_basis_mask,
            aligned_final_orbital_value_table,
            torch.zeros_like(aligned_final_orbital_value_table),
        )
        result["final_orbital_residual"] = torch.where(
            orbital_basis_mask,
            result["aligned_final_orbital_value_table"] - result["orbital_value_table"],
            torch.zeros_like(result["orbital_value_table"]),
        )
        final_dense_orbital_coefficients = _sparse_to_dense_ao_coefficients(
            final_step_data["orbital_value_table"],
            result["orbital_basis_index_table"],
            orbital_basis_mask,
        )
        aligned_final_dense_orbital_coefficients = _align_orbital_gauge(
            result["dense_orbital_coefficients"],
            final_dense_orbital_coefficients,
            orbital_shell_dense_mask,
        )
        result["aligned_final_dense_orbital_coefficients"] = torch.where(
            orbital_shell_dense_mask,
            aligned_final_dense_orbital_coefficients,
            torch.zeros_like(aligned_final_dense_orbital_coefficients),
        )
        result["final_dense_orbital_residual"] = torch.where(
            orbital_shell_dense_mask,
            result["aligned_final_dense_orbital_coefficients"]
            - result["dense_orbital_coefficients"],
            torch.zeros_like(result["dense_orbital_coefficients"]),
        )

        next_index = self._next_index_by_entry_index[index]
        if next_index is None:
            aligned_next_orbital_value_table = result["orbital_value_table"]
            next_orbital_delta = torch.zeros_like(result["orbital_value_table"])
            aligned_next_dense_orbital_coefficients = result["dense_orbital_coefficients"]
            next_dense_orbital_delta = torch.zeros_like(result["dense_orbital_coefficients"])
            result["has_next_step"] = torch.tensor(False, dtype=torch.bool)
        else:
            next_step_data = self._load_linked_step(next_index)
            aligned_next_orbital_value_table = _align_orbital_gauge(
                result["orbital_value_table"],
                next_step_data["orbital_value_table"],
                orbital_basis_mask,
            )
            next_orbital_delta = aligned_next_orbital_value_table - result["orbital_value_table"]
            next_dense_orbital_coefficients = _sparse_to_dense_ao_coefficients(
                next_step_data["orbital_value_table"],
                result["orbital_basis_index_table"],
                orbital_basis_mask,
            )
            aligned_next_dense_orbital_coefficients = _align_orbital_gauge(
                result["dense_orbital_coefficients"],
                next_dense_orbital_coefficients,
                orbital_shell_dense_mask,
            )
            next_dense_orbital_delta = (
                aligned_next_dense_orbital_coefficients - result["dense_orbital_coefficients"]
            )
            result["has_next_step"] = torch.tensor(True, dtype=torch.bool)
        result["aligned_next_orbital_value_table"] = torch.where(
            orbital_basis_mask,
            aligned_next_orbital_value_table,
            torch.zeros_like(aligned_next_orbital_value_table),
        )
        result["next_orbital_delta"] = torch.where(
            orbital_basis_mask,
            next_orbital_delta,
            torch.zeros_like(next_orbital_delta),
        )
        result["aligned_next_dense_orbital_coefficients"] = torch.where(
            orbital_shell_dense_mask,
            aligned_next_dense_orbital_coefficients,
            torch.zeros_like(aligned_next_dense_orbital_coefficients),
        )
        result["next_dense_orbital_delta"] = torch.where(
            orbital_shell_dense_mask,
            next_dense_orbital_delta,
            torch.zeros_like(next_dense_orbital_delta),
        )
        if self.rollout_horizon > 0:
            rollout_tables: list[torch.Tensor] = []
            rollout_deltas: list[torch.Tensor] = []
            dense_rollout_tables: list[torch.Tensor] = []
            dense_rollout_deltas: list[torch.Tensor] = []
            rollout_mask: list[bool] = []
            future_index = next_index
            for _ in range(self.rollout_horizon):
                if future_index is None:
                    aligned_future_orbital_value_table = torch.zeros_like(result["orbital_value_table"])
                    future_orbital_delta = torch.zeros_like(result["orbital_value_table"])
                    aligned_future_dense_orbital_coefficients = torch.zeros_like(
                        result["dense_orbital_coefficients"]
                    )
                    future_dense_orbital_delta = torch.zeros_like(
                        result["dense_orbital_coefficients"]
                    )
                    rollout_mask.append(False)
                else:
                    future_step_data = self._load_linked_step(future_index)
                    aligned_future_orbital_value_table = _align_orbital_gauge(
                        result["orbital_value_table"],
                        future_step_data["orbital_value_table"],
                        orbital_basis_mask,
                    )
                    future_orbital_delta = (
                        aligned_future_orbital_value_table - result["orbital_value_table"]
                    )
                    future_dense_orbital_coefficients = _sparse_to_dense_ao_coefficients(
                        future_step_data["orbital_value_table"],
                        result["orbital_basis_index_table"],
                        orbital_basis_mask,
                    )
                    aligned_future_dense_orbital_coefficients = _align_orbital_gauge(
                        result["dense_orbital_coefficients"],
                        future_dense_orbital_coefficients,
                        orbital_shell_dense_mask,
                    )
                    future_dense_orbital_delta = (
                        aligned_future_dense_orbital_coefficients
                        - result["dense_orbital_coefficients"]
                    )
                    rollout_mask.append(True)
                    future_index = self._next_index_by_entry_index[future_index]
                rollout_tables.append(
                    torch.where(
                        orbital_basis_mask,
                        aligned_future_orbital_value_table,
                        torch.zeros_like(aligned_future_orbital_value_table),
                    )
                )
                rollout_deltas.append(
                    torch.where(
                        orbital_basis_mask,
                        future_orbital_delta,
                        torch.zeros_like(future_orbital_delta),
                    )
                )
                dense_rollout_tables.append(
                    torch.where(
                        orbital_shell_dense_mask,
                        aligned_future_dense_orbital_coefficients,
                        torch.zeros_like(aligned_future_dense_orbital_coefficients),
                    )
                )
                dense_rollout_deltas.append(
                    torch.where(
                        orbital_shell_dense_mask,
                        future_dense_orbital_delta,
                        torch.zeros_like(future_dense_orbital_delta),
                    )
                )
            result["rollout_aligned_orbital_value_tables"] = torch.stack(rollout_tables, dim=0)
            result["rollout_orbital_deltas"] = torch.stack(rollout_deltas, dim=0)
            result["rollout_aligned_dense_orbital_coefficients"] = torch.stack(
                dense_rollout_tables,
                dim=0,
            )
            result["rollout_dense_orbital_deltas"] = torch.stack(
                dense_rollout_deltas,
                dim=0,
            )
            result["rollout_step_mask"] = torch.tensor(rollout_mask, dtype=torch.bool)
        return result

    @property
    def sample_names(self) -> list[str]:
        return sorted(
            {
                str(entry.sample_metadata["sample_name"])
                for entry in self._entries
            }
        )

    def sample_name_at(self, index: int) -> str:
        return str(self._entries[index].sample_metadata["sample_name"])

    def accepted_iteration_index_at(self, index: int) -> int:
        return int(self._entries[index].step_metadata["accepted_iteration_index"])

    def indices_by_sample(self) -> dict[str, list[int]]:
        grouped: dict[str, list[int]] = {}
        for index, entry in enumerate(self._entries):
            sample_name = str(entry.sample_metadata["sample_name"])
            grouped.setdefault(sample_name, []).append(index)
        for sample_name, indices in grouped.items():
            indices.sort(key=self.accepted_iteration_index_at)
            grouped[sample_name] = indices
        return grouped

    def max_atomic_number(self) -> int:
        maximum = 0
        for cache in self._iter_static_cache():
            maximum = max(
                maximum,
                int(torch.max(cache.data["atomic_numbers"]).item()),
            )
        return maximum

    def max_angular_momentum(self) -> int:
        maximum = 0
        for cache in self._iter_static_cache():
            maximum = max(
                maximum,
                int(torch.max(cache.data["ao_angular_momenta"]).item()),
            )
        return maximum

    def max_ao_local_index(self) -> int:
        maximum = 0
        for cache in self._iter_static_cache():
            maximum = max(
                maximum,
                int(torch.max(cache.data["ao_shell_local_indices"]).item()),
            )
        return maximum

    def _iter_static_cache(self) -> list[_StaticSampleCache]:
        for entry in self._entries:
            self._load_static(entry.sample_dir, entry.sample_metadata)
        return list(self._static_cache.values())

    def _build_step_links(self) -> None:
        next_index_by_entry_index: list[int | None] = [None] * len(self._entries)
        final_index_by_entry_index: list[int] = [0] * len(self._entries)
        indices_by_sample_dir: dict[Path, list[int]] = {}
        for index, entry in enumerate(self._entries):
            indices_by_sample_dir.setdefault(entry.sample_dir, []).append(index)

        for entry_indices in indices_by_sample_dir.values():
            entry_indices.sort(
                key=lambda entry_index: int(
                    self._entries[entry_index].step_metadata["accepted_iteration_index"]
                )
            )
            final_index = entry_indices[-1]
            for position, entry_index in enumerate(entry_indices):
                final_index_by_entry_index[entry_index] = final_index
                if position + 1 < len(entry_indices):
                    next_index_by_entry_index[entry_index] = entry_indices[position + 1]

        self._next_index_by_entry_index = next_index_by_entry_index
        self._final_index_by_entry_index = final_index_by_entry_index

    def _load_linked_step(self, entry_index: int) -> dict[str, Any]:
        entry = self._entries[entry_index]
        return self._load_step(
            entry.step_dir,
            entry.sample_metadata,
            entry.step_metadata,
        )

    def _load_static(
        self,
        sample_dir: Path,
        sample_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        cache = self._static_cache.get(sample_dir)
        if cache is not None:
            return cache.data

        static_dir = sample_dir / "static"
        static_metadata = _read_json(static_dir / "metadata.json")

        n_atoms = int(sample_metadata["n_atoms"])
        n_shells = int(sample_metadata["n_shells"])
        n_basis_functions = int(sample_metadata["n_basis_functions"])
        n_orbitals = int(sample_metadata["n_orbitals"])
        n_structures = int(sample_metadata["n_structures"])
        n_total_electrons = int(sample_metadata["n_total_electrons"])
        n_differentiable_parameters = int(sample_metadata["n_differentiable_parameters"])
        spin_multiplicity = int(sample_metadata["spin_multiplicity"])
        n_open_shell_electrons = spin_multiplicity - 1
        n_active_beta_electrons = (
            int(sample_metadata["n_active_electrons"]) - n_open_shell_electrons
        ) // 2

        raw_structure_orbitals = _read_array(
            static_dir / "raw_structure_orbitals_i32.bin",
            np.int32,
            (n_structures, n_total_electrons),
        )
        structure_pair_topology = _structure_pair_topology(
            raw_structure_orbitals,
            n_total_electrons=n_total_electrons,
            n_active_electrons=int(sample_metadata["n_active_electrons"]),
            spin_multiplicity=spin_multiplicity,
        )
        static_np: dict[str, Any] = {
            "sample_name": str(sample_metadata["sample_name"]),
            "sample_dir": str(sample_dir),
            "format_version": int(sample_metadata["format_version"]),
            "n_structures": n_structures,
            "n_total_electrons": n_total_electrons,
            "n_active_electrons": int(sample_metadata["n_active_electrons"]),
            "spin_multiplicity": spin_multiplicity,
            "n_open_shell_electrons": n_open_shell_electrons,
            "n_active_beta_electrons": n_active_beta_electrons,
            "n_atoms": n_atoms,
            "n_shells": n_shells,
            "n_basis_functions": n_basis_functions,
            "n_orbitals": n_orbitals,
            "n_active_orbitals": int(sample_metadata["n_active_orbitals"]),
            "n_differentiable_parameters": n_differentiable_parameters,
            "nuclear_repulsion_energy": float(sample_metadata["nuclear_repulsion_energy"]),
            "raw_structure_orbitals": raw_structure_orbitals.astype(np.int64, copy=False),
            "structure_occupancy": _structure_occupancy(raw_structure_orbitals, n_orbitals),
            "orbital_basis_index_table": _read_array(
                static_dir / "orbital_basis_index_table_i32.bin",
                np.int32,
                (n_orbitals, n_basis_functions),
            ),
            "orbital_basis_counts": _read_vector(
                static_dir / "orbital_basis_counts_i32.bin",
                np.int32,
                n_orbitals,
            ),
            "original_orbital_basis_counts": _read_vector(
                static_dir / "original_orbital_basis_counts_i32.bin",
                np.int32,
                n_orbitals,
            ),
            "differentiable_parameter_indices": _read_vector(
                static_dir / "differentiable_parameter_indices_i32.bin",
                np.int32,
                n_differentiable_parameters,
            ),
            "atomic_numbers": _read_vector(
                static_dir / "atomic_numbers_i32.bin",
                np.int32,
                n_atoms,
            ),
            "atomic_coordinates": _read_array(
                static_dir / "atomic_coordinates_f64.bin",
                np.float64,
                (n_atoms, 3),
            ),
            "shell_to_atom": _read_vector(
                static_dir / "shell_to_atom_i32.bin",
                np.int32,
                n_shells,
            ),
            "shell_angular_momenta": _read_vector(
                static_dir / "shell_angular_momenta_i32.bin",
                np.int32,
                n_shells,
            ),
            "shell_n_primitives": _read_vector(
                static_dir / "shell_n_primitives_i32.bin",
                np.int32,
                n_shells,
            ),
            "shell_ao_starts": _read_vector(
                static_dir / "shell_ao_starts_i32.bin",
                np.int32,
                n_shells,
            ),
            "shell_ao_counts": _read_vector(
                static_dir / "shell_ao_counts_i32.bin",
                np.int32,
                n_shells,
            ),
            "ao_to_atom": _read_vector(
                static_dir / "ao_to_atom_i32.bin",
                np.int32,
                n_basis_functions,
            ),
            "ao_to_shell": _read_vector(
                static_dir / "ao_to_shell_i32.bin",
                np.int32,
                n_basis_functions,
            ),
            "ao_angular_momenta": _read_vector(
                static_dir / "ao_angular_momenta_i32.bin",
                np.int32,
                n_basis_functions,
            ),
            "ao_shell_local_indices": _read_vector(
                static_dir / "ao_shell_local_indices_i32.bin",
                np.int32,
                n_basis_functions,
            ),
            "ao_cartesian_exponents": _read_array(
                static_dir / "ao_cartesian_exponents_i32.bin",
                np.int32,
                (n_basis_functions, 3),
            ),
            "atomic_coordinate_unit": str(static_metadata["atomic_coordinate_unit"]),
        }
        static_np.update(structure_pair_topology)
        static_np["orbital_basis_mask"] = _orbital_basis_mask(
            static_np["orbital_basis_counts"],
            n_basis_functions,
        )
        static_np["orbital_shell_dense_mask"] = _orbital_shell_dense_mask(
            static_np["orbital_basis_index_table"],
            static_np["orbital_basis_mask"],
            static_np["ao_to_shell"],
            static_np["shell_ao_starts"],
            static_np["shell_ao_counts"],
        )
        tensor_map = _to_tensor_map(static_np, self.dtype)
        cache = _StaticSampleCache(data=tensor_map)
        self._static_cache[sample_dir] = cache
        return cache.data

    def _load_step(
        self,
        step_dir: Path,
        sample_metadata: dict[str, Any],
        step_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        cached = self._step_cache.get(step_dir)
        if cached is not None:
            return cached

        n_orbitals = int(sample_metadata["n_orbitals"])
        n_basis_functions = int(sample_metadata["n_basis_functions"])
        n_structures = int(sample_metadata["n_structures"])

        orbital_value_table = _read_array(
            step_dir / "orbital_value_table_f64.bin",
            np.float64,
            (n_orbitals, n_basis_functions),
        )
        overlap_matrix = _column_major_matrix(
            step_dir / "overlap_matrix_f64.bin",
            n_structures,
        )
        hamiltonian_matrix = _column_major_matrix(
            step_dir / "hamiltonian_matrix_f64.bin",
            n_structures,
        )
        one_electron_hamiltonian_matrix = _column_major_matrix(
            step_dir / "one_electron_hamiltonian_matrix_f64.bin",
            n_structures,
        )
        step_np: dict[str, Any] = {
            "accepted_iteration_index": int(step_metadata["accepted_iteration_index"]),
            "target_total_energy": float(step_metadata["total_energy"]),
            "one_electron_reference_energy": float(
                step_metadata.get("one_electron_reference_energy", 0.0)
            ),
            "orbital_value_table": orbital_value_table,
            "overlap_matrix": overlap_matrix,
            "hamiltonian_matrix": hamiltonian_matrix,
            "one_electron_hamiltonian_matrix": one_electron_hamiltonian_matrix,
            "two_electron_hamiltonian_matrix": hamiltonian_matrix
            - one_electron_hamiltonian_matrix,
            "sparse_orbital_energy_gradient": _read_array(
                step_dir / "sparse_orbital_energy_gradient_f64.bin",
                np.float64,
                (n_orbitals, n_basis_functions),
            ),
        }
        reference_energy_gradient = _read_optional_array(
            step_dir / "sparse_orbital_reference_energy_gradient_f64.bin",
            np.float64,
            (n_orbitals, n_basis_functions),
        )
        if reference_energy_gradient is None:
            reference_energy_gradient = np.zeros(
                (n_orbitals, n_basis_functions),
                dtype=np.float64,
            )
        step_np["sparse_orbital_reference_energy_gradient"] = reference_energy_gradient
        step_np["sparse_orbital_residual_energy_gradient"] = (
            step_np["sparse_orbital_energy_gradient"] - reference_energy_gradient
        )
        step_data = _to_tensor_map(step_np, self.dtype)
        if self.cache_steps:
            self._step_cache[step_dir] = step_data
        return step_data


def move_sample_to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
