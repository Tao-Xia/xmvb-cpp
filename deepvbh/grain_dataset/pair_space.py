"""Pair-space layout and tensor builders for DeepVBH."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairSpaceLayout:
  """Defines the packed pair space used by HLSP structure tensors.

  Args:
    max_active_orbitals: Maximum active-orbital count used for dataset padding.

  The class uses the same unordered packed-pair convention as the C++
  `TwoElectronIndexer`. For local orbital indices `i` and `j`, the packed index
  is

  `max(i, j) * (max(i, j) + 1) / 2 + min(i, j)`.
  """

  max_active_orbitals: int

  def packed_pair_count(self, active_orbitals: int | None = None) -> int:
    """Returns the number of packed pairs for the given active-orbital count.

    Args:
      active_orbitals: Number of active orbitals. When omitted, the dataset
        padding size `max_active_orbitals` is used.

    Returns:
      Packed pair count with shape `P = n * (n + 1) // 2`.
    """

    orbitals = (
        self.max_active_orbitals
        if active_orbitals is None
        else active_orbitals
    )
    return orbitals * (orbitals + 1) // 2

  def packed_pair_index(self, orbital_index_a: int, orbital_index_b: int) -> int:
    """Returns the unordered packed pair index for one orbital pair.

    Args:
      orbital_index_a: Zero-based local orbital index.
      orbital_index_b: Zero-based local orbital index.

    Returns:
      Zero-based packed pair index in the range `[0, P)`.
    """

    high_index = max(orbital_index_a, orbital_index_b)
    low_index = min(orbital_index_a, orbital_index_b)
    return high_index * (high_index + 1) // 2 + low_index

  def pair_orbital_indices(self) -> np.ndarray:
    """Returns the packed pair-to-orbital lookup table.

    Returns:
      Array with shape `[P_max, 2]` storing local orbital indices `(high, low)`
      in packed order.
    """

    pair_count = self.packed_pair_count()
    indices = np.zeros((pair_count, 2), dtype=np.int32)
    packed_index = 0
    for high_index in range(self.max_active_orbitals):
      for low_index in range(high_index + 1):
        indices[packed_index, 0] = high_index
        indices[packed_index, 1] = low_index
        packed_index += 1
    return indices

  def pair_mask(self, active_orbitals: int) -> np.ndarray:
    """Builds the valid-token mask for one sample.

    Args:
      active_orbitals: Number of active orbitals in the current sample.

    Returns:
      Boolean mask with shape `[P_max]`.
    """

    mask = np.zeros((self.packed_pair_count(),), dtype=bool)
    mask[: self.packed_pair_count(active_orbitals)] = True
    return mask

  def structure_pair_occupancies(
      self,
      structure_pair_orbital_indices: np.ndarray,
      structure_pair_mask: np.ndarray,
      dtype: np.dtype[np.generic],
  ) -> np.ndarray:
    """Builds packed HLSP pair-occupancy vectors for all structures.

    Args:
      structure_pair_orbital_indices: Array with shape
        `[n_structures, n_structure_pairs, 2]`.
      structure_pair_mask: Boolean array with shape
        `[n_structures, n_structure_pairs]`.
      dtype: Output floating-point dtype.

    Returns:
      Packed pair-occupancy array with shape `[n_structures, P_max]`.
    """

    n_structures = int(structure_pair_orbital_indices.shape[0])
    occupancies = np.zeros(
        (n_structures, self.packed_pair_count()),
        dtype=dtype,
    )
    for structure_index in range(n_structures):
      structure_pairs = structure_pair_orbital_indices[structure_index]
      valid_pairs = structure_pair_mask[structure_index]
      for pair_index, is_valid in enumerate(valid_pairs):
        if not is_valid:
          continue
        left_local = int(structure_pairs[pair_index, 0])
        right_local = int(structure_pairs[pair_index, 1])
        packed_index = self.packed_pair_index(left_local, right_local)
        occupancies[structure_index, packed_index] = 1
    return occupancies

  def pair_feature_matrix(
      self,
      one_electron_matrix: np.ndarray,
      overlap_matrix: np.ndarray,
      dtype: np.dtype[np.generic],
  ) -> np.ndarray:
    """Builds local pair-token features from one-body matrices.

    Args:
      one_electron_matrix: Active-space one-electron matrix with shape `[n, n]`.
      overlap_matrix: Active-space overlap matrix with shape `[n, n]`.
      dtype: Output floating-point dtype.

    Returns:
      Pair feature matrix with shape `[P_max, 3]`, containing
      `[h_ij, S_ij, is_diagonal_pair]` in packed pair order.
    """

    active_orbitals = int(one_electron_matrix.shape[0])
    features = np.zeros((self.packed_pair_count(), 3), dtype=dtype)
    for high_index in range(active_orbitals):
      for low_index in range(high_index + 1):
        packed_index = self.packed_pair_index(high_index, low_index)
        one_electron_value = 0.5 * (
            one_electron_matrix[high_index, low_index]
            + one_electron_matrix[low_index, high_index]
        )
        overlap_value = 0.5 * (
            overlap_matrix[high_index, low_index]
            + overlap_matrix[low_index, high_index]
        )
        features[packed_index, 0] = one_electron_value
        features[packed_index, 1] = overlap_value
        features[packed_index, 2] = 1.0 if high_index == low_index else 0.0
    return features

  def pair_relation_matrix(
      self,
      packed_two_electron_integrals: np.ndarray,
      active_orbitals: int,
      dtype: np.dtype[np.generic],
  ) -> np.ndarray:
    """Decodes the complete packed active-space ERI into pair space.

    Args:
      packed_two_electron_integrals: Packed active-space ERI array with shape
        `[P * (P + 1) // 2]`.
      active_orbitals: Number of active orbitals in the current sample.
      dtype: Output floating-point dtype.

    Returns:
      Symmetric pair-relation matrix with shape `[P_max, P_max]`.
    """

    pair_count = self.packed_pair_count(active_orbitals)
    relation_matrix = np.zeros(
        (self.packed_pair_count(), self.packed_pair_count()),
        dtype=dtype,
    )
    for high_index in range(pair_count):
      packed_row_start = high_index * (high_index + 1) // 2
      for low_index in range(high_index + 1):
        relation_value = packed_two_electron_integrals[packed_row_start + low_index]
        relation_matrix[high_index, low_index] = relation_value
        relation_matrix[low_index, high_index] = relation_value
    return relation_matrix
