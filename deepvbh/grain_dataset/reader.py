"""Low-level binary readers for DeepVBH trace exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class TraceArrayReader:
  """Loads binary arrays and metadata from exported DeepVBH traces."""

  def read_json(self, path: Path) -> dict[str, Any]:
    """Loads one JSON file.

    Args:
      path: Path to the JSON file.

    Returns:
      Parsed JSON mapping.
    """

    return json.loads(path.read_text())

  def read_scalar(self, path: Path, dtype: np.dtype[Any]) -> Any:
    """Loads one scalar value from a binary file.

    Args:
      path: Path to the binary file.
      dtype: Scalar dtype stored on disk.

    Returns:
      Scalar value with the requested dtype.
    """

    values = np.fromfile(path, dtype=dtype)
    if values.size != 1:
      raise ValueError(f"expected scalar file at {path}, got {values.size} values")
    return values[0]

  def read_vector(self, path: Path, dtype: np.dtype[Any]) -> np.ndarray:
    """Loads one flat binary array.

    Args:
      path: Path to the binary file.
      dtype: Element dtype stored on disk.

    Returns:
      Vector with shape `[length]`.
    """

    return np.fromfile(path, dtype=dtype)

  def read_optional_vector(
      self,
      path: Path,
      dtype: np.dtype[Any],
  ) -> np.ndarray | None:
    """Loads one optional flat binary array.

    Args:
      path: Path to the optional binary file.
      dtype: Element dtype stored on disk.

    Returns:
      Vector with shape `[length]` when the file exists, otherwise `None`.
    """

    if not path.exists():
      return None
    return self.read_vector(path, dtype)

  def read_square_matrix(
      self,
      path: Path,
      size: int,
      dtype: np.dtype[Any],
  ) -> np.ndarray:
    """Loads one square matrix stored in Fortran order.

    Args:
      path: Path to the binary file.
      size: Matrix dimension.
      dtype: Element dtype stored on disk.

    Returns:
      Matrix with shape `[size, size]`.
    """

    values = self.read_vector(path, dtype)
    expected_size = size * size
    if values.size != expected_size:
      raise ValueError(
          f"expected {expected_size} values for square matrix at {path}, "
          f"got {values.size}"
      )
    return values.reshape((size, size), order="F")

  def pad_square_matrix(
      self,
      matrix: np.ndarray,
      target_size: int,
      dtype: np.dtype[Any],
  ) -> np.ndarray:
    """Zero-pads one square matrix to a larger square shape.

    Args:
      matrix: Input matrix with shape `[size, size]`.
      target_size: Output matrix dimension.
      dtype: Output dtype.

    Returns:
      Zero-padded matrix with shape `[target_size, target_size]`.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
      raise ValueError(f"expected square matrix, got shape {matrix.shape}")
    if matrix.shape[0] > target_size:
      raise ValueError(
          f"cannot pad matrix of size {matrix.shape[0]} into target size {target_size}"
      )
    padded = np.zeros((target_size, target_size), dtype=dtype)
    size = matrix.shape[0]
    padded[:size, :size] = matrix.astype(dtype, copy=False)
    return padded
