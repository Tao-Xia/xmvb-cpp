"""Trajectory step scoring based on model-input integral changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .reader import TraceArrayReader
from .schema import SampleDirectoryMetadata


@dataclass(frozen=True)
class StepDeltaProfile:
  """Relative input-change metrics for one accepted SCF step."""

  step_dir: Path
  overlap_delta: float
  one_electron_delta: float
  two_electron_delta: float
  combined_delta: float


@dataclass
class StepDeltaScorer:
  """Scores steps by the relative change of the model input integrals."""

  reader: TraceArrayReader = field(default_factory=TraceArrayReader)

  def score(
      self,
      sample: SampleDirectoryMetadata,
      step_dirs: list[Path],
  ) -> list[StepDeltaProfile]:
    if not step_dirs:
      raise ValueError("step delta scoring requires at least one step directory")

    profiles: list[StepDeltaProfile] = []
    previous_overlap = None
    previous_one_electron = None
    previous_two_electron = None

    for step_dir in step_dirs:
      overlap = self.reader.read_square_matrix(
          step_dir / "active_orbital_overlap_matrix_f64.bin",
          sample.n_active_orbitals,
          np.float64,
      )
      one_electron = self.reader.read_square_matrix(
          step_dir / "active_one_electron_integrals_f64.bin",
          sample.n_active_orbitals,
          np.float64,
      )
      two_electron = self.reader.read_vector(
          step_dir / "packed_active_two_electron_integrals_f64.bin",
          np.float64,
      )

      if (
          previous_overlap is None
          or previous_one_electron is None
          or previous_two_electron is None
      ):
        overlap_delta = 0.0
        one_electron_delta = 0.0
        two_electron_delta = 0.0
      else:
        overlap_delta = self.relative_delta(
            previous_overlap,
            overlap,
        )
        one_electron_delta = self.relative_delta(
            previous_one_electron,
            one_electron,
        )
        two_electron_delta = self.relative_delta(
            previous_two_electron,
            two_electron,
        )

      profiles.append(
          StepDeltaProfile(
              step_dir=step_dir,
              overlap_delta=overlap_delta,
              one_electron_delta=one_electron_delta,
              two_electron_delta=two_electron_delta,
              combined_delta=(
                  overlap_delta + one_electron_delta + two_electron_delta
              ),
          )
      )
      previous_overlap = overlap
      previous_one_electron = one_electron
      previous_two_electron = two_electron

    return profiles

  def relative_delta(
      self,
      previous_array: np.ndarray,
      current_array: np.ndarray,
  ) -> float:
    delta_norm = float(np.linalg.norm(current_array - previous_array))
    reference_norm = max(float(np.linalg.norm(current_array)), 1.0e-12)
    return delta_norm / reference_norm
