"""Compressed trajectory step selection for DeepVBH training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .schema import SampleDirectoryMetadata
from .step_deltas import StepDeltaProfile, StepDeltaScorer


@dataclass(frozen=True)
class CompressedTrajectoryStepSelector:
  """Selects representative training steps and final-only evaluation steps."""

  train_selected_steps: int = 5
  delta_scorer: StepDeltaScorer = field(default_factory=StepDeltaScorer)

  def validate(self) -> None:
    if self.train_selected_steps <= 0:
      raise ValueError("train_selected_steps must be positive")

  def select_train_steps(
      self,
      sample: SampleDirectoryMetadata,
      step_dirs: list[Path],
  ) -> list[Path]:
    self.validate()
    if not step_dirs:
      raise ValueError("train step selection requires at least one step directory")

    target_count = min(len(step_dirs), self.train_selected_steps)
    if target_count >= len(step_dirs):
      return step_dirs
    if target_count == 1:
      return [step_dirs[-1]]

    profiles = self.delta_scorer.score(sample, step_dirs)
    cumulative_delta_positions = self.cumulative_delta_positions(profiles)
    total_delta = cumulative_delta_positions[-1]
    if total_delta <= 1.0e-12:
      return self.select_by_index_positions(
          step_dirs,
          self.uniform_index_positions(len(step_dirs), target_count),
      )

    step_targets = self.uniform_index_positions(len(step_dirs), target_count)
    delta_targets = self.uniform_delta_positions(total_delta, target_count)
    return self.select_by_delta_positions(
        step_dirs,
        cumulative_delta_positions,
        delta_targets,
        step_targets,
    )

  def select_eval_steps(self, step_dirs: list[Path]) -> list[Path]:
    """Returns the final accepted step for evaluation splits."""

    if not step_dirs:
      raise ValueError("evaluation step selection requires at least one step directory")
    return [step_dirs[-1]]

  def cumulative_delta_positions(
      self,
      profiles: list[StepDeltaProfile],
  ) -> list[float]:
    cumulative_positions: list[float] = []
    running_total = 0.0
    for profile in profiles:
      running_total += profile.combined_delta
      cumulative_positions.append(running_total)
    return cumulative_positions

  def uniform_index_positions(
      self,
      step_count: int,
      target_count: int,
  ) -> list[float]:
    if target_count <= 1:
      return [step_count - 1]
    last_index = step_count - 1
    return [
        step_index * last_index / (target_count - 1)
        for step_index in range(target_count)
    ]

  def uniform_delta_positions(
      self,
      total_delta: float,
      target_count: int,
  ) -> list[float]:
    if target_count <= 1:
      return [total_delta]
    return [
        step_index * total_delta / (target_count - 1)
        for step_index in range(target_count)
    ]

  def select_by_delta_positions(
      self,
      step_dirs: list[Path],
      cumulative_delta_positions: list[float],
      delta_targets: list[float],
      step_targets: list[float],
  ) -> list[Path]:
    chosen_indices: list[int] = []
    used_indices: set[int] = set()
    for delta_target, step_target in zip(delta_targets, step_targets):
      chosen_index = self.choose_delta_index(
          step_dirs,
          cumulative_delta_positions,
          delta_target,
          step_target,
          used_indices,
      )
      chosen_indices.append(chosen_index)
      used_indices.add(chosen_index)
    chosen_indices.sort()
    return [step_dirs[index] for index in chosen_indices]

  def select_by_index_positions(
      self,
      step_dirs: list[Path],
      positions: list[float],
  ) -> list[Path]:
    chosen_indices: list[int] = []
    used_indices: set[int] = set()
    for position in positions:
      chosen_index = self.choose_index(step_dirs, position, used_indices)
      chosen_indices.append(chosen_index)
      used_indices.add(chosen_index)
    chosen_indices.sort()
    return [step_dirs[index] for index in chosen_indices]

  def choose_delta_index(
      self,
      step_dirs: list[Path],
      cumulative_delta_positions: list[float],
      delta_target: float,
      step_target: float,
      used_indices: set[int],
  ) -> int:
    ranked_indices = sorted(
        range(len(step_dirs)),
        key=lambda index: (
            abs(cumulative_delta_positions[index] - delta_target),
            abs(index - step_target),
            index,
        ),
    )
    for index in ranked_indices:
      if index not in used_indices:
        return index
    raise RuntimeError("failed to choose a unique delta-quantile step index")

  def choose_index(
      self,
      step_dirs: list[Path],
      position: float,
      used_indices: set[int],
  ) -> int:
    ranked_indices = sorted(
        range(len(step_dirs)),
        key=lambda index: (abs(index - position), index),
    )
    for index in ranked_indices:
      if index not in used_indices:
        return index
    raise RuntimeError("failed to choose a unique step index")
