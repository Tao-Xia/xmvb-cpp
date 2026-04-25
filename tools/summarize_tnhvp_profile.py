#!/usr/bin/env python3
"""Summarize TNHVP profiling logs produced by xmvb-cpp.

The optimizer emits machine-readable `tnhvp_policy`, `tnhvp_outcome`, and
`tnhvp_hvp` lines when the profiling environment variables are enabled.  This
script keeps the post-processing reproducible across sbatch sweeps by extracting
the same wall-time, Krylov, trust-ratio, and final-gradient metrics from every
result file in a directory.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def find_summary_value(text: str, label: str) -> str | None:
  match = re.search(rf"^{re.escape(label)}\s*:\s*(.*)$", text, re.MULTILINE)
  return match.group(1).strip() if match else None


def parse_float_field(line: str, name: str) -> float | None:
  match = re.search(rf"{re.escape(name)}=([-+0-9.eE]+)", line)
  if not match:
    return None
  try:
    return float(match.group(1))
  except ValueError:
    return None


def parse_int_field(line: str, name: str) -> int | None:
  value = parse_float_field(line, name)
  return None if value is None else int(value)


def summarize_hvp_file(path: Path, text: str) -> None:
  def metric(name: str) -> float | None:
    match = re.search(rf"^{re.escape(name)} = ([-+0-9.eE]+)", text, re.MULTILINE)
    return float(match.group(1)) if match else None

  full = metric("full_cached_external_avg_wall_time_seconds")
  core = metric("core_only_cached_external_avg_wall_time_seconds")
  outer = metric("outer_only_cached_external_avg_wall_time_seconds")
  active_2e = metric("full_cached_diag_avg_active_2e_wall_time_seconds")
  h1e = metric("full_cached_diag_avg_h1e_fused_wall_time_seconds")
  outer_total = metric("full_cached_diag_avg_outer_response_wall_time_seconds")
  outer_integrals = metric(
      "full_cached_diag_avg_outer_response_active_space_integrals_wall_time_seconds")
  outer_structure = metric(
      "full_cached_diag_avg_outer_response_structure_matrices_wall_time_seconds")
  outer_gradient = metric(
      "full_cached_diag_avg_outer_response_active_gradient_wall_time_seconds")
  outer_pullback = metric(
      "full_cached_diag_avg_outer_response_orbital_pullback_wall_time_seconds")

  print(f"{path.name}: HVP")
  if full is not None and core is not None:
    ratio = full / core if core > 0.0 else float("inf")
    print(f"  full={full:.6g}s core={core:.6g}s full/core={ratio:.3g}")
  if outer is not None or outer_total is not None:
    print(f"  outer_only={outer or 0.0:.6g}s outer_in_full={outer_total or 0.0:.6g}s")
  print(f"  h1e={h1e or 0.0:.6g}s active2e={active_2e or 0.0:.6g}s")
  print(
      "  outer_breakdown "
      f"integrals={outer_integrals or 0.0:.6g}s "
      f"structure={outer_structure or 0.0:.6g}s "
      f"active_grad={outer_gradient or 0.0:.6g}s "
      f"pullback={outer_pullback or 0.0:.6g}s")


def summarize_tn_file(path: Path, text: str) -> None:
  policies = re.findall(r"tnhvp_policy[^\n]*", text)
  outcomes = re.findall(r"tnhvp_outcome[^\n]*", text)
  hvp_lines = re.findall(r"tnhvp_hvp[^\n]*", text)
  iter_lines = re.findall(
      r"^\s*(\d+)\s+(-?\d+\.\d+)\s+([-+0-9.eE]+)\s+"
      r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
      text,
      re.MULTILINE)

  accepted = sum(" accepted=true " in f" {line} " for line in outcomes)
  rejected = sum(" accepted=false " in f" {line} " for line in outcomes)
  cheap_reject = sum(" cheap_reject=true" in line for line in outcomes)
  full_retry = sum(" full_retry=true" in line for line in outcomes)
  refined = sum(" refined=true" in line for line in outcomes)
  rescue = sum(" krylov_rescue=true" in line for line in outcomes)
  policy_outer = sum(" outer=true" in line for line in policies)
  stall_full = sum(" stall_full=true" in line for line in policies)

  cg_iterations = [
      value
      for line in outcomes
      if (value := parse_int_field(line, "cg_iters")) is not None
  ]
  trust_ratios = [
      value
      for line in outcomes
      if (value := parse_float_field(line, "trust")) is not None
  ]
  cheap_apply_times = [
      value
      for line in hvp_lines
      if "role=cheap" in line and
      (value := parse_float_field(line, "avg_apply_s")) is not None
  ]
  outer_apply_times = [
      value
      for line in hvp_lines
      if "outer_included=true" in line and
      (value := parse_float_field(line, "avg_apply_s")) is not None
  ]

  print(f"{path.name}: TN")
  print(
      "  status="
      f"{find_summary_value(text, 'Status') or 'running/partial'} "
      f"scf={find_summary_value(text, 'SCF iteration wall time') or 'n/a'} "
      f"final_proj_inf={find_summary_value(text, 'Final projected |g|_inf') or 'n/a'}")
  if iter_lines:
    last_iter = iter_lines[-1]
    print(
        f"  last_iter={last_iter[0]} E={last_iter[1]} "
        f"ginf={last_iter[3]} g2={last_iter[4]}")
  print(
      f"  outcomes={len(outcomes)} accepted={accepted} rejected={rejected} "
      f"cheap_reject={cheap_reject} full_retry={full_retry} "
      f"refined={refined} rescue={rescue}")
  if cg_iterations:
    print(
        f"  cg_avg={sum(cg_iterations) / len(cg_iterations):.3g} "
        f"cg_max={max(cg_iterations)}")
  if trust_ratios:
    print(
        f"  trust_avg={sum(trust_ratios) / len(trust_ratios):.3g} "
        f"trust_min={min(trust_ratios):.3g} "
        f"trust_max={max(trust_ratios):.3g}")
  print(f"  policy_outer={policy_outer} stall_full={stall_full}")
  if cheap_apply_times:
    print(f"  cheap_hvp_avg={sum(cheap_apply_times) / len(cheap_apply_times):.6g}s")
  if outer_apply_times:
    print(f"  outer_hvp_avg={sum(outer_apply_times) / len(outer_apply_times):.6g}s")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "paths",
      nargs="*",
      type=Path,
      default=[Path("test/tnhvp_profile/results")],
      help="Result files or directories to summarize.")
  args = parser.parse_args()

  files: list[Path] = []
  for path in args.paths:
    if path.is_dir():
      files.extend(sorted(path.glob("*.out")))
    else:
      files.append(path)

  for path in files:
    text = path.read_text(errors="replace")
    if "full_cached_external_avg_wall_time_seconds" in text:
      summarize_hvp_file(path, text)
    elif "tnhvp_" in text or "SCF iteration wall time" in text:
      summarize_tn_file(path, text)


if __name__ == "__main__":
  main()
