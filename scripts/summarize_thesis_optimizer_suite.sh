#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
single_summary_script="${repo_root}/scripts/summarize_vb_optimizer_benchmark.sh"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/summarize_thesis_optimizer_suite.sh <suite_root> [summary.tsv]

Collects all per-system triplet summaries under a thesis suite directory into
one combined TSV.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

suite_root="$1"
summary_tsv="${2:-${suite_root%/}/suite_summary.tsv}"
systems_tsv="${suite_root%/}/systems.tsv"

if [[ ! -d "${suite_root}" ]]; then
  echo "missing suite_root: ${suite_root}" >&2
  exit 1
fi
if [[ ! -f "${systems_tsv}" ]]; then
  echo "missing systems.tsv: ${systems_tsv}" >&2
  exit 1
fi
if [[ ! -x "${single_summary_script}" ]]; then
  echo "missing single-summary script: ${single_summary_script}" >&2
  exit 1
fi

mkdir -p "$(dirname "${summary_tsv}")"
printf 'system_tag\tinput_file\tmode\tjob_id\tscheduler_state\tscheduler_elapsed_seconds\tconverged\ttermination_reason\titerations\tfinal_total_energy\tfinal_gradient_inf_norm\toptimizer_internal_wall_time_seconds\tend_to_end_wall_time_seconds\txmo_cpu_time_seconds\toptimizer_or_method\thvp_mode\tprimary_log\tsecondary_log\n' \
  > "${summary_tsv}"

tail -n +2 "${systems_tsv}" | while IFS=$'\t' read -r system_tag input_file benchmark_dir; do
  if [[ -z "${system_tag}" ]]; then
    continue
  fi
  if [[ ! -d "${benchmark_dir}" ]]; then
    echo "skip missing benchmark_dir: ${benchmark_dir}" >&2
    continue
  fi
  if [[ ! -f "${benchmark_dir}/jobs.tsv" ]]; then
    echo "skip missing jobs.tsv: ${benchmark_dir}/jobs.tsv" >&2
    continue
  fi

  per_system_summary="${benchmark_dir}/summary.tsv"
  bash "${single_summary_script}" "${benchmark_dir}" "${per_system_summary}" >/dev/null

  tail -n +2 "${per_system_summary}" | while IFS= read -r row; do
    printf '%s\t%s\t%s\n' "${system_tag}" "${input_file}" "${row}" >> "${summary_tsv}"
  done
done

echo "summary_tsv = ${summary_tsv}"
