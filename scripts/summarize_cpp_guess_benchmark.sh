#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/summarize_cpp_guess_benchmark.sh <benchmark_dir|jobs.tsv> [summary.tsv]

Reads the benchmark jobs.tsv produced by submit_cpp_guess_benchmark_pair.sh and
extracts the key convergence metrics from each run log into a summary table.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

input_path="$1"
if [[ -d "$input_path" ]]; then
  jobs_tsv="${input_path%/}/jobs.tsv"
  summary_tsv="${2:-${input_path%/}/summary.tsv}"
else
  jobs_tsv="$input_path"
  summary_tsv="${2:-$(dirname "$jobs_tsv")/summary.tsv}"
fi

if [[ ! -f "$jobs_tsv" ]]; then
  echo "missing jobs.tsv: $jobs_tsv" >&2
  exit 1
fi

mkdir -p "$(dirname "$summary_tsv")"

printf 'mode\tjob_id\tconverged\ttermination_reason\titerations\tinitial_total_energy\tfinal_total_energy\tenergy_delta\tfinal_gradient_inf_norm\tinput_total_wall_time_seconds\truntime_total_wall_time_seconds\ttotal_wall_time_seconds\trun_log\n' > "$summary_tsv"

tail -n +2 "$jobs_tsv" | while IFS=$'\t' read -r mode job_id run_log trace_root slurm_dir; do
  if [[ -z "${mode}" ]]; then
    continue
  fi

  if [[ ! -f "$run_log" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$job_id" "missing_log" "" "" "" "" "" "" "" "" "" "$run_log" \
      >> "$summary_tsv"
    continue
  fi

  awk -F' = ' \
      -v mode="$mode" \
      -v job_id="$job_id" \
      -v run_log="$run_log" '
    $0 ~ /^[A-Za-z0-9_]+ = / {
      values[$1] = $2
    }
    END {
      energy_delta = ""
      if (values["initial_total_energy"] != "" && values["final_total_energy"] != "") {
        energy_delta = values["final_total_energy"] - values["initial_total_energy"]
      }
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        mode,
        job_id,
        values["converged"],
        values["termination_reason"],
        values["iterations"],
        values["initial_total_energy"],
        values["final_total_energy"],
        energy_delta,
        values["final_gradient_inf_norm"],
        values["input_total_wall_time_seconds"],
        values["runtime_total_wall_time_seconds"],
        values["total_wall_time_seconds"],
        run_log
    }
  ' "$run_log" >> "$summary_tsv"
done

echo "summary_tsv = $summary_tsv"
