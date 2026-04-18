#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_script="${JOB_SCRIPT:-$repo_root/vbscf-cpp.sh}"
benchmark_root="${BENCHMARK_ROOT:-$repo_root/benchmarks}"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
algorithm="${ALGORITHM:-original}"
cpus_per_task="${CPUS_PER_TASK:-1}"
omp_threads="${OMP_NUM_THREADS:-$cpus_per_task}"
partition="${PARTITION:-6226r}"
account="${ACCOUNT:-weiwu}"
qos="${QOS:-}"
time_limit="${TIME_LIMIT:-24:00:00}"
mem="${MEM:-}"
dump_trace="${DUMP_TRACE:-0}"
dry_run="${DRY_RUN:-0}"
job_prefix="${JOB_PREFIX:-vbguess}"
onnx_model="${ONNX_MODEL:-}"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/submit_cpp_guess_benchmark_pair.sh <input.xmi> [xmvb-cpp args...]

This helper submits two single-input benchmark jobs:
  1. legacy_guess
  2. cpp_guess

The purpose is to compare full-convergence behavior for the two initial-guess
paths under otherwise identical xmvb-cpp arguments.

Defaults:
  CPUS_PER_TASK=1
  OMP_NUM_THREADS=1
  OPTIMIZER_BACKEND=lbfgspp
  ALGORITHM=original
  DUMP_TRACE=0

Environment overrides:
  JOB_SCRIPT         Default: <repo>/vbscf-cpp.sh
  BENCHMARK_ROOT     Default: <repo>/benchmarks
  OPTIMIZER_BACKEND  Default: lbfgspp
  ALGORITHM          Default: original
  CPUS_PER_TASK      Default: 1
  OMP_NUM_THREADS    Default: CPUS_PER_TASK
  PARTITION          Default: 6226r
  ACCOUNT            Default: weiwu
  QOS                Default: empty
  TIME_LIMIT         Default: 24:00:00
  MEM                Default: unset
  DUMP_TRACE         Default: 0
  DRY_RUN            Default: 0
  JOB_PREFIX         Default: vbguess
  ONNX_MODEL         Default: empty

Examples:
  bash scripts/submit_cpp_guess_benchmark_pair.sh sample.xmi
  bash scripts/submit_cpp_guess_benchmark_pair.sh sample.xmi --max-iterations 40
  DRY_RUN=1 bash scripts/submit_cpp_guess_benchmark_pair.sh sample.xmi --max-iterations 40
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

input_file="$(readlink -f "$1")"
shift
common_extra_args=("$@")

if [[ ! -f "$input_file" ]]; then
  echo "missing input file: $input_file" >&2
  exit 1
fi

if [[ ! -x "$job_script" ]]; then
  echo "missing job script: $job_script" >&2
  exit 1
fi

if ! [[ "$cpus_per_task" =~ ^[0-9]+$ ]] || (( cpus_per_task < 1 )); then
  echo "CPUS_PER_TASK must be a positive integer" >&2
  exit 1
fi

if ! [[ "$omp_threads" =~ ^[0-9]+$ ]] || (( omp_threads < 1 )); then
  echo "OMP_NUM_THREADS must be a positive integer" >&2
  exit 1
fi

sample_stem="$(basename "${input_file%.xmi}")"
timestamp="$(date +%Y%m%d_%H%M%S)"
benchmark_dir="${BENCHMARK_DIR:-$benchmark_root/${sample_stem}_cpp_guess_pair_${timestamp}}"
submit_log="${benchmark_dir}/submit.log"
jobs_tsv="${benchmark_dir}/jobs.tsv"

mkdir -p "$benchmark_dir"

build_export_spec() {
  local log_dir="$1"
  local dataset_root="$2"
  local export_args=(
    "ALL"
    "OPTIMIZER_BACKEND=${optimizer_backend}"
    "ALGORITHM=${algorithm}"
    "OMP_NUM_THREADS=${omp_threads}"
    "LOG_DIR=${log_dir}"
    "DATASET_ROOT=${dataset_root}"
    "DUMP_TRACE=${dump_trace}"
  )
  if [[ -n "$onnx_model" ]]; then
    export_args+=("ONNX_MODEL=${onnx_model}")
  fi

  local export_spec=""
  local export_arg=""
  for export_arg in "${export_args[@]}"; do
    if [[ -n "$export_spec" ]]; then
      export_spec+=","
    fi
    export_spec+="$export_arg"
  done
  printf '%s' "$export_spec"
}

submit_mode() {
  local mode_name="$1"
  local job_name="$2"
  shift 2
  local mode_extra_args=("$@")
  local mode_dir="${benchmark_dir}/${mode_name}"
  local log_dir="${mode_dir}/logs"
  local slurm_dir="${mode_dir}/slurm"
  local trace_root="${mode_dir}/trace"
  local export_spec
  local job_id

  mkdir -p "$log_dir" "$slurm_dir"
  if [[ "$dump_trace" != "0" && "$dump_trace" != "false" && "$dump_trace" != "FALSE" ]]; then
    mkdir -p "$trace_root"
  fi

  export_spec="$(build_export_spec "$log_dir" "$trace_root")"

  local sbatch_args=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--cpus-per-task=${cpus_per_task}"
    "--output=${slurm_dir}/%x-%j.slog"
    "--error=${slurm_dir}/%x-%j.serr"
    "--export=${export_spec}"
  )

  if [[ -n "$partition" ]]; then
    sbatch_args+=("--partition=${partition}")
  fi
  if [[ -n "$account" ]]; then
    sbatch_args+=("--account=${account}")
  fi
  if [[ -n "$qos" ]]; then
    sbatch_args+=("--qos=${qos}")
  fi
  if [[ -n "$time_limit" ]]; then
    sbatch_args+=("--time=${time_limit}")
  fi
  if [[ -n "$mem" ]]; then
    sbatch_args+=("--mem=${mem}")
  fi

  sbatch_args+=(
    "$job_script"
    "$input_file"
  )
  if (( ${#common_extra_args[@]} > 0 )); then
    sbatch_args+=("${common_extra_args[@]}")
  fi
  if (( ${#mode_extra_args[@]} > 0 )); then
    sbatch_args+=("${mode_extra_args[@]}")
  fi

  {
    echo "mode = ${mode_name}"
    echo "job_name = ${job_name}"
    echo "log_dir = ${log_dir}"
    echo "slurm_dir = ${slurm_dir}"
    echo "trace_root = ${trace_root}"
    if (( ${#mode_extra_args[@]} > 0 )); then
      echo "mode_extra_args = ${mode_extra_args[*]}"
    else
      echo "mode_extra_args = <none>"
    fi
    if (( ${#common_extra_args[@]} > 0 )); then
      echo "common_extra_args = ${common_extra_args[*]}"
    else
      echo "common_extra_args = <none>"
    fi
  } | tee -a "$submit_log"

  if (( dry_run != 0 )); then
    printf 'DRY_RUN ' | tee -a "$submit_log"
    printf '%q ' "${sbatch_args[@]}" | tee -a "$submit_log"
    printf '\n' | tee -a "$submit_log"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$mode_name" "DRY_RUN" "$log_dir/${sample_stem}.log" "$trace_root" "$slurm_dir" \
      >> "$jobs_tsv"
    return
  fi

  job_id="$("${sbatch_args[@]}")"
  echo "submitted_job_id = ${job_id}" | tee -a "$submit_log"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$mode_name" "$job_id" "$log_dir/${sample_stem}.log" "$trace_root" "$slurm_dir" \
    >> "$jobs_tsv"
}

{
  echo "repo_root = $repo_root"
  echo "job_script = $job_script"
  echo "input_file = $input_file"
  echo "sample_stem = $sample_stem"
  echo "benchmark_dir = $benchmark_dir"
  echo "optimizer_backend = $optimizer_backend"
  echo "algorithm = $algorithm"
  echo "cpus_per_task = $cpus_per_task"
  echo "omp_threads = $omp_threads"
  echo "partition = ${partition:-<unset>}"
  echo "account = ${account:-<unset>}"
  echo "qos = ${qos:-<unset>}"
  echo "time_limit = ${time_limit:-<unset>}"
  echo "mem = ${mem:-<unset>}"
  echo "dump_trace = ${dump_trace}"
  echo "submit_log = $submit_log"
} | tee -a "$submit_log"

printf 'mode\tjob_id\trun_log\ttrace_root\tslurm_dir\n' > "$jobs_tsv"

submit_mode "legacy_guess" "${job_prefix}_legacy_${sample_stem}"
submit_mode "cpp_guess" "${job_prefix}_cpp_${sample_stem}" --orbital-guess-source cpp

{
  echo "completed = true"
  echo "jobs_tsv = $jobs_tsv"
  echo "hint = bash ${repo_root}/scripts/summarize_cpp_guess_benchmark.sh ${benchmark_dir}"
} | tee -a "$submit_log"
