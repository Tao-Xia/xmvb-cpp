#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
script_dir="${repo_root}/scripts"
source "${script_dir}/xmvb_cpp_repo_root.sh"

cpp_job_script="${CPP_JOB_SCRIPT:-${repo_root}/vbscf-cpp.sh}"
xmvb_job_script="${XMVB_JOB_SCRIPT:-${repo_root}/xmvb.sh}"
benchmark_root="${BENCHMARK_ROOT:-${repo_root}/benchmarks}"
cpus_per_task="${CPUS_PER_TASK:-32}"
omp_threads="${OMP_NUM_THREADS:-${cpus_per_task}}"
partition="${PARTITION:-6226r}"
account="${ACCOUNT:-weiwu}"
qos="${QOS:-}"
time_limit="${TIME_LIMIT:-24:00:00}"
mem="${MEM:-}"
dump_trace="${DUMP_TRACE:-0}"
dry_run="${DRY_RUN:-0}"
job_prefix="${JOB_PREFIX:-vbcmp}"
legacy_xmvb_version="${LEGACY_XMVB_VERSION:-latest}"
tnhvp_hvp_mode="${TNHVP_HVP_MODE:-exact_ctx}"
tnhvp_max_cg_iterations="${TNHVP_MAX_CG_ITERATIONS:-0}"
tnhvp_disable_outer_response="${TNHVP_DISABLE_OUTER_RESPONSE:-0}"
orbital_guess_source="${ORBITAL_GUESS_SOURCE:-}"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/submit_vb_optimizer_benchmark_triplet.sh <input.xmi> [xmvb-cpp args...]

Submits three benchmark jobs for the same input:
  1. cpp_tnhvp
  2. cpp_lbfgs
  3. legacy_xmvb

Notes:
  Positional extra arguments are forwarded only to the two xmvb-cpp jobs.
  The legacy XMVB job receives only `--version <LEGACY_XMVB_VERSION>` plus
  the input deck, so keep method comparisons driven by the shared `.xmi`.

Environment overrides:
  CPP_JOB_SCRIPT               Default: <repo>/vbscf-cpp.sh
  XMVB_JOB_SCRIPT              Default: <repo>/xmvb.sh
  BENCHMARK_ROOT               Default: <repo>/benchmarks
  CPUS_PER_TASK                Default: 32
  OMP_NUM_THREADS              Default: CPUS_PER_TASK
  PARTITION                    Default: 6226r
  ACCOUNT                      Default: weiwu
  QOS                          Default: empty
  TIME_LIMIT                   Default: 24:00:00
  MEM                          Default: unset
  DUMP_TRACE                   Default: 0
  DRY_RUN                      Default: 0
  JOB_PREFIX                   Default: vbcmp
  LEGACY_XMVB_VERSION          Default: latest
  TNHVP_HVP_MODE               Default: exact_ctx
  TNHVP_MAX_CG_ITERATIONS      Default: 0 (auto)
  TNHVP_DISABLE_OUTER_RESPONSE Default: 0
  ORBITAL_GUESS_SOURCE         Default: launcher default

Examples:
  bash scripts/submit_vb_optimizer_benchmark_triplet.sh test/FeCl2.xmi
  CPUS_PER_TASK=32 bash scripts/submit_vb_optimizer_benchmark_triplet.sh test/241_VBSCF.xmi
  DRY_RUN=1 bash scripts/submit_vb_optimizer_benchmark_triplet.sh src/test_molecule/F2.xmi
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

input_arg="$1"
shift
cpp_common_extra_args=("$@")
submit_dir="$PWD"

if ! input_file="$(
  xmvb_cpp_resolve_input_file "${input_arg}" "${submit_dir}" "${script_dir}" "${repo_root}"
)"; then
  echo "missing input file: ${input_arg}" >&2
  exit 1
fi
input_file="$(readlink -f "${input_file}")"

if [[ ! -x "${cpp_job_script}" ]]; then
  echo "missing xmvb-cpp launcher: ${cpp_job_script}" >&2
  exit 1
fi
if [[ ! -x "${xmvb_job_script}" ]]; then
  echo "missing legacy xmvb launcher: ${xmvb_job_script}" >&2
  exit 1
fi

if ! [[ "${cpus_per_task}" =~ ^[0-9]+$ ]] || (( cpus_per_task < 1 )); then
  echo "CPUS_PER_TASK must be a positive integer" >&2
  exit 1
fi
if ! [[ "${omp_threads}" =~ ^[0-9]+$ ]] || (( omp_threads < 1 )); then
  echo "OMP_NUM_THREADS must be a positive integer" >&2
  exit 1
fi
if ! [[ "${tnhvp_max_cg_iterations}" =~ ^[0-9]+$ ]] ||
    (( tnhvp_max_cg_iterations < 0 )); then
  echo "TNHVP_MAX_CG_ITERATIONS must be a non-negative integer" >&2
  exit 1
fi

sample_stem="$(basename "${input_file%.xmi}")"
timestamp="$(date +%Y%m%d_%H%M%S)"
benchmark_dir="${BENCHMARK_DIR:-${benchmark_root}/${sample_stem}_optimizer_triplet_${timestamp}}"
submit_log="${benchmark_dir}/submit.log"
jobs_tsv="${benchmark_dir}/jobs.tsv"

mkdir -p "${benchmark_dir}"

append_export_arg() {
  local export_spec="$1"
  local name="$2"
  local value="$3"
  if [[ -n "${value}" ]]; then
    export_spec+=",${name}=${value}"
  fi
  printf '%s' "${export_spec}"
}

build_cpp_export_spec() {
  local optimizer_backend="$1"
  local log_dir="$2"
  local trace_root="$3"
  local export_spec="ALL"

  export_spec="$(append_export_arg "${export_spec}" "OPTIMIZER_BACKEND" "${optimizer_backend}")"
  export_spec="$(append_export_arg "${export_spec}" "XMVB_CPP_REPO" "${repo_root}")"
  export_spec="$(append_export_arg "${export_spec}" "OMP_NUM_THREADS" "${omp_threads}")"
  export_spec="$(append_export_arg "${export_spec}" "LOG_DIR" "${log_dir}")"
  export_spec="$(append_export_arg "${export_spec}" "DATASET_ROOT" "${trace_root}")"
  export_spec="$(append_export_arg "${export_spec}" "DUMP_TRACE" "${dump_trace}")"
  export_spec="$(append_export_arg "${export_spec}" "ORBITAL_GUESS_SOURCE" "${orbital_guess_source}")"
  printf '%s' "${export_spec}"
}

build_tnhvp_export_spec() {
  local log_dir="$1"
  local trace_root="$2"
  local export_spec

  export_spec="$(
    build_cpp_export_spec "nonredundant_truncated_newton" "${log_dir}" "${trace_root}"
  )"
  if [[ "${tnhvp_disable_outer_response}" != "0" &&
        "${tnhvp_disable_outer_response}" != "false" &&
        "${tnhvp_disable_outer_response}" != "FALSE" ]]; then
    export_spec="$(append_export_arg \
      "${export_spec}" \
      "XMVB_CPP_DISABLE_EXACT_CTX_OUTER_RESPONSE" \
      "1")"
  fi
  printf '%s' "${export_spec}"
}

append_common_sbatch_args() {
  local -n sbatch_args_ref="$1"

  if [[ -n "${partition}" ]]; then
    sbatch_args_ref+=("--partition=${partition}")
  fi
  if [[ -n "${account}" ]]; then
    sbatch_args_ref+=("--account=${account}")
  fi
  if [[ -n "${qos}" ]]; then
    sbatch_args_ref+=("--qos=${qos}")
  fi
  if [[ -n "${time_limit}" ]]; then
    sbatch_args_ref+=("--time=${time_limit}")
  fi
  if [[ -n "${mem}" ]]; then
    sbatch_args_ref+=("--mem=${mem}")
  fi
}

submit_cpp_mode() {
  local mode_name="$1"
  local job_name="$2"
  local export_spec="$3"
  shift 3
  local mode_extra_args=("$@")
  local mode_dir="${benchmark_dir}/${mode_name}"
  local log_dir="${mode_dir}/logs"
  local slurm_dir="${mode_dir}/slurm"
  local trace_root="${mode_dir}/trace"
  local run_log="${log_dir}/${sample_stem}.log"
  local sbatch_args=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--chdir=${mode_dir}"
    "--cpus-per-task=${cpus_per_task}"
    "--output=${slurm_dir}/%x-%j.slog"
    "--error=${slurm_dir}/%x-%j.serr"
    "--export=${export_spec}"
  )
  local job_id=""

  mkdir -p "${mode_dir}" "${log_dir}" "${slurm_dir}"
  if [[ "${dump_trace}" != "0" && "${dump_trace}" != "false" &&
        "${dump_trace}" != "FALSE" ]]; then
    mkdir -p "${trace_root}"
  fi

  append_common_sbatch_args sbatch_args
  sbatch_args+=("${cpp_job_script}" "${input_file}")
  if (( ${#cpp_common_extra_args[@]} > 0 )); then
    sbatch_args+=("${cpp_common_extra_args[@]}")
  fi
  if (( ${#mode_extra_args[@]} > 0 )); then
    sbatch_args+=("${mode_extra_args[@]}")
  fi

  {
    echo "mode = ${mode_name}"
    echo "job_name = ${job_name}"
    echo "job_script = ${cpp_job_script}"
    echo "work_dir = ${mode_dir}"
    echo "primary_log = ${run_log}"
    echo "secondary_log = <none>"
    echo "slurm_dir = ${slurm_dir}"
    if (( ${#mode_extra_args[@]} > 0 )); then
      echo "mode_extra_args = ${mode_extra_args[*]}"
    else
      echo "mode_extra_args = <none>"
    fi
  } | tee -a "${submit_log}"

  if (( dry_run != 0 )); then
    printf 'DRY_RUN ' | tee -a "${submit_log}"
    printf '%q ' "${sbatch_args[@]}" | tee -a "${submit_log}"
    printf '\n' | tee -a "${submit_log}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${mode_name}" "DRY_RUN" "${cpp_job_script}" "${run_log}" "<none>" "${mode_dir}" "${slurm_dir}" \
      >> "${jobs_tsv}"
    return
  fi

  job_id="$("${sbatch_args[@]}")"
  echo "submitted_job_id = ${job_id}" | tee -a "${submit_log}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode_name}" "${job_id}" "${cpp_job_script}" "${run_log}" "<none>" "${mode_dir}" "${slurm_dir}" \
    >> "${jobs_tsv}"
}

submit_xmvb_mode() {
  local mode_name="$1"
  local job_name="$2"
  local mode_dir="${benchmark_dir}/${mode_name}"
  local slurm_dir="${mode_dir}/slurm"
  local primary_log="${mode_dir}/${sample_stem}.xmo"
  local secondary_log="${mode_dir}/${sample_stem}.cmdout"
  local sbatch_args=(
    sbatch
    --parsable
    "--job-name=${job_name}"
    "--chdir=${mode_dir}"
    "--cpus-per-task=${cpus_per_task}"
    "--output=${slurm_dir}/%x-%j.slog"
    "--error=${slurm_dir}/%x-%j.serr"
    "--export=ALL,OMP_NUM_THREADS=${omp_threads}"
  )
  local job_id=""

  mkdir -p "${mode_dir}" "${slurm_dir}"

  append_common_sbatch_args sbatch_args
  sbatch_args+=(
    "${xmvb_job_script}"
    "--version" "${legacy_xmvb_version}"
    "${input_file}"
  )

  {
    echo "mode = ${mode_name}"
    echo "job_name = ${job_name}"
    echo "job_script = ${xmvb_job_script}"
    echo "work_dir = ${mode_dir}"
    echo "primary_log = ${primary_log}"
    echo "secondary_log = ${secondary_log}"
    echo "slurm_dir = ${slurm_dir}"
    echo "legacy_xmvb_version = ${legacy_xmvb_version}"
  } | tee -a "${submit_log}"

  if (( dry_run != 0 )); then
    printf 'DRY_RUN ' | tee -a "${submit_log}"
    printf '%q ' "${sbatch_args[@]}" | tee -a "${submit_log}"
    printf '\n' | tee -a "${submit_log}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${mode_name}" "DRY_RUN" "${xmvb_job_script}" "${primary_log}" "${secondary_log}" "${mode_dir}" "${slurm_dir}" \
      >> "${jobs_tsv}"
    return
  fi

  job_id="$("${sbatch_args[@]}")"
  echo "submitted_job_id = ${job_id}" | tee -a "${submit_log}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode_name}" "${job_id}" "${xmvb_job_script}" "${primary_log}" "${secondary_log}" "${mode_dir}" "${slurm_dir}" \
    >> "${jobs_tsv}"
}

{
  echo "repo_root = ${repo_root}"
  echo "cpp_job_script = ${cpp_job_script}"
  echo "xmvb_job_script = ${xmvb_job_script}"
  echo "input_file = ${input_file}"
  echo "sample_stem = ${sample_stem}"
  echo "benchmark_dir = ${benchmark_dir}"
  echo "cpus_per_task = ${cpus_per_task}"
  echo "omp_threads = ${omp_threads}"
  echo "partition = ${partition:-<unset>}"
  echo "account = ${account:-<unset>}"
  echo "qos = ${qos:-<unset>}"
  echo "time_limit = ${time_limit:-<unset>}"
  echo "mem = ${mem:-<unset>}"
  echo "dump_trace = ${dump_trace}"
  echo "legacy_xmvb_version = ${legacy_xmvb_version}"
  echo "tnhvp_hvp_mode = ${tnhvp_hvp_mode}"
  echo "tnhvp_max_cg_iterations = ${tnhvp_max_cg_iterations}"
  echo "submit_log = ${submit_log}"
} | tee -a "${submit_log}"

printf 'mode\tjob_id\tjob_script\tprimary_log\tsecondary_log\twork_dir\tslurm_dir\n' > "${jobs_tsv}"

tnhvp_extra_args=(
  "--nonredundant-truncated-newton-hvp-mode" "${tnhvp_hvp_mode}"
)
if (( tnhvp_max_cg_iterations > 0 )); then
  tnhvp_extra_args+=(
    "--nonredundant-truncated-newton-max-cg-iterations"
    "${tnhvp_max_cg_iterations}"
  )
fi

submit_cpp_mode \
  "cpp_tnhvp" \
  "${job_prefix}_tn_${sample_stem}" \
  "$(build_tnhvp_export_spec "${benchmark_dir}/cpp_tnhvp/logs" "${benchmark_dir}/cpp_tnhvp/trace")" \
  "${tnhvp_extra_args[@]}"
submit_cpp_mode \
  "cpp_lbfgs" \
  "${job_prefix}_lb_${sample_stem}" \
  "$(build_cpp_export_spec "lbfgspp" "${benchmark_dir}/cpp_lbfgs/logs" "${benchmark_dir}/cpp_lbfgs/trace")"
submit_xmvb_mode "legacy_xmvb" "${job_prefix}_fx_${sample_stem}"

{
  echo "completed = true"
  echo "jobs_tsv = ${jobs_tsv}"
  echo "hint = bash ${repo_root}/scripts/summarize_vb_optimizer_benchmark.sh ${benchmark_dir}"
} | tee -a "${submit_log}"
