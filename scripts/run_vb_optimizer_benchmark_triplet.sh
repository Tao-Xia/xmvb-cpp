#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

usage() {
  cat >&2 <<'EOF'
usage: sbatch -c <threads> scripts/run_vb_optimizer_benchmark_triplet.sh <benchmark_dir> <input.xmi> [xmvb-cpp args...]

Runs cpp_tnhvp, cpp_lbfgs, and legacy_xmvb sequentially inside one Slurm
allocation so all modes share the same node placement. Optional warm-up runs
can be requested through BENCHMARK_WARMUP_RUNS.
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

benchmark_dir="$1"
input_file="$2"
shift 2
cpp_common_extra_args=("$@")

if [[ ! -d "${benchmark_dir}" ]]; then
  echo "missing benchmark_dir: ${benchmark_dir}" >&2
  exit 1
fi
if [[ ! -f "${input_file}" ]]; then
  echo "missing input file: ${input_file}" >&2
  exit 1
fi

cpp_job_script="${CPP_JOB_SCRIPT:-${repo_root}/vbscf-cpp.sh}"
xmvb_job_script="${XMVB_JOB_SCRIPT:-${repo_root}/xmvb.sh}"
legacy_xmvb_version="${LEGACY_XMVB_VERSION:-latest}"
tnhvp_hvp_mode="${TNHVP_HVP_MODE:-exact_ctx}"
tnhvp_max_cg_iterations="${TNHVP_MAX_CG_ITERATIONS:-0}"
tnhvp_disable_outer_response="${TNHVP_DISABLE_OUTER_RESPONSE:-0}"
orbital_guess_source="${ORBITAL_GUESS_SOURCE:-}"
dump_trace="${DUMP_TRACE:-0}"
warmup_runs="${BENCHMARK_WARMUP_RUNS:-1}"
job_prefix="${JOB_PREFIX:-vbcmp}"
thread_count="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

sample_stem="$(basename "${input_file%.xmi}")"
run_log="${benchmark_dir}/run.log"

if ! [[ "${warmup_runs}" =~ ^[0-9]+$ ]]; then
  echo "BENCHMARK_WARMUP_RUNS must be a non-negative integer" >&2
  exit 1
fi

mkdir -p "${benchmark_dir}"
exec > >(tee -a "${run_log}") 2>&1

run_cpp_mode() {
  local mode_name="$1"
  local optimizer_backend="$2"
  shift 2
  local mode_extra_args=("$@")
  local mode_dir="${benchmark_dir}/${mode_name}"
  local mode_log_dir="${mode_dir}/logs"
  local mode_trace_dir="${mode_dir}/trace"
  local warmup_index=0

  mkdir -p "${mode_dir}" "${mode_log_dir}"

  for ((warmup_index = 1; warmup_index <= warmup_runs; ++warmup_index)); do
    local warmup_dir="${mode_dir}/warmup_${warmup_index}"
    local warmup_log_dir="${warmup_dir}/logs"
    local warmup_trace_dir="${warmup_dir}/trace"
    mkdir -p "${warmup_log_dir}"
    if [[ "${dump_trace}" != "0" && "${dump_trace}" != "false" &&
          "${dump_trace}" != "FALSE" ]]; then
      mkdir -p "${warmup_trace_dir}"
    fi
    echo "[warmup ${mode_name}] run ${warmup_index}/${warmup_runs}"
    (
      cd "${warmup_dir}"
      env \
        SLURM_SUBMIT_DIR="${warmup_dir}" \
        SLURM_JOB_NAME="${job_prefix}_${mode_name}_warmup_${sample_stem}" \
        XMVB_CPP_REPO="${repo_root}" \
        OPTIMIZER_BACKEND="${optimizer_backend}" \
        OMP_NUM_THREADS="${thread_count}" \
        LOG_DIR="${warmup_log_dir}" \
        DATASET_ROOT="${warmup_trace_dir}" \
        DUMP_TRACE="${dump_trace}" \
        ORBITAL_GUESS_SOURCE="${orbital_guess_source}" \
        bash "${cpp_job_script}" "${input_file}" "${cpp_common_extra_args[@]}" \
          "${mode_extra_args[@]}"
    )
  done

  if [[ "${dump_trace}" != "0" && "${dump_trace}" != "false" &&
        "${dump_trace}" != "FALSE" ]]; then
    mkdir -p "${mode_trace_dir}"
  fi
  echo "[measure ${mode_name}]"
  (
    cd "${mode_dir}"
    env \
      SLURM_SUBMIT_DIR="${mode_dir}" \
      SLURM_JOB_NAME="${job_prefix}_${mode_name}_${sample_stem}" \
      XMVB_CPP_REPO="${repo_root}" \
      OPTIMIZER_BACKEND="${optimizer_backend}" \
      OMP_NUM_THREADS="${thread_count}" \
      LOG_DIR="${mode_log_dir}" \
      DATASET_ROOT="${mode_trace_dir}" \
      DUMP_TRACE="${dump_trace}" \
      ORBITAL_GUESS_SOURCE="${orbital_guess_source}" \
      bash "${cpp_job_script}" "${input_file}" "${cpp_common_extra_args[@]}" \
        "${mode_extra_args[@]}"
  )
}

run_xmvb_mode() {
  local mode_name="$1"
  local mode_dir="${benchmark_dir}/${mode_name}"
  local warmup_index=0

  mkdir -p "${mode_dir}"

  for ((warmup_index = 1; warmup_index <= warmup_runs; ++warmup_index)); do
    local warmup_dir="${mode_dir}/warmup_${warmup_index}"
    mkdir -p "${warmup_dir}"
    echo "[warmup ${mode_name}] run ${warmup_index}/${warmup_runs}"
    (
      cd "${warmup_dir}"
      env \
        SLURM_JOB_NAME="${job_prefix}_${mode_name}_warmup_${sample_stem}" \
        OMP_NUM_THREADS="${thread_count}" \
        bash "${xmvb_job_script}" --version "${legacy_xmvb_version}" "${input_file}"
    )
  done

  echo "[measure ${mode_name}]"
  (
    cd "${mode_dir}"
    env \
      SLURM_JOB_NAME="${job_prefix}_${mode_name}_${sample_stem}" \
      OMP_NUM_THREADS="${thread_count}" \
      bash "${xmvb_job_script}" --version "${legacy_xmvb_version}" "${input_file}"
  )
}

tnhvp_extra_args=(
  "--nonredundant-truncated-newton-hvp-mode" "${tnhvp_hvp_mode}"
)
if [[ "${tnhvp_disable_outer_response}" != "0" &&
      "${tnhvp_disable_outer_response}" != "false" &&
      "${tnhvp_disable_outer_response}" != "FALSE" ]]; then
  tnhvp_extra_args+=("--disable-exact-ctx-outer-response")
fi
if (( tnhvp_max_cg_iterations > 0 )); then
  tnhvp_extra_args+=(
    "--nonredundant-truncated-newton-max-cg-iterations"
    "${tnhvp_max_cg_iterations}"
  )
fi

echo "benchmark_dir = ${benchmark_dir}"
echo "input_file = ${input_file}"
echo "sample_stem = ${sample_stem}"
echo "thread_count = ${thread_count}"
echo "warmup_runs = ${warmup_runs}"
echo "legacy_xmvb_version = ${legacy_xmvb_version}"
echo "tnhvp_hvp_mode = ${tnhvp_hvp_mode}"
echo "tnhvp_max_cg_iterations = ${tnhvp_max_cg_iterations}"
echo "dump_trace = ${dump_trace}"
echo "host = $(hostname)"

run_cpp_mode "cpp_lbfgs" "lbfgspp"
run_cpp_mode "cpp_tnhvp" "nonredundant_truncated_newton" "${tnhvp_extra_args[@]}"
run_xmvb_mode "legacy_xmvb"

echo "completed = true"
