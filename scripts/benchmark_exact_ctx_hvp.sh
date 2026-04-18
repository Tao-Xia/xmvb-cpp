#!/usr/bin/env bash

#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=40GB
#SBATCH --exclusive

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sbatch -c <threads> scripts/benchmark_exact_ctx_hvp.sh <input.xmi> [benchmark args...]

Defaults:
  repeats               1
  warmup                0
  include_uncached      false
  output directory      current submit directory

Optional environment overrides:
  XMVB_CPP_REPO
  XMVB_CPP_BENCH_BIN
  XMVB_CPP_KEEP_LD_LIBRARY_PATH=1
  LOG_DIR

Examples:
  sbatch -c 1 scripts/benchmark_exact_ctx_hvp.sh test/241_VBSCF.xmi
  sbatch -c 8 scripts/benchmark_exact_ctx_hvp.sh test/241_VBSCF.xmi --repeats 3
  sbatch -c 32 scripts/benchmark_exact_ctx_hvp.sh test/10698_VBSCF.xmi \
    --nonredundant-adapt false --include-uncached false
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
script_path="${BASH_SOURCE[0]}"
script_dir="$(cd "$(dirname "${script_path}")" && pwd -P)"

source "${script_dir}/xmvb_cpp_repo_root.sh"

if ! repo_root="$(xmvb_cpp_resolve_repo_root "${submit_dir}" "${script_path}" "${1}")"; then
  echo "failed to infer repo_root" >&2
  echo "set XMVB_CPP_REPO explicitly or submit from <repo>/test" >&2
  exit 1
fi

if ! input_file="$(xmvb_cpp_resolve_input_file "${1}" "${submit_dir}" "${script_dir}" "${repo_root}")"; then
  echo "missing input file: ${1}" >&2
  exit 1
fi

benchmark_bin="${XMVB_CPP_BENCH_BIN:-${repo_root}/build/src/benchmark_exact_ctx_hvp}"
if [[ ! -x "${benchmark_bin}" ]]; then
  echo "missing benchmark executable: ${benchmark_bin}" >&2
  echo "try: cd ${repo_root} && cmake --build build --target benchmark_exact_ctx_hvp -j8" >&2
  exit 1
fi

runtime_env_helper="${repo_root}/scripts/xmvb_cpp_runtime_env.sh"
if [[ ! -f "${runtime_env_helper}" ]]; then
  echo "missing runtime env helper: ${runtime_env_helper}" >&2
  exit 1
fi
source "${runtime_env_helper}"
prepare_xmvb_cpp_runtime_env "${benchmark_bin}" "${repo_root}"

thread_count="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
log_dir="${LOG_DIR:-$submit_dir}"
job_id="${SLURM_JOB_ID:-manual}"
sample_stem="$(basename "${input_file%.xmi}")"
stdout_log="${log_dir}/${sample_stem}.bench-hvp.${job_id}.stdout.log"
stderr_log="${log_dir}/${sample_stem}.bench-hvp.${job_id}.stderr.log"
mkdir -p "${log_dir}"

prepare_xmvb_cpp_thread_env "${thread_count}"

launch_command=()
slurm_exclusive="false"
launch_cpu_bind="direct"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Benchmark jobs should run as one explicitly core-bound step so repeated
  # measurements are comparable across submissions.
  launch_command=(
    srun
    --ntasks=1
    --cpus-per-task="${thread_count}"
    --cpu-bind=cores
    --exact
  )
  slurm_exclusive="true"
  launch_cpu_bind="cores"
fi

extra_args=("${@:2}")
command=(
  "${benchmark_bin}"
  "${input_file}"
  "--repeats" "1"
  "--warmup" "0"
  "--include-uncached" "false"
)
if (( ${#extra_args[@]} > 0 )); then
  command+=("${extra_args[@]}")
fi

echo "repo_root = ${repo_root}"
echo "benchmark = ${benchmark_bin}"
echo "input_file = ${input_file}"
echo "hostname = $(hostname)"
echo "slurm_job_id = ${SLURM_JOB_ID:-<unset>}"
echo "slurm_nodelist = ${SLURM_JOB_NODELIST:-<unset>}"
echo "slurm_exclusive = ${slurm_exclusive}"
echo "launch_cpu_bind = ${launch_cpu_bind}"
log_xmvb_cpp_thread_env
echo "stdout_log = ${stdout_log}"
echo "stderr_log = ${stderr_log}"
if (( ${#extra_args[@]} > 0 )); then
  echo "extra_args = ${extra_args[*]}"
fi

cd "${repo_root}"
"${launch_command[@]}" "${command[@]}" >"${stdout_log}" 2>"${stderr_log}"

echo "completed = true"
echo "stdout_log = ${stdout_log}"
echo "stderr_log = ${stderr_log}"
