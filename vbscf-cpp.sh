#!/usr/bin/env bash

#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=60GB
#SBATCH --exclusive

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sbatch -c <threads> vbscf-cpp.sh <input.xmi> [xmvb-cpp args...]

Notes:
  sbatch -c <threads> controls OMP_NUM_THREADS for xmvb-cpp.
  To compare against legacy XMVB `-n N`, use the same `-c N` here.

Defaults:
  benchmark mode       DUMP_TRACE=0
  optimizer backend    OPTIMIZER_BACKEND=lbfgspp
  algorithm            determined by the input .xmi keywords
  output directory      current submit directory

Optional environment overrides:
  XMVB_CPP_BIN
  XMVB_CPP_KEEP_LD_LIBRARY_PATH=1
  OPTIMIZER_BACKEND
  ORBITAL_GUESS_SOURCE=cpp|legacy
  DUMP_TRACE=1
  DATASET_ROOT
  LOG_DIR
  ONNX_MODEL
  XMVB_CPP_LOG_RUNTIME_PROGRESS=1
  XMVB_CPP_LOG_OBJECTIVE_PROGRESS=1

Examples:
  sbatch -c 8 vbscf-cpp.sh sample.xmi
  sbatch -c 8 vbscf-cpp.sh sample.xmi --max-iterations 40
  sbatch -c 8 vbscf-cpp.sh sample.xmi \
    --structure-space-mode adaptive_mvp \
    --adaptive-determinant-score-mode outside_only
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
extra_args=("${@:2}")
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
repo_root_helper="${script_dir}/scripts/xmvb_cpp_repo_root.sh"
if [[ ! -f "${repo_root_helper}" ]]; then
  echo "missing repo root helper: ${repo_root_helper}" >&2
  exit 1
fi
source "${repo_root_helper}"

repo_root="$(xmvb_cpp_resolve_repo_root "${submit_dir}" "${BASH_SOURCE[0]}" "$1" || true)"
if [[ -z "${repo_root}" ]]; then
  echo "failed to infer repo_root"
  echo "set XMVB_CPP_REPO explicitly or invoke the launcher from a path inside the repo"
  exit 1
fi
binary_path="${XMVB_CPP_BIN:-${repo_root}/build/src/xmvb-cpp.exe}"
input_file="$(
  xmvb_cpp_resolve_input_file "$1" "${submit_dir}" "${script_dir}" "${repo_root}" ||
      true
)"
job_id="${SLURM_JOB_ID:-manual}"
sample_stem="$(basename "${input_file%.xmi}")"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
orbital_guess_source="${ORBITAL_GUESS_SOURCE:-cpp}"
dump_trace="${DUMP_TRACE:-0}"
thread_count="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
log_dir="${LOG_DIR:-$submit_dir}"
run_log="${log_dir}/${sample_stem}.log"
dataset_root="${DATASET_ROOT:-${submit_dir}/${sample_stem}_trace}"
onnx_model="${ONNX_MODEL:-}"

dump_trace_enabled=1
if [[ -z "${dump_trace}" || "${dump_trace}" == "0" ||
      "${dump_trace}" == "false" || "${dump_trace}" == "FALSE" ]]; then
  dump_trace_enabled=0
fi

mkdir -p "${log_dir}"
if (( dump_trace_enabled != 0 )); then
  mkdir -p "${dataset_root}"
fi

exec >"${run_log}" 2>&1

if [[ ! -f "${input_file}" ]]; then
  echo "missing input file: ${input_file}"
  exit 1
fi

if [[ ! -x "${binary_path}" ]]; then
  echo "missing executable: ${binary_path}"
  echo "try: cd ${repo_root} && ./build.sh build"
  exit 1
fi

runtime_env_helper="${repo_root}/scripts/xmvb_cpp_runtime_env.sh"
if [[ ! -f "${runtime_env_helper}" ]]; then
  echo "missing runtime env helper: ${runtime_env_helper}"
  exit 1
fi
source "${runtime_env_helper}"
prepare_xmvb_cpp_runtime_env "${binary_path}" "${repo_root}"

export OMP_NUM_THREADS="${thread_count}"
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_STACKSIZE="${OMP_STACKSIZE:-1G}"

launch_command=()
slurm_exclusive="false"
launch_cpu_bind="direct"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Launch one explicit Slurm step so the OpenMP team stays pinned to cores
  # instead of inheriting whatever placement the batch shell happened to get.
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

ulimit -c 0
ulimit -s unlimited
ulimit -v unlimited

command=(
  "${binary_path}"
  "${input_file}"
  "--optimizer-backend" "${optimizer_backend}"
  "--orbital-guess-source" "${orbital_guess_source}"
)

if (( dump_trace_enabled != 0 )); then
  command+=("--dump-trace-dir" "${dataset_root}")
fi

if [[ -n "${onnx_model}" ]]; then
  command+=("--onnx-model" "${onnx_model}")
fi
if (( ${#extra_args[@]} > 0 )); then
  command+=("${extra_args[@]}")
fi

echo "repo_root = ${repo_root}"
echo "binary = ${binary_path}"
echo "input_file = ${input_file}"
echo "submit_dir = ${submit_dir}"
echo "optimizer_backend = ${optimizer_backend}"
echo "algorithm = <from input file>"
echo "orbital_guess_source = ${orbital_guess_source}"
echo "slurm_exclusive = ${slurm_exclusive}"
echo "launch_cpu_bind = ${launch_cpu_bind}"
echo "omp_threads = ${OMP_NUM_THREADS}"
echo "dump_trace = ${dump_trace_enabled}"
echo "ld_library_path = ${LD_LIBRARY_PATH:-<unset>}"
echo "log_file = ${run_log}"
if (( dump_trace_enabled != 0 )); then
  echo "dataset_root = ${dataset_root}"
fi
if (( ${#extra_args[@]} > 0 )); then
  echo "extra_args = ${extra_args[*]}"
fi

cd "${repo_root}"
"${launch_command[@]}" "${command[@]}"

echo "completed = true"
echo "log_file = ${run_log}"
if (( dump_trace_enabled != 0 )); then
  echo "dataset_root = ${dataset_root}"
fi
