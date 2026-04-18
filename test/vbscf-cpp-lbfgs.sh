#!/usr/bin/env bash

#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=60GB
#SBATCH --exclusive

set -euo pipefail

xmvb_cpp_resolve_input_file_local() {
  local raw_path="${1:-}"
  local submit_dir="${2:-}"
  local script_dir="${3:-}"
  local repo_root="${4:-}"
  local candidate=""

  if [[ -z "${raw_path}" ]]; then
    return 1
  fi

  for candidate in \
    "${raw_path}" \
    "${submit_dir}/${raw_path}" \
    "${script_dir}/${raw_path}" \
    "${repo_root}/${raw_path}" \
    "${repo_root}/test/${raw_path}" \
    "${repo_root}/test_molecule/${raw_path}" \
    "${repo_root}/src/test_molecule/${raw_path}"; do
    if [[ -f "${candidate}" ]]; then
      readlink -f "${candidate}"
      return 0
    fi
  done
  return 1
}

_xmvb_cpp_is_truthy() {
  local value="${1:-}"
  case "${value}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_xmvb_cpp_is_conda_lib_dir() {
  local dir="$1"
  case "${dir}" in
    */anaconda*/lib|*/anaconda*/lib64|*/anaconda*/lib/*|*/anaconda*/lib64/*)
      return 0
      ;;
    */miniconda*/lib|*/miniconda*/lib64|*/miniconda*/lib/*|*/miniconda*/lib64/*)
      return 0
      ;;
    */miniforge*/lib|*/miniforge*/lib64|*/miniforge*/lib/*|*/miniforge*/lib64/*)
      return 0
      ;;
    */mambaforge*/lib|*/mambaforge*/lib64|*/mambaforge*/lib/*|*/mambaforge*/lib64/*)
      return 0
      ;;
    */conda/envs/*/lib|*/conda/envs/*/lib64|*/conda/envs/*/lib/*|*/conda/envs/*/lib64/*)
      return 0
      ;;
    */condabin)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

_xmvb_cpp_append_ld_path_dir() {
  local dir="$1"
  [[ -n "${dir}" ]] || return 0
  [[ -d "${dir}" ]] || return 0

  if [[ -n "${_XMVB_CPP_LD_LIBRARY_PATH_RESULT:-}" ]]; then
    case ":${_XMVB_CPP_LD_LIBRARY_PATH_RESULT}:" in
      *:"${dir}":*)
        return 0
        ;;
    esac
    _XMVB_CPP_LD_LIBRARY_PATH_RESULT="${_XMVB_CPP_LD_LIBRARY_PATH_RESULT}:${dir}"
  else
    _XMVB_CPP_LD_LIBRARY_PATH_RESULT="${dir}"
  fi
}

prepare_xmvb_cpp_runtime_env_local() {
  local binary_path="$1"
  local repo_root="${2:-}"
  local original_ld_library_path="${LD_LIBRARY_PATH-}"
  local parsed_runpath=""
  local dir=""
  local -a path_entries=()

  _XMVB_CPP_LD_LIBRARY_PATH_RESULT=""

  if command -v readelf >/dev/null 2>&1; then
    parsed_runpath="$(
      readelf -d "${binary_path}" 2>/dev/null |
        sed -n \
          -e 's/.*Library runpath: \[\(.*\)\]/\1/p' \
          -e 's/.*Library rpath: \[\(.*\)\]/\1/p' |
        head -n 1
    )"
  fi

  if [[ -n "${parsed_runpath}" ]]; then
    IFS=':' read -r -a path_entries <<< "${parsed_runpath}"
    for dir in "${path_entries[@]}"; do
      _xmvb_cpp_append_ld_path_dir "${dir}"
    done
  fi

  if [[ -n "${repo_root}" ]]; then
    _xmvb_cpp_append_ld_path_dir \
      "${repo_root}/third_party/onnxruntime-linux-x64-1.20.1/lib"
  fi

  if [[ -n "${original_ld_library_path}" ]]; then
    IFS=':' read -r -a path_entries <<< "${original_ld_library_path}"
    for dir in "${path_entries[@]}"; do
      if ! _xmvb_cpp_is_truthy "${XMVB_CPP_KEEP_LD_LIBRARY_PATH:-0}" &&
          _xmvb_cpp_is_conda_lib_dir "${dir}"; then
        continue
      fi
      _xmvb_cpp_append_ld_path_dir "${dir}"
    done
  fi

  export XMVB_CPP_ORIGINAL_LD_LIBRARY_PATH="${original_ld_library_path}"
  if [[ -n "${_XMVB_CPP_LD_LIBRARY_PATH_RESULT}" ]]; then
    export LD_LIBRARY_PATH="${_XMVB_CPP_LD_LIBRARY_PATH_RESULT}"
  else
    unset LD_LIBRARY_PATH
  fi
}

prepare_xmvb_cpp_thread_env_local() {
  local thread_count="${1:-1}"

  export OMP_NUM_THREADS="${thread_count}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export GOTO_NUM_THREADS="${GOTO_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
  export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
  export OMP_PLACES="${OMP_PLACES:-cores}"
  export OMP_STACKSIZE="${OMP_STACKSIZE:-1G}"
}

log_xmvb_cpp_thread_env_local() {
  echo "omp_threads = ${OMP_NUM_THREADS:-<unset>}"
  echo "openblas_threads = ${OPENBLAS_NUM_THREADS:-<unset>}"
  echo "goto_threads = ${GOTO_NUM_THREADS:-<unset>}"
  echo "mkl_threads = ${MKL_NUM_THREADS:-<unset>}"
  echo "mkl_dynamic = ${MKL_DYNAMIC:-<unset>}"
  echo "omp_proc_bind = ${OMP_PROC_BIND:-<unset>}"
  echo "omp_places = ${OMP_PLACES:-<unset>}"
  echo "omp_stacksize = ${OMP_STACKSIZE:-<unset>}"
}

usage() {
  cat >&2 <<'EOF'
usage: sbatch -c <threads> vbscf-cpp.sh <input.xmi> [xmvb-cpp args...]

Defaults:
  benchmark mode       DUMP_TRACE=0
  optimizer backend    OPTIMIZER_BACKEND=lbfgspp
  output directory      current submit directory

Optional environment overrides:
  XMVB_CPP_KEEP_LD_LIBRARY_PATH=1
  OPTIMIZER_BACKEND
  DUMP_TRACE=1
  DATASET_ROOT
  LOG_DIR
  ONNX_MODEL
  XMVB_CPP_LOG_RUNTIME_PROGRESS=1
  XMVB_CPP_LOG_OBJECTIVE_PROGRESS=1

Examples:
  sbatch -c 1 vbscf-cpp.sh sample.xmi
  sbatch -c 32 /path/to/vbscf-cpp-lbfgs.sh 240_VBSCF.xmi
  sbatch -c 32 /path/to/vbscf-cpp-lbfgs.sh /path/to/input.xmi
  sbatch -c 1 vbscf-cpp.sh sample.xmi --max-iterations 40
  sbatch -c 1 vbscf-cpp.sh sample.xmi \
    --structure-space-mode adaptive_mvp \
    --adaptive-determinant-score-mode outside_only
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
input_arg="$1"
repo_root="/pool1/home/xiatao/project/xmvb-cpp"
binary_path="/pool1/home/xiatao/project/xmvb-cpp/build/src/xmvb-cpp.exe"
if ! input_file="$(xmvb_cpp_resolve_input_file_local "${input_arg}" "${submit_dir}" "${script_dir}" "${repo_root}")"; then
  echo "missing input file: ${input_arg}" >&2
  exit 1
fi
extra_args=("${@:2}")
job_id="${SLURM_JOB_ID:-manual}"
sample_stem="$(basename "${input_file%.xmi}")"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
dump_trace="${DUMP_TRACE:-0}"
thread_count="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
log_dir="${LOG_DIR:-$submit_dir}"
stdout_log="${log_dir}/${sample_stem}.${job_id}.stdout.log"
stderr_log="${log_dir}/${sample_stem}.${job_id}.stderr.log"
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

if [[ ! -x "${binary_path}" ]]; then
  echo "missing executable: ${binary_path}" >&2
  echo "try: cd ${repo_root} && ./build.sh build" >&2
  exit 1
fi

prepare_xmvb_cpp_runtime_env_local "${binary_path}" "${repo_root}"
prepare_xmvb_cpp_thread_env_local "${thread_count}"

launch_command=()
slurm_exclusive="false"
launch_cpu_bind="direct"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Run the OpenMP job as a single Slurm step with explicit core binding so
  # wall-time comparisons are less sensitive to step placement and co-scheduled
  # noise on shared cores.
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
echo "hostname = $(hostname)"
echo "slurm_job_id = ${SLURM_JOB_ID:-<unset>}"
echo "slurm_nodelist = ${SLURM_JOB_NODELIST:-<unset>}"
echo "slurm_exclusive = ${slurm_exclusive}"
echo "launch_cpu_bind = ${launch_cpu_bind}"
log_xmvb_cpp_thread_env_local
echo "dump_trace = ${dump_trace_enabled}"
echo "ld_library_path = ${LD_LIBRARY_PATH:-<unset>}"
echo "stdout_log = ${stdout_log}"
echo "stderr_log = ${stderr_log}"
if (( dump_trace_enabled != 0 )); then
  echo "dataset_root = ${dataset_root}"
fi
if (( ${#extra_args[@]} > 0 )); then
  echo "extra_args = ${extra_args[*]}"
fi

cd "${repo_root}"
"${launch_command[@]}" "${command[@]}" >"${stdout_log}" 2>"${stderr_log}"

echo "completed = true"
echo "stdout_log = ${stdout_log}"
echo "stderr_log = ${stderr_log}"
if (( dump_trace_enabled != 0 )); then
  echo "dataset_root = ${dataset_root}"
fi
