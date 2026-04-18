#!/usr/bin/env bash

#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr

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

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch $0 <input.xmi>" >&2
  exit 1
fi

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
input_arg="$1"
repo_root="/pool1/home/xiatao/project/xmvb-cpp"
if ! input_file="$(xmvb_cpp_resolve_input_file_local "${input_arg}" "${submit_dir}" "${script_dir}" "${repo_root}")"; then
  echo "missing input file: ${input_arg}" >&2
  exit 1
fi

binary_path="/pool1/home/xiatao/project/xmvb-cpp/build/src/xmvb-cpp.exe"
data_root="${DATA_ROOT:-${repo_root}/data}"
dataset_root="${DATASET_ROOT:-${data_root}/traces}"
log_dir="${LOG_DIR:-${dataset_root}/logs}"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
algorithm="${ALGORITHM:-original}"
onnx_model="${ONNX_MODEL:-}"
force_rerun="${FORCE_RERUN:-0}"
dump_trace="${DUMP_TRACE:-1}"
scratch_root="${SCRATCH_ROOT:-/job_dir}"
job_id="${SLURM_JOB_ID:-manual}"
job_name="${SLURM_JOB_NAME:-xmvb_cpp}"
cpus_per_task="${SLURM_CPUS_PER_TASK:-1}"
omp_threads="${OMP_NUM_THREADS:-${cpus_per_task}}"
sample_stem="$(basename "${input_file%.xmi}")"
sample_dir="${dataset_root}/${sample_stem}"
stdout_log="${log_dir}/${sample_stem}.${job_id}.stdout.log"
stderr_log="${log_dir}/${sample_stem}.${job_id}.stderr.log"
run_dir="${scratch_root}/${job_name}_${job_id}"
ml_work_dir="${ML_WORK_DIR:-${run_dir}/ml_work}"
ml_keep_work_dir="${ML_KEEP_WORK_DIR:-false}"
ml_initial_step_scale="${ML_INITIAL_STEP_SCALE:-}"
ml_minimum_step_scale="${ML_MINIMUM_STEP_SCALE:-}"
ml_step_shrink_factor="${ML_STEP_SHRINK_FACTOR:-}"
ml_max_backtracks="${ML_MAX_BACKTRACKS:-}"
ml_fallback_max_iterations="${ML_FALLBACK_MAX_ITERATIONS:-}"

dump_trace_enabled=1
if [[ -z "${dump_trace}" || "${dump_trace}" == "0" ||
      "${dump_trace}" == "false" || "${dump_trace}" == "FALSE" ]]; then
  dump_trace_enabled=0
fi

if [[ ! -d "${repo_root}" ]]; then
  echo "missing repo_root: ${repo_root}" >&2
  exit 1
fi

if [[ ! -x "${binary_path}" ]]; then
  echo "missing executable: ${binary_path}" >&2
  echo "try: cd ${repo_root} && ./build.sh build" >&2
  exit 1
fi

prepare_xmvb_cpp_runtime_env_local "${binary_path}" "${repo_root}"
prepare_xmvb_cpp_thread_env_local "${omp_threads}"

if [[ ! -d "${data_root}" ]]; then
  echo "missing data_root: ${data_root}" >&2
  exit 1
fi

if [[ "${optimizer_backend}" == "deepvbh_onnx" ||
      "${optimizer_backend}" == "deepvbh_onnx_direct_final" ]]; then
  if [[ -z "${onnx_model}" ]]; then
    echo "ONNX_MODEL is required for optimizer backend ${optimizer_backend}" >&2
    exit 1
  fi
fi

if [[ "${algorithm}" != "original" ]]; then
  echo "xt-xmvb.sh currently supports only ALGORITHM=original" >&2
  exit 1
fi

mkdir -p "${log_dir}" "${run_dir}"
if (( dump_trace_enabled != 0 )); then
  mkdir -p "${dataset_root}"
fi
chmod 700 "${run_dir}"

if (( dump_trace_enabled != 0 )) && [[ -f "${sample_dir}/metadata.json" ]]; then
  if grep -q '"status": "completed"' "${sample_dir}/metadata.json"; then
    if [[ "${force_rerun}" == "0" ]]; then
      echo "skip completed: ${sample_stem}"
      exit 0
    fi
    rm -rf "${sample_dir}"
  else
    if [[ "${force_rerun}" == "0" ]]; then
      echo "incomplete existing sample directory: ${sample_dir}" >&2
      exit 1
    fi
    rm -rf "${sample_dir}"
  fi
fi

ulimit -c 0
ulimit -s unlimited
ulimit -v unlimited

command=(
  "${binary_path}"
  "${input_file}"
  "--optimizer-backend" "${optimizer_backend}"
  "--algorithm" "${algorithm}"
)

if (( dump_trace_enabled != 0 )); then
  command+=("--dump-trace-dir" "${dataset_root}")
fi

if [[ -n "${onnx_model}" ]]; then
  command+=("--onnx-model" "${onnx_model}")
  command+=("--ml-work-dir" "${ml_work_dir}")
  command+=("--ml-keep-work-dir" "${ml_keep_work_dir}")
fi

if [[ -n "${ml_initial_step_scale}" ]]; then
  command+=("--ml-initial-step-scale" "${ml_initial_step_scale}")
fi
if [[ -n "${ml_minimum_step_scale}" ]]; then
  command+=("--ml-minimum-step-scale" "${ml_minimum_step_scale}")
fi
if [[ -n "${ml_step_shrink_factor}" ]]; then
  command+=("--ml-step-shrink-factor" "${ml_step_shrink_factor}")
fi
if [[ -n "${ml_max_backtracks}" ]]; then
  command+=("--ml-max-backtracks" "${ml_max_backtracks}")
fi
if [[ -n "${ml_fallback_max_iterations}" ]]; then
  command+=("--ml-fallback-max-iterations" "${ml_fallback_max_iterations}")
fi

echo "repo_root = ${repo_root}"
echo "binary = ${binary_path}"
echo "input_file = ${input_file}"
echo "dataset_root = ${dataset_root}"
echo "optimizer_backend = ${optimizer_backend}"
echo "algorithm = ${algorithm}"
echo "hostname = $(hostname)"
echo "slurm_job_id = ${SLURM_JOB_ID:-<unset>}"
echo "slurm_nodelist = ${SLURM_JOB_NODELIST:-<unset>}"
log_xmvb_cpp_thread_env_local
echo "dump_trace = ${dump_trace_enabled}"
echo "ld_library_path = ${LD_LIBRARY_PATH:-<unset>}"
if [[ -n "${onnx_model}" ]]; then
  echo "onnx_model = ${onnx_model}"
  echo "ml_work_dir = ${ml_work_dir}"
fi
echo "stdout_log = ${stdout_log}"
echo "stderr_log = ${stderr_log}"

cd "${repo_root}"
"${command[@]}" >"${stdout_log}" 2>"${stderr_log}"

touch "${run_dir}/_OK"
if (( dump_trace_enabled != 0 )); then
  actual_sample_dir="$(sed -n 's/^trace_sample_dir = //p' "${stdout_log}" | tail -n 1)"
  if [[ -z "${actual_sample_dir}" ]]; then
    actual_sample_dir="${sample_dir}"
  fi

  if [[ ! -d "${actual_sample_dir}" ]]; then
    echo "missing trace output directory: ${actual_sample_dir}" >&2
    exit 1
  fi

  cp "${input_file}" "${actual_sample_dir}/"
  echo "sample_dir = ${actual_sample_dir}"
else
  echo "sample_dir = <disabled>"
fi
