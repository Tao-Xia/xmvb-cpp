#!/usr/bin/env bash

# Build a deterministic runtime library search path for the XMVB C++ binaries.
# The compiled executables carry the correct RUNPATH, but user-submitted jobs can
# inherit older Anaconda libstdc++ entries through LD_LIBRARY_PATH, which then
# override RUNPATH resolution and break startup with missing CXXABI symbols.

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

prepare_xmvb_cpp_runtime_env() {
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

prepare_xmvb_cpp_thread_env() {
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

log_xmvb_cpp_thread_env() {
  echo "omp_threads = ${OMP_NUM_THREADS:-<unset>}"
  echo "openblas_threads = ${OPENBLAS_NUM_THREADS:-<unset>}"
  echo "goto_threads = ${GOTO_NUM_THREADS:-<unset>}"
  echo "mkl_threads = ${MKL_NUM_THREADS:-<unset>}"
  echo "mkl_dynamic = ${MKL_DYNAMIC:-<unset>}"
  echo "omp_proc_bind = ${OMP_PROC_BIND:-<unset>}"
  echo "omp_places = ${OMP_PLACES:-<unset>}"
  echo "omp_stacksize = ${OMP_STACKSIZE:-<unset>}"
}
