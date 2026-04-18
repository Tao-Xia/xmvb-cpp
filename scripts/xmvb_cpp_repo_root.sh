#!/usr/bin/env bash

# Shared path-resolution helpers for XMVB-C++ launcher scripts.
#
# These helpers intentionally walk upward from candidate paths instead of
# assuming the caller submitted from a particular directory. That keeps the
# launchers stable when users invoke `sbatch /abs/path/to/script.sh ...` from
# outside the repository or when the repo is visible through multiple mount
# points such as `/pool1/...` and `/export/home/...`.

xmvb_cpp_is_repo_root() {
  local candidate="${1:-}"
  [[ -n "${candidate}" &&
     -d "${candidate}" &&
     -f "${candidate}/build.sh" &&
     -d "${candidate}/src" &&
     -d "${candidate}/scripts" ]]
}

xmvb_cpp_find_repo_root_from() {
  local start_path="${1:-}"
  local current_dir=""

  if [[ -z "${start_path}" ]]; then
    return 1
  fi
  if [[ ! -e "${start_path}" ]]; then
    return 1
  fi
  if [[ -f "${start_path}" ]]; then
    start_path="$(dirname "${start_path}")"
  fi
  current_dir="$(readlink -f "${start_path}")"

  while true; do
    if xmvb_cpp_is_repo_root "${current_dir}"; then
      printf '%s\n' "${current_dir}"
      return 0
    fi
    if [[ "${current_dir}" == "/" ]]; then
      break
    fi
    current_dir="$(dirname "${current_dir}")"
  done
  return 1
}

xmvb_cpp_resolve_repo_root() {
  local submit_dir="${1:-}"
  local script_path="${2:-}"
  local input_hint="${3:-}"
  local script_dir=""
  local candidate=""
  local resolved=""

  if [[ -n "${script_path}" ]]; then
    script_dir="$(cd "$(dirname "${script_path}")" && pwd -P)"
  fi

  for candidate in \
    "${XMVB_CPP_REPO:-}" \
    "${script_dir}" \
    "${submit_dir}" \
    "${PWD}" \
    "${input_hint}"; do
    if [[ -z "${candidate}" ]]; then
      continue
    fi
    if resolved="$(xmvb_cpp_find_repo_root_from "${candidate}" 2>/dev/null)"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  return 1
}

xmvb_cpp_resolve_input_file() {
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
