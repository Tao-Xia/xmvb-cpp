#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
build_type="${CMAKE_BUILD_TYPE:-Release}"
jobs="${XMVB_BUILD_JOBS:-4}"
extra_args=("${@:2}")
repo_root="$(pwd -P)"
build_profile="${XMVB_CPP_BUILD_PROFILE:-auto}"

parse_cache_value() {
  local cache_file="$1"
  local key="$2"
  sed -n "s#^${key}:[^=]*=##p" "${cache_file}" | head -n 1
}

has_cmake_arg_key() {
  local key="$1"
  local arg=""
  for arg in "${extra_args[@]}"; do
    case "${arg}" in
      -D"${key}"=*|-D"${key}":*=*)
        return 0
        ;;
    esac
  done
  return 1
}

append_default_cmake_arg() {
  local key="$1"
  local value="$2"
  if ! has_cmake_arg_key "${key}"; then
    extra_args+=("-D${key}=${value}")
  fi
}

append_candidate_prefix_from_arg() {
  local arg="$1"
  local prefix=""
  case "${arg}" in
    -DLAPACK_ROOT_DIR=*)
      prefix="${arg#*=}"
      ;;
    -DLAPACK_ROOT_DIR:PATH=*)
      prefix="${arg#*=}"
      ;;
    -DLIBXC_ROOT_DIR=*)
      prefix="${arg#*=}"
      ;;
    -DLIBXC_ROOT_DIR:PATH=*)
      prefix="${arg#*=}"
      ;;
    -DCINT_ROOT_DIR=*)
      prefix="${arg#*=}"
      ;;
    -DCINT_ROOT_DIR:PATH=*)
      prefix="${arg#*=}"
      ;;
    -DEIGEN3_ROOT_DIR=*)
      prefix="${arg#*=}"
      ;;
    -DEIGEN3_ROOT_DIR:PATH=*)
      prefix="${arg#*=}"
      ;;
  esac
  if [[ -n "${prefix}" ]]; then
    candidate_prefixes+=("${prefix}")
  fi
}

detect_preferred_compilers() {
  local candidate=""
  local cache_file="${build_dir}/CMakeCache.txt"

  candidate_prefixes=()
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    candidate_prefixes+=("${CONDA_PREFIX}")
  fi
  for candidate in "${extra_args[@]}"; do
    append_candidate_prefix_from_arg "${candidate}"
  done
  if [[ -f "${cache_file}" ]]; then
    for candidate in EIGEN3_ROOT_DIR LAPACK_ROOT_DIR LIBXC_ROOT_DIR CINT_ROOT_DIR; do
      value="$(parse_cache_value "${cache_file}" "${candidate}")"
      if [[ -n "${value}" ]]; then
        candidate_prefixes+=("${value}")
      fi
    done
  fi

  for candidate in "${candidate_prefixes[@]}"; do
    if [[ -x "${candidate}/bin/x86_64-conda-linux-gnu-cc" &&
          -x "${candidate}/bin/x86_64-conda-linux-gnu-c++" ]]; then
      detected_cc="${candidate}/bin/x86_64-conda-linux-gnu-cc"
      detected_cxx="${candidate}/bin/x86_64-conda-linux-gnu-c++"
      detected_prefix="${candidate}"
      return 0
    fi
  done

  detected_cc=""
  detected_cxx=""
  detected_prefix=""
  return 1
}

infer_missing_compiler_pair() {
  if [[ -n "${selected_cc}" && -z "${selected_cxx}" ]]; then
    if [[ "${selected_cc}" == */x86_64-conda-linux-gnu-cc ]]; then
      local candidate_cxx="${selected_cc%/x86_64-conda-linux-gnu-cc}/x86_64-conda-linux-gnu-c++"
      if [[ -x "${candidate_cxx}" ]]; then
        selected_cxx="${candidate_cxx}"
      fi
    fi
  fi

  if [[ -n "${selected_cxx}" && -z "${selected_cc}" ]]; then
    if [[ "${selected_cxx}" == */x86_64-conda-linux-gnu-c++ ]]; then
      local candidate_cc="${selected_cxx%/x86_64-conda-linux-gnu-c++}/x86_64-conda-linux-gnu-cc"
      if [[ -x "${candidate_cc}" ]]; then
        selected_cc="${candidate_cc}"
      fi
    fi
  fi
}

detect_clangd_prefix_from_cache() {
  local cache_file="$1"
  local key
  local value
  for key in EIGEN3_ROOT_DIR LAPACK_ROOT_DIR LIBXC_ROOT_DIR CINT_ROOT_DIR; do
    value="$(parse_cache_value "${cache_file}" "${key}")"
    if [[ -n "${value}" && -x "${value}/bin/clangd" ]]; then
      printf '%s\n' "${value}"
      return 0
    fi
  done
  return 1
}

write_vscode_clangd_settings() {
  local workspace_root="$1"
  local build_dir_name="$2"
  local clangd_path="$3"
  local query_driver_glob="$4"

  mkdir -p "${workspace_root}/.vscode"
  cat > "${workspace_root}/.vscode/settings.json" <<EOF
{
  "clangd.path": "${clangd_path}",
  "clangd.arguments": [
    "--background-index",
    "--compile-commands-dir=\${workspaceFolder}/${build_dir_name}",
    "--query-driver=/usr/bin/g++,/usr/bin/c++,${query_driver_glob}",
    "--header-insertion=never"
  ],
  "cmake.copyCompileCommands": "\${workspaceFolder}/compile_commands.json",
  "C_Cpp.intelliSenseEngine": "disabled"
}
EOF
}

detected_cc=""
detected_cxx=""
detected_prefix=""
selected_cc="${CC:-}"
selected_cxx="${CXX:-}"
candidate_prefixes=()
value=""
slater_profile_enabled=""
slater_gentoo_prefix=""
slater_dependency_prefix=""
slater_onnx_root=""

apply_slater_build_profile() {
  case "${build_profile}" in
    auto)
      case "${repo_root}" in
        /export/home/xiatao/project/xmvb-cpp|/pool1/home/xiatao/project/xmvb-cpp)
          ;;
        *)
          return 0
          ;;
      esac
      ;;
    slater)
      ;;
    off|disabled|manual)
      return 0
      ;;
    *)
      return 0
      ;;
  esac

  slater_gentoo_prefix="/export/home/lxyan/gentoo"
  slater_dependency_prefix="/export/home/xiatao/miniconda3/envs/xmvb-dev"
  slater_onnx_root="${repo_root}/third_party/onnxruntime-linux-x64-1.20.1"

  if [[ ! -x "${slater_gentoo_prefix}/usr/bin/gcc" ||
        ! -x "${slater_gentoo_prefix}/usr/bin/g++" ||
        ! -d "${slater_dependency_prefix}" ||
        ! -d "${slater_onnx_root}" ]]; then
    return 0
  fi

  if [[ -z "${selected_cc}" ]]; then
    selected_cc="${slater_gentoo_prefix}/usr/bin/gcc"
  fi
  if [[ -z "${selected_cxx}" ]]; then
    selected_cxx="${slater_gentoo_prefix}/usr/bin/g++"
  fi

  append_default_cmake_arg LAPACK_ROOT_DIR "${slater_dependency_prefix}"
  append_default_cmake_arg LIBXC_ROOT_DIR "${slater_dependency_prefix}"
  append_default_cmake_arg CINT_ROOT_DIR "${slater_dependency_prefix}"
  append_default_cmake_arg EIGEN3_ROOT_DIR "${slater_dependency_prefix}"
  append_default_cmake_arg ONNXRUNTIME_ROOT_DIR "${slater_onnx_root}"
  append_default_cmake_arg XMVB_CPP_ENABLE_ONNX_RUNTIME ON

  slater_profile_enabled="1"
}

apply_slater_build_profile

if [[ -n "${slater_profile_enabled}" ]]; then
  echo "Using slater build profile:"
  echo "  repo_root=${repo_root}"
  echo "  gentoo_prefix=${slater_gentoo_prefix}"
  echo "  dependency_prefix=${slater_dependency_prefix}"
  echo "  onnxruntime_root=${slater_onnx_root}"
fi

if [[ -z "${selected_cc}" || -z "${selected_cxx}" ]]; then
  detect_preferred_compilers || true
  if [[ -z "${selected_cc}" ]]; then
    selected_cc="${detected_cc}"
  fi
  if [[ -z "${selected_cxx}" ]]; then
    selected_cxx="${detected_cxx}"
  fi
fi

infer_missing_compiler_pair

cache_file="${build_dir}/CMakeCache.txt"
if [[ -f "${cache_file}" && -n "${selected_cc}" && -n "${selected_cxx}" ]]; then
  cached_cc="$(parse_cache_value "${cache_file}" CMAKE_C_COMPILER)"
  cached_cxx="$(parse_cache_value "${cache_file}" CMAKE_CXX_COMPILER)"
  if [[ "${cached_cc}" != "${selected_cc}" || "${cached_cxx}" != "${selected_cxx}" ]]; then
    echo "Resetting ${build_dir} CMake cache to switch compilers:"
    echo "  C   ${cached_cc:-<unset>} -> ${selected_cc}"
    echo "  C++ ${cached_cxx:-<unset>} -> ${selected_cxx}"
    rm -rf \
      "${build_dir}/CMakeCache.txt" \
      "${build_dir}/CMakeFiles" \
      "${build_dir}/build.ninja" \
      "${build_dir}/cmake_install.cmake" \
      "${build_dir}/.ninja_deps" \
      "${build_dir}/.ninja_log"
  fi
fi

if [[ -n "${selected_cc}" && -n "${selected_cxx}" ]]; then
  export CC="${selected_cc}"
  export CXX="${selected_cxx}"
  echo "Using compilers:"
  echo "  CC=${CC}"
  echo "  CXX=${CXX}"
fi

cmake_args=(
  -S .
  -B "${build_dir}"
  -G Ninja
  -DCMAKE_BUILD_TYPE="${build_type}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DXMVB_CPP_BUILD_DEV_TARGETS=OFF \
)

if [[ -n "${selected_cc}" && -n "${selected_cxx}" ]]; then
  cmake_args+=(
    -DCMAKE_C_COMPILER="${selected_cc}"
    -DCMAKE_CXX_COMPILER="${selected_cxx}"
  )
fi

cmake_args+=("${extra_args[@]}")

cmake "${cmake_args[@]}"

cmake --build "${build_dir}" --target run_cpp_vbscf -j "${jobs}"

build_dir_abs="$(cd "${build_dir}" && pwd)"
ln -sfn "${build_dir_abs}/compile_commands.json" compile_commands.json

clangd_prefix=""
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/clangd" ]]; then
  clangd_prefix="${CONDA_PREFIX}"
elif [[ -n "${detected_prefix}" && -x "${detected_prefix}/bin/clangd" ]]; then
  clangd_prefix="${detected_prefix}"
elif [[ -f "${cache_file}" ]]; then
  clangd_prefix="$(detect_clangd_prefix_from_cache "${cache_file}" || true)"
fi

if [[ -n "${clangd_prefix}" ]]; then
  write_vscode_clangd_settings \
    "$(pwd)" \
    "${build_dir}" \
    "${clangd_prefix}/bin/clangd" \
    "${clangd_prefix}/bin/*"
fi

echo
echo "Built standalone xmvb-cpp at ${build_dir}/src/xmvb-cpp.exe"
