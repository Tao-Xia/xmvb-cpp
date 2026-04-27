#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
submit_triplet_script="${repo_root}/scripts/submit_vb_optimizer_benchmark_triplet.sh"

partition="${PARTITION:-6526Y}"
account="${ACCOUNT:-weiwu}"
qos="${QOS:-}"
cpus_per_task="${CPUS_PER_TASK:-32}"
omp_threads="${OMP_NUM_THREADS:-${cpus_per_task}}"
time_limit="${TIME_LIMIT:-24:00:00}"
mem="${MEM:-}"
dump_trace="${DUMP_TRACE:-0}"
dry_run="${DRY_RUN:-0}"
job_prefix="${JOB_PREFIX:-thesis}"
legacy_xmvb_version="${LEGACY_XMVB_VERSION:-latest}"
tnhvp_hvp_mode="${TNHVP_HVP_MODE:-exact_ctx}"
tnhvp_max_cg_iterations="${TNHVP_MAX_CG_ITERATIONS:-0}"
tnhvp_disable_outer_response="${TNHVP_DISABLE_OUTER_RESPONSE:-0}"
orbital_guess_source="${ORBITAL_GUESS_SOURCE:-}"
timestamp="$(date +%Y%m%d_%H%M%S)"
suite_root="${SUITE_ROOT:-${repo_root}/benchmarks/thesis_optimizer_suite_${timestamp}}"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/submit_thesis_optimizer_suite.sh [system ...] [-- xmvb-cpp args...]

Submits the thesis comparison suite. Each selected system launches:
  1. cpp_tnhvp
  2. cpp_lbfgs
  3. legacy_xmvb via xmvb.sh

Default systems:
  F2 benzene MnF2 FeCl2 10698 240

Accepted system aliases:
  F2
  benzene, C6H6, 241, 241_VBSCF
  MnF2
  FeCl2
  10698, 10698_VBSCF
  240, 240_tnhvp

Environment overrides:
  SUITE_ROOT                   Default: <repo>/benchmarks/thesis_optimizer_suite_<timestamp>
  PARTITION                    Default: 6526Y
  ACCOUNT                      Default: weiwu
  QOS                          Default: empty
  CPUS_PER_TASK                Default: 32
  OMP_NUM_THREADS              Default: CPUS_PER_TASK
  TIME_LIMIT                   Default: 24:00:00
  MEM                          Default: unset
  DUMP_TRACE                   Default: 0
  DRY_RUN                      Default: 0
  JOB_PREFIX                   Default: thesis
  LEGACY_XMVB_VERSION          Default: latest
  TNHVP_HVP_MODE               Default: exact_ctx
  TNHVP_MAX_CG_ITERATIONS      Default: 0
  TNHVP_DISABLE_OUTER_RESPONSE Default: 0
  ORBITAL_GUESS_SOURCE         Default: launcher default

Examples:
  bash scripts/submit_thesis_optimizer_suite.sh
  bash scripts/submit_thesis_optimizer_suite.sh F2 benzene 240
  bash scripts/submit_thesis_optimizer_suite.sh -- --max-iterations 80
  DRY_RUN=1 bash scripts/submit_thesis_optimizer_suite.sh MnF2 FeCl2
EOF
}

list_systems() {
  cat <<'EOF'
F2           -> test/F2.xmi
benzene      -> test/241_VBSCF.xmi
MnF2         -> test/MnF2.xmi
FeCl2        -> test/FeCl2.xmi
10698        -> test/10698_VBSCF.xmi
240          -> test/240_tnhvp.xmi
EOF
}

canonicalize_system_tag() {
  local raw_tag="${1:-}"
  case "${raw_tag}" in
    F2|f2)
      printf 'F2\n'
      ;;
    benzene|Benzene|c6h6|C6H6|241|241_VBSCF)
      printf 'benzene\n'
      ;;
    MnF2|mnf2)
      printf 'MnF2\n'
      ;;
    FeCl2|fecl2)
      printf 'FeCl2\n'
      ;;
    10698|10698_VBSCF)
      printf '10698\n'
      ;;
    240|240_tnhvp)
      printf '240\n'
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_system_input() {
  local canonical_tag="$1"
  case "${canonical_tag}" in
    F2)
      printf '%s\n' "${repo_root}/test/F2.xmi"
      ;;
    benzene)
      printf '%s\n' "${repo_root}/test/241_VBSCF.xmi"
      ;;
    MnF2)
      printf '%s\n' "${repo_root}/test/MnF2.xmi"
      ;;
    FeCl2)
      printf '%s\n' "${repo_root}/test/FeCl2.xmi"
      ;;
    10698)
      printf '%s\n' "${repo_root}/test/10698_VBSCF.xmi"
      ;;
    240)
      printf '%s\n' "${repo_root}/test/240_tnhvp.xmi"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ ! -x "${submit_triplet_script}" ]]; then
  echo "missing triplet submit script: ${submit_triplet_script}" >&2
  exit 1
fi

system_args=()
cpp_extra_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --list)
      list_systems
      exit 0
      ;;
    --root)
      if [[ $# -lt 2 ]]; then
        echo "--root requires a path argument" >&2
        exit 1
      fi
      suite_root="$2"
      shift 2
      ;;
    --)
      shift
      cpp_extra_args=("$@")
      break
      ;;
    *)
      system_args+=("$1")
      shift
      ;;
  esac
done

if ! [[ "${cpus_per_task}" =~ ^[0-9]+$ ]] || (( cpus_per_task < 1 )); then
  echo "CPUS_PER_TASK must be a positive integer" >&2
  exit 1
fi
if ! [[ "${omp_threads}" =~ ^[0-9]+$ ]] || (( omp_threads < 1 )); then
  echo "OMP_NUM_THREADS must be a positive integer" >&2
  exit 1
fi

if (( ${#system_args[@]} == 0 )); then
  system_args=(F2 benzene MnF2 FeCl2 10698 240)
fi

mkdir -p "${suite_root}"
submit_log="${suite_root}/submit_all.log"
systems_tsv="${suite_root}/systems.tsv"
printf 'system_tag\tinput_file\tbenchmark_dir\n' > "${systems_tsv}"

{
  echo "repo_root = ${repo_root}"
  echo "suite_root = ${suite_root}"
  echo "partition = ${partition}"
  echo "account = ${account:-<unset>}"
  echo "qos = ${qos:-<unset>}"
  echo "cpus_per_task = ${cpus_per_task}"
  echo "omp_threads = ${omp_threads}"
  echo "time_limit = ${time_limit}"
  echo "mem = ${mem:-<unset>}"
  echo "dump_trace = ${dump_trace}"
  echo "dry_run = ${dry_run}"
  echo "job_prefix = ${job_prefix}"
  echo "legacy_xmvb_version = ${legacy_xmvb_version}"
  echo "tnhvp_hvp_mode = ${tnhvp_hvp_mode}"
  echo "tnhvp_max_cg_iterations = ${tnhvp_max_cg_iterations}"
  echo "tnhvp_disable_outer_response = ${tnhvp_disable_outer_response}"
  echo "orbital_guess_source = ${orbital_guess_source:-<unset>}"
  if (( ${#cpp_extra_args[@]} > 0 )); then
    echo "cpp_extra_args = ${cpp_extra_args[*]}"
  else
    echo "cpp_extra_args = <none>"
  fi
  echo "systems_tsv = ${systems_tsv}"
} | tee -a "${submit_log}"

declare -A seen_systems=()

submit_one_system() {
  local user_tag="$1"
  local system_tag=""
  local input_file=""
  local benchmark_dir=""

  if ! system_tag="$(canonicalize_system_tag "${user_tag}")"; then
    echo "unknown system tag: ${user_tag}" >&2
    exit 1
  fi
  if [[ -n "${seen_systems[${system_tag}]:-}" ]]; then
    echo "skip duplicate system tag: ${user_tag} -> ${system_tag}" | tee -a "${submit_log}"
    return
  fi
  seen_systems["${system_tag}"]=1

  input_file="$(resolve_system_input "${system_tag}")"
  if [[ ! -f "${input_file}" ]]; then
    echo "missing input file for ${system_tag}: ${input_file}" >&2
    exit 1
  fi
  benchmark_dir="${suite_root}/${system_tag}"

  {
    echo
    echo "[submit] system_tag = ${system_tag}"
    echo "input_file = ${input_file}"
    echo "benchmark_dir = ${benchmark_dir}"
  } | tee -a "${submit_log}"

  env \
    PARTITION="${partition}" \
    ACCOUNT="${account}" \
    QOS="${qos}" \
    CPUS_PER_TASK="${cpus_per_task}" \
    OMP_NUM_THREADS="${omp_threads}" \
    TIME_LIMIT="${time_limit}" \
    MEM="${mem}" \
    DUMP_TRACE="${dump_trace}" \
    DRY_RUN="${dry_run}" \
    JOB_PREFIX="${job_prefix}_${system_tag}" \
    LEGACY_XMVB_VERSION="${legacy_xmvb_version}" \
    TNHVP_HVP_MODE="${tnhvp_hvp_mode}" \
    TNHVP_MAX_CG_ITERATIONS="${tnhvp_max_cg_iterations}" \
    TNHVP_DISABLE_OUTER_RESPONSE="${tnhvp_disable_outer_response}" \
    ORBITAL_GUESS_SOURCE="${orbital_guess_source}" \
    BENCHMARK_DIR="${benchmark_dir}" \
    bash "${submit_triplet_script}" "${input_file}" "${cpp_extra_args[@]}"

  printf '%s\t%s\t%s\n' "${system_tag}" "${input_file}" "${benchmark_dir}" >> "${systems_tsv}"
}

for system_arg in "${system_args[@]}"; do
  submit_one_system "${system_arg}"
done

{
  echo
  echo "completed = true"
  echo "suite_root = ${suite_root}"
  echo "systems_tsv = ${systems_tsv}"
  echo "hint = bash ${repo_root}/scripts/summarize_thesis_optimizer_suite.sh ${suite_root}"
} | tee -a "${submit_log}"
