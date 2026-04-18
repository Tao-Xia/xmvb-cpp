#!/usr/bin/env bash

#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=60GB

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: sbatch -c <threads> pf-vbscf.sh <input.xmi> [pf tool args...]

Current default mode:
  PF_MODE=scf            run build/src/check_pf_scf

Supported PF_MODE values:
  scf                    forward Pfaffian SCF check
  one_step               compare one determinant step vs one Pfaffian step
  active_grad            check Pfaffian active-space gradient
  orbital_grad           check Pfaffian orbital gradient

Optional environment overrides:
  XMVB_CPP_REPO
  XMVB_CPP_KEEP_LD_LIBRARY_PATH=1
  PF_MODE
  PF_BIN
  PF_K
  PF_SEED
  PF_COUNT
  PF_STEP
  PF_COMPONENT
  PF_INIT_STEP
  PF_ARMIJO
  PF_MAX_BACKTRACKS
  LOG_DIR

Examples:
  sbatch -c 1 pf-vbscf.sh sample.xmi
  PF_MODE=one_step sbatch -c 1 pf-vbscf.sh sample.xmi
  PF_MODE=active_grad sbatch -c 1 pf-vbscf.sh sample.xmi --component overlap
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root_helper="${script_dir}/scripts/xmvb_cpp_repo_root.sh"
if [[ ! -f "${repo_root_helper}" ]]; then
  echo "missing repo root helper: ${repo_root_helper}" >&2
  exit 1
fi
source "${repo_root_helper}"

repo_root="$(xmvb_cpp_resolve_repo_root "${submit_dir}" "${BASH_SOURCE[0]}" "$1" || true)"
if [[ -z "${repo_root}" ]]; then
  echo "failed to infer repo_root" >&2
  echo "set XMVB_CPP_REPO explicitly or invoke the launcher from a path inside the repo" >&2
  exit 1
fi
input_file="$(
  xmvb_cpp_resolve_input_file "$1" "${submit_dir}" "${script_dir}" "${repo_root}" ||
      true
)"
extra_args=("${@:2}")
sample_stem="$(basename "${input_file%.xmi}")"
thread_count="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
log_dir="${LOG_DIR:-$submit_dir}"
pf_mode="${PF_MODE:-scf}"

pf_k="${PF_K:-3}"
pf_seed="${PF_SEED:-20260328}"
pf_count="${PF_COUNT:-4}"
pf_step="${PF_STEP:-1e-6}"
pf_component="${PF_COMPONENT:-all}"
pf_init_step="${PF_INIT_STEP:-1.0}"
pf_armijo="${PF_ARMIJO:-1e-4}"
pf_max_backtracks="${PF_MAX_BACKTRACKS:-12}"

case "${pf_mode}" in
  scf)
    default_bin="${repo_root}/build/src/check_pf_scf"
    run_log="${log_dir}/${sample_stem}.pf.log"
    command=(
      "${PF_BIN:-${default_bin}}"
      "${input_file}"
      "--k" "${pf_k}"
      "--seed" "${pf_seed}"
    )
    ;;
  one_step)
    default_bin="${repo_root}/build/src/compare_pf_one_step"
    run_log="${log_dir}/${sample_stem}.pf-step.log"
    command=(
      "${PF_BIN:-${default_bin}}"
      "${input_file}"
      "--k" "${pf_k}"
      "--seed" "${pf_seed}"
      "--init-step" "${pf_init_step}"
      "--armijo" "${pf_armijo}"
      "--max-backtracks" "${pf_max_backtracks}"
    )
    ;;
  active_grad)
    default_bin="${repo_root}/build/src/check_pf_active_grad"
    run_log="${log_dir}/${sample_stem}.pf-active-grad.log"
    command=(
      "${PF_BIN:-${default_bin}}"
      "${input_file}"
      "--component" "${pf_component}"
      "--count" "${pf_count}"
      "--step" "${pf_step}"
    )
    ;;
  orbital_grad)
    default_bin="${repo_root}/build/src/check_pf_orbital_grad"
    run_log="${log_dir}/${sample_stem}.pf-orbital-grad.log"
    command=(
      "${PF_BIN:-${default_bin}}"
      "${input_file}"
      "--count" "${pf_count}"
      "--step" "${pf_step}"
    )
    ;;
  *)
    echo "invalid PF_MODE: ${pf_mode}" >&2
    usage
    exit 1
    ;;
esac

mkdir -p "${log_dir}"
exec >"${run_log}" 2>&1

if [[ ! -f "${input_file}" ]]; then
  echo "missing input file: ${input_file}"
  exit 1
fi

if [[ ! -x "${command[0]}" ]]; then
  echo "missing executable: ${command[0]}"
  echo "try: cd ${repo_root} && ./build.sh build"
  exit 1
fi

runtime_env_helper="${repo_root}/scripts/xmvb_cpp_runtime_env.sh"
if [[ ! -f "${runtime_env_helper}" ]]; then
  echo "missing runtime env helper: ${runtime_env_helper}"
  exit 1
fi
source "${runtime_env_helper}"
prepare_xmvb_cpp_runtime_env "${command[0]}" "${repo_root}"

if (( ${#extra_args[@]} > 0 )); then
  command+=("${extra_args[@]}")
fi

export OMP_NUM_THREADS="${thread_count}"
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_STACKSIZE="${OMP_STACKSIZE:-1G}"

ulimit -c 0
ulimit -s unlimited
ulimit -v unlimited

echo "repo_root = ${repo_root}"
echo "pf_mode = ${pf_mode}"
echo "binary = ${command[0]}"
echo "input_file = ${input_file}"
echo "submit_dir = ${submit_dir}"
echo "omp_threads = ${OMP_NUM_THREADS}"
echo "ld_library_path = ${LD_LIBRARY_PATH:-<unset>}"
echo "log_file = ${run_log}"
if (( ${#extra_args[@]} > 0 )); then
  echo "extra_args = ${extra_args[*]}"
fi

cd "${repo_root}"
"${command[@]}"

echo "completed = true"
echo "log_file = ${run_log}"
