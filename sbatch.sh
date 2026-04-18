#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_input="${repo_root}/data/training_xmi/6e6o_full/10698_VBSCF.xmi"

if [[ $# -gt 1 ]]; then
  echo "usage: bash sbatch.sh [input.xmi]" >&2
  exit 1
fi

input_file="${INPUT_FILE:-${1:-${default_input}}}"
job_script="${JOB_SCRIPT:-${repo_root}/scripts/xt-xmvb.sh}"
dataset_root="${DATASET_ROOT:-${repo_root}/data/matrix_data}"
log_dir="${LOG_DIR:-${dataset_root}/logs}"
partition="${PARTITION:-6226r}"
account="${ACCOUNT:-weiwu}"
time_limit="${TIME_LIMIT:-24:00:00}"
cpus_per_task="${CPUS_PER_TASK:-32}"
omp_threads="${OMP_NUM_THREADS:-${cpus_per_task}}"
xmvb_cpp_num_threads="${XMVB_CPP_NUM_THREADS:-${omp_threads}}"
mem="${MEM:-}"
force_rerun="${FORCE_RERUN:-1}"
runtime_progress="${XMVB_CPP_LOG_RUNTIME_PROGRESS:-1}"
objective_progress="${XMVB_CPP_LOG_OBJECTIVE_PROGRESS:-1}"
dry_run="${DRY_RUN:-0}"

PARTITION="${partition}" \
ACCOUNT="${account}" \
TIME_LIMIT="${time_limit}" \
CPUS_PER_TASK="${cpus_per_task}" \
OMP_NUM_THREADS="${omp_threads}" \
XMVB_CPP_NUM_THREADS="${xmvb_cpp_num_threads}" \
DATASET_ROOT="${dataset_root}" \
LOG_DIR="${log_dir}" \
JOB_SCRIPT="${job_script}" \
FORCE_RERUN="${force_rerun}" \
XMVB_CPP_LOG_RUNTIME_PROGRESS="${runtime_progress}" \
XMVB_CPP_LOG_OBJECTIVE_PROGRESS="${objective_progress}" \
MEM="${mem}" \
DRY_RUN="${dry_run}" \
bash "${repo_root}/scripts/submit_single_trace_slurm.sh" "${input_file}"
