#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB_SCRIPT="${JOB_SCRIPT:-${repo_root}/scripts/xt-xmvb.sh}" \
INPUT_DIR="${INPUT_DIR:-${repo_root}/data/training_xmi/6e6o_full}" \
MANIFEST_FILE="${MANIFEST_FILE:-${repo_root}/data/training_xmi/6e6o_full/manifest.txt}" \
DATASET_ROOT="${DATASET_ROOT:-${repo_root}/data/matrix_data}" \
PARTITION="${PARTITION:-6226r}" \
ACCOUNT="${ACCOUNT:-weiwu}" \
TIME_LIMIT="${TIME_LIMIT:-24:00:00}" \
CPUS_PER_TASK="${CPUS_PER_TASK:-32}" \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" \
XMVB_CPP_NUM_THREADS="${XMVB_CPP_NUM_THREADS:-32}" \
MAX_RUNNING="${MAX_RUNNING:-4}" \
"${repo_root}/scripts/submit_trace_array_slurm.sh"
