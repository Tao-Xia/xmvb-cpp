#!/usr/bin/env bash
#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=60GB
#SBATCH --exclusive
set -euo pipefail
repo_root="/pool1/home/xiatao/project/xmvb-cpp"
export XMVB_CPP_ENABLE_DAVIDSON_EIGENSOLVER=1
export OPTIMIZER_BACKEND="nonredundant_truncated_newton"
export LOG_DIR="${repo_root}/test"
exec "${repo_root}/vbscf-cpp.sh" "$@"
