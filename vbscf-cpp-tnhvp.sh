#!/usr/bin/env bash
#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=60GB
#SBATCH --exclusive
set -euo pipefail
repo_root="/pool1/home/xiatao/project/xmvb-cpp"
export OPTIMIZER_BACKEND="${OPTIMIZER_BACKEND:-nonredundant_truncated_newton}"
exec "${repo_root}/vbscf-cpp.sh" "$@"
