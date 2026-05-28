#!/usr/bin/env bash
#SBATCH -o %x-%j.slog
#SBATCH -e %x-%j.serr
#SBATCH -p 6226r
#SBATCH -A weiwu
#SBATCH --mem=60GB
set -euo pipefail
repo_root="/pool1/home/xiatao/project/xmvb-cpp"
exec "${repo_root}/xmvb.sh" "$@"
