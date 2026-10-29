#!/bin/bash
#SBATCH --job-name=cpp-tnvb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=cpp-tnvb-%j.slurm.log

set -euo pipefail
tnvb_input=${1:?Usage: sbatch submit.slurm input.xmi /installed/bin/xmvb-cpp.exe}
tnvb_executable=${2:?Provide the independently installed xmvb-cpp.exe}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
srun "$tnvb_executable" "$tnvb_input" > "${tnvb_input%.xmi}.xmo" 2>&1
