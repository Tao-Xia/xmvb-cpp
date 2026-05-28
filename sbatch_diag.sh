#!/bin/bash
#SBATCH --job-name=cayley-diag
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cayley_diag_%j.out

export OMP_NUM_THREADS=4
export XMVB_CPP_USE_CAYLEY_RETRACTION=1

EXE=/pool1/home/xiatao/project/xmvb-cpp/build/src/xmvb-cpp.exe

echo "=== MnF2 Cayley ON (10 iters diagnostic) ==="
$EXE /pool1/home/xiatao/project/xmvb-cpp/test/MnF2_tnhvp.xmi --optimizer-backend nonredundant_truncated_newton --nonredundant-truncated-newton-hvp-mode exact_ctx --max-iterations 10
