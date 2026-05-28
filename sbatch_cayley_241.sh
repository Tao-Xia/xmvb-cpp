#!/bin/bash
#SBATCH --job-name=cayley-241
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cayley_241_%j.out

export OMP_NUM_THREADS=4
export XMVB_CPP_USE_CAYLEY_RETRACTION=1

EXE=/pool1/home/xiatao/project/xmvb-cpp/build/src/xmvb-cpp.exe
TEST=/pool1/home/xiatao/project/xmvb-cpp/test

echo "=== TNHVP Cayley ON: 241_iscf6.xmi ==="
$EXE $TEST/241_iscf6.xmi --optimizer-backend nonredundant_truncated_newton --nonredundant-truncated-newton-hvp-mode exact_ctx --max-iterations 200

echo ""
echo "=== TNHVP Cayley OFF: 241_iscf6.xmi ==="
unset XMVB_CPP_USE_CAYLEY_RETRACTION
$EXE $TEST/241_iscf6.xmi --optimizer-backend nonredundant_truncated_newton --nonredundant-truncated-newton-hvp-mode exact_ctx --max-iterations 200
