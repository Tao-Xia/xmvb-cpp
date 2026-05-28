#!/bin/bash
#SBATCH --job-name=cayley-tnhvp
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cayley_tnhvp_%j.out

export OMP_NUM_THREADS=4
export XMVB_CPP_USE_CAYLEY_RETRACTION=1

EXE=/pool1/home/xiatao/project/xmvb-cpp/build/src/xmvb-cpp.exe
TEST=/pool1/home/xiatao/project/xmvb-cpp/test

echo "=== TNHVP Cayley ON: F2.xmi ==="
$EXE $TEST/F2.xmi --optimizer-backend nonredundant_truncated_newton --hvp-mode exact_ctx --max-iterations 50

echo ""
echo "=== TNHVP Cayley OFF: F2.xmi ==="
unset XMVB_CPP_USE_CAYLEY_RETRACTION
$EXE $TEST/F2.xmi --optimizer-backend nonredundant_truncated_newton --hvp-mode exact_ctx --max-iterations 50

echo ""
echo "=== TNHVP Cayley ON: MnF2_tnhvp.xmi ==="
export XMVB_CPP_USE_CAYLEY_RETRACTION=1
$EXE $TEST/MnF2_tnhvp.xmi --optimizer-backend nonredundant_truncated_newton --hvp-mode exact_ctx --max-iterations 500

echo ""
echo "=== TNHVP Cayley OFF: MnF2_tnhvp.xmi ==="
unset XMVB_CPP_USE_CAYLEY_RETRACTION
$EXE $TEST/MnF2_tnhvp.xmi --optimizer-backend nonredundant_truncated_newton --hvp-mode exact_ctx --max-iterations 500
