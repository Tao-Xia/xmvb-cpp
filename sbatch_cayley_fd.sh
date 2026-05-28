#!/bin/bash
#SBATCH --job-name=cayley-fd
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cayley_fd_%j.out

export OMP_NUM_THREADS=4
export XMVB_CPP_USE_CAYLEY_RETRACTION=1

EXE=/pool1/home/xiatao/project/xmvb-cpp/build/src/check_exact_ctx_hvp
TEST=/pool1/home/xiatao/project/xmvb-cpp/test

echo "=== Cayley HVP FD: F2.xmi ==="
$EXE $TEST/F2.xmi

echo ""
echo "=== Cayley HVP FD: 241_iscf6.xmi ==="
$EXE $TEST/241_iscf6.xmi

echo ""
echo "=== Cayley HVP FD: MnF2_tnhvp.xmi ==="
$EXE $TEST/MnF2_tnhvp.xmi
