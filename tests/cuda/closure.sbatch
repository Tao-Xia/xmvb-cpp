#!/usr/bin/env bash
#SBATCH --job-name=cuvb-closure
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=cuvb-closure-%j.out
#SBATCH --error=cuvb-closure-%j.err

set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9
set -u
source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
out="$source_dir/cuvb-closure-$SLURM_JOB_ID"
mkdir -p "$out"
export LC_ALL=C OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export OMP_MAX_ACTIVE_LEVELS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib:${LD_LIBRARY_PATH:-}"
hostname > "$out/node.txt"
lscpu > "$out/cpu.txt"
nvidia-smi -q > "$out/gpu.txt"
module list > "$out/modules.txt" 2>&1
find "$source_dir/src/cuda" "$source_dir/src/mcscf/optimization/ciah" \
  "$source_dir/src/input/loading" "$source_dir/src/integrals/ao" "$source_dir/src/libcint" \
  "$source_dir/tests/cuda" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$out/sources.sha256"
targets=(test_cuda_structure_action test_cuda_production_ci test_cuda_exact \
         test_cuda_orbitals test_cuda_ciah test_cuda_point test_cuda_tangent test_cuda_chart
         test_cuda_device_davidson test_cuda_ci_refresh test_cuda_transport test_cuda_coarse
         test_cuda_diagonal test_cuda_macro test_cuda_ci_ah_inverse test_cuda_selected_metric
         test_davidson test_ciah test_neo test_neo_globalization)
cmake -S "$source_dir" -B "$source_dir/build-cuvbscf" \
  > "$out/configure.log" 2>&1
cmake --build "$source_dir/build-cuvbscf" --target "${targets[@]}" cuvbscf -j 16 \
  > "$out/build.log" 2>&1
for test in "${targets[@]}"; do
  sha256sum "$source_dir/build-cuvbscf/src/$test" >> "$out/binaries.sha256"
  srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    /usr/bin/time -v -o "$out/$test.time" \
    "$source_dir/build-cuvbscf/src/$test" > "$out/$test.out" 2> "$out/$test.err"
done
