#!/usr/bin/env bash
#SBATCH --job-name=cuvb-ci
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=cuvb-ci-%j.out
#SBATCH --error=cuvb-ci-%j.err

set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9
set -u
source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
out="$source_dir/cuvb-ci-$SLURM_JOB_ID"
mode=${CUVB_MODE:---gpu}
if [[ "$mode" != --gpu && "$mode" != --point ]]; then
  echo "CUVB_MODE must be --gpu or --point" >&2
  exit 64
fi
mkdir -p "$out"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib:${LD_LIBRARY_PATH:-}"
hostname > "$out/node.txt"
lscpu > "$out/cpu.txt"
nvidia-smi -q > "$out/gpu.txt"
module list > "$out/modules.txt" 2>&1
find "$source_dir/src/cuda" "$source_dir/tests/cuda" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$out/sources.sha256"
cmake -S "$source_dir" -B "$source_dir/build-cuvbscf" > "$out/configure.log" 2>&1
cmake --build "$source_dir/build-cuvbscf" --target \
  c12_cuda_benchmark test_cuda_structure_action test_cuda_production_ci -j 16 \
  > "$out/build.log" 2>&1
cp "$source_dir/build-cuvbscf/src/c12_cuda_benchmark" "$out/"
sha256sum "$out/c12_cuda_benchmark" > "$out/binary.sha256"
for test in test_cuda_structure_action test_cuda_production_ci; do
  srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    "$source_dir/build-cuvbscf/src/$test" > "$out/$test.out" 2> "$out/$test.err"
done
bundle="$source_dir/cuvbscf-c12-297979/bundle/cases"
for case in mixed20 cov ion02 full; do
  mkdir -p "$out/$case"
  cp "$bundle/$case/input.xmi" "$bundle/$case/cc-pvdz.gbs" "$out/$case/"
  (
    cd "$out/$case"
    sha256sum input.xmi cc-pvdz.gbs > input.sha256
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
      --format=csv,noheader,nounits -lms 100 > gpu_samples.csv &
    sampler=$!
    trap 'kill "$sampler" 2>/dev/null || true' EXIT
    srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
      /usr/bin/time -v -o gpu.time "$out/c12_cuda_benchmark" "$mode" input.xmi \
      > gpu.out 2> gpu.err
  )
done
