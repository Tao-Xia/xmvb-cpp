#!/usr/bin/env bash
#SBATCH --job-name=cuvbscf-c12
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=cuvbscf-c12-%j.out
#SBATCH --error=cuvbscf-c12-%j.err
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
set -u
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9

source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
[[ -f "$source_dir/testdata/vbscf/csd_LOFLEA_vb.xmi" &&
   -f "$source_dir/basis/cc-pvdz.gbs" && -f "$deps/lib/libcint.so" ]] || exit 66
artifact="$source_dir/cuvbscf-c12-$SLURM_JOB_ID"
mkdir -p "$artifact/bundle" "$artifact/run"
export LC_ALL=C
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_PLACES=cores OMP_PROC_BIND=close OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib:${LD_LIBRARY_PATH:-}"

hostname > "$artifact/node.txt"
lscpu > "$artifact/cpu.txt"
module list > "$artifact/modules.txt" 2>&1
nvidia-smi -L > "$artifact/gpus.txt"
"$(command -v nvcc)" --version > "$artifact/nvcc.txt"
"$(command -v g++)" --version > "$artifact/cxx.txt"
env | sort | grep -E '^(CUDA_|OMP_|OPENBLAS_|MKL_|BLIS_|SLURM_)' \
  > "$artifact/environment.txt"
git -C "$source_dir" rev-parse HEAD > "$artifact/revision.txt" 2>/dev/null || true
git -C "$source_dir" status --short > "$artifact/dirty.txt" 2>/dev/null || true
find "$source_dir/src/cuda" "$source_dir/tests/cuda" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$artifact/cuda_sources.sha256"

build="$source_dir/build-cuvbscf"
cmake -S "$source_dir" -B "$build" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release -DXMVB_BUILD_DEV_TARGETS=ON \
  -DXMVB_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_C_COMPILER="$(command -v gcc)" \
  -DCMAKE_CXX_COMPILER="$(command -v g++)" \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)" \
  -DCINT_ROOT_DIR="$deps" -DEIGEN3_ROOT_DIR="$deps" \
  -DLAPACK_ROOT_DIR="$deps" -DLIBXC_ROOT_DIR="$deps" \
  > "$artifact/configure.log" 2>&1
cmake --build "$build" --target test_cuda_structure_action c12_cuda_benchmark \
  -j 16 > "$artifact/build.log" 2>&1
cp "$build/src/test_cuda_structure_action" "$build/src/c12_cuda_benchmark" "$artifact/"
sha256sum "$artifact/test_cuda_structure_action" "$artifact/c12_cuda_benchmark" \
  "$source_dir/testdata/vbscf/csd_LOFLEA_vb.xmi" "$source_dir/basis/cc-pvdz.gbs" \
  > "$artifact/hashes.sha256"
cp "$source_dir/testdata/vbscf/csd_LOFLEA_vb.xmi" "$artifact/bundle/original.xmi"
cp "$source_dir/basis/cc-pvdz.gbs" "$artifact/bundle/cc-pvdz.gbs"
python3 "$source_dir/tests/mcscf/c12_spaces.py" prepare \
  "$artifact/bundle/original.xmi" "$artifact/bundle" > "$artifact/preparation.json"
cp "$artifact/bundle/cases.json" "$artifact/"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  "$artifact/test_cuda_structure_action" > "$artifact/small.out" 2> "$artifact/small.err"

# Each solve is a separate native process so /usr/bin/time measures its own
# RSS and wall time. The H/S parity probe is outside both solve timings.
for case in mixed3 mixed20 cov ion02 full; do
  case_dir="$artifact/run/$case"
  mkdir -p "$case_dir"
  cp "$artifact/bundle/cases/$case/input.xmi" \
     "$artifact/bundle/cases/$case/cc-pvdz.gbs" "$case_dir/"
  (cd "$case_dir" && sha256sum input.xmi cc-pvdz.gbs > input.sha256)
  (cd "$case_dir" && srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    "$artifact/c12_cuda_benchmark" --check input.xmi > check.out 2> check.err)
  (cd "$case_dir" && srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    /usr/bin/time -v -o cpu.time "$artifact/c12_cuda_benchmark" --cpu input.xmi \
    > cpu.out 2> cpu.err)
  (
    cd "$case_dir"
    if [[ ${XMVB_GPU_SAMPLE:-0} == 1 ]]; then
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
        --format=csv,noheader,nounits -lms 500 > gpu_samples.csv &
      sampler=$!
      trap 'kill "$sampler" 2>/dev/null || true' EXIT
    fi
    srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
      /usr/bin/time -v -o gpu.time "$artifact/c12_cuda_benchmark" --gpu input.xmi \
      > gpu.out 2> gpu.err
  )
done
python3 "$source_dir/tests/cuda/summarize.py" "$artifact/run" \
  > "$artifact/paired.csv"
