#!/usr/bin/env bash
#SBATCH --job-name=cuvb-action
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=cuvb-action-%j.out
#SBATCH --error=cuvb-action-%j.err
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9
set -u
source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
out="$source_dir/cuvb-action-$SLURM_JOB_ID"
mkdir -p "$out"
export OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib:${LD_LIBRARY_PATH:-}"
hostname > "$out/node.txt"
lscpu > "$out/cpu.txt"
nvidia-smi -L > "$out/gpus.txt"
module list > "$out/modules.txt" 2>&1
cmake --build "$source_dir/build-cuvbscf" --target c12_cuda_benchmark test_cuda_structure_action -j 16 > "$out/build.log" 2>&1
cp "$source_dir/build-cuvbscf/src/c12_cuda_benchmark" "$out/"
sha256sum "$out/c12_cuda_benchmark" > "$out/binary.sha256"
find "$source_dir/src/cuda" "$source_dir/tests/cuda" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "$out/sources.sha256"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  "$source_dir/build-cuvbscf/src/test_cuda_structure_action" > "$out/small.out" 2> "$out/small.err"
bundle="$source_dir/cuvbscf-c12-297979/bundle/cases"
for case in mixed20 cov ion02 full; do
  mkdir -p "$out/$case"
  cp "$bundle/$case/input.xmi" "$bundle/$case/cc-pvdz.gbs" "$out/$case/"
  (cd "$out/$case" && sha256sum input.xmi cc-pvdz.gbs > input.sha256)
  if [[ ${XMVB_CPU_PHASE_BASELINE:-0} == 1 ]]; then
    (cd "$out/$case" && srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
      /usr/bin/time -v -o cpu_phases.time "$out/c12_cuda_benchmark" --cpu-action input.xmi \
      > cpu_phases.out 2> cpu_phases.err)
  fi
  if [[ ${XMVB_GPU_CHECK:-0} == 1 ]]; then
    (cd "$out/$case" && srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
      "$out/c12_cuda_benchmark" --check input.xmi > check.out 2> check.err)
  fi
  (cd "$out/$case" && srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    /usr/bin/time -v -o action.time "$out/c12_cuda_benchmark" --action input.xmi \
    > action.out 2> action.err)
done
if command -v nsys >/dev/null; then
  nsys --version > "$out/nsys_version.txt"
  (cd "$out/cov" && srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    nsys profile --sample=none --cpuctxsw=none --trace=cuda,cublas,nvtx \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite=true --output="$out/cov/nsys" \
      "$out/c12_cuda_benchmark" --action input.xmi > profile.out 2> profile.err)
  srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_gpu_proj_sum --format csv \
    "$out/cov/nsys.nsys-rep" > "$out/cov/nsys_stats.csv" 2> "$out/cov/stats.err"
fi
