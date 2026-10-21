#!/usr/bin/env bash
#SBATCH --job-name=cuvb-profile
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=cuvb-profile-%j.out
#SBATCH --error=cuvb-profile-%j.err

set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9
set -u
source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
out="$source_dir/cuvb-profile-$SLURM_JOB_ID"
mkdir -p "$out"
export LC_ALL=C OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export OMP_MAX_ACTIVE_LEVELS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib:${LD_LIBRARY_PATH:-}"
hostname > "$out/node.txt"
lscpu > "$out/cpu.txt"
nvidia-smi -q > "$out/gpu.txt"
module list > "$out/modules.txt" 2>&1
find "$source_dir/src/cuda" "$source_dir/src/input/loading" \
  "$source_dir/src/integrals/ao" "$source_dir/src/libcint" \
  "$source_dir/tools/cuvbscf" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/sources.sha256"
cmake --build "$source_dir/build-cuvbscf" --target cuvbscf -j 16 > "$out/build.log" 2>&1
cp "$source_dir/build-cuvbscf/src/cuvbscf" "$out/"
sha256sum "$out/cuvbscf" > "$out/binary.sha256"
case_name=${CUVB_PROFILE_CASE:-full}
[[ "$case_name" == mixed20 || "$case_name" == cov || "$case_name" == ion02 || "$case_name" == full ]] || exit 64
printf '%s\n' "$case_name" > "$out/case.txt"
profile_range=${CUVB_PROFILE_RANGE:-gpu.scf.macro}
printf '%s\n' "$profile_range" > "$out/range.txt"
bundle="${CUVB_CASE_BUNDLE:-$source_dir/cuvbscf-c12-297979/bundle/cases}/$case_name"
printf '%s\n' "$bundle" > "$out/bundle.txt"
cp "$bundle/input.xmi" "$bundle/cc-pvdz.gbs" "$out/"
cd "$out"
sha256sum input.xmi cc-pvdz.gbs > input.sha256
nsys --version > nsys_version.txt
# Capture one selected registered range (the first macro by default). SCF continues normally;
# profiled process time is attribution evidence, never a native speed result.
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  nsys profile --sample=none --cpuctxsw=none --trace=cuda,cublas,nvtx \
    --cuda-graph-trace=graph --capture-range=nvtx --nvtx-capture="$profile_range" \
    --capture-range-end=stop --force-overwrite=true --output="$out/nsys" \
    "$out/cuvbscf" input.xmi > profile.out 2> profile.err
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,nvtx_gpu_proj_sum \
    --format csv "$out/nsys.nsys-rep" > nsys_stats.csv 2> stats.err
