#!/usr/bin/env bash
#SBATCH --job-name=cuvb-exchange
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=cuvb-exchange-%j.out
#SBATCH --error=cuvb-exchange-%j.err

set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9
set -u
source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
out="$source_dir/cuvb-exchange-$SLURM_JOB_ID"
mkdir -p "$out"
export LC_ALL=C OMP_NUM_THREADS=32 OMP_PLACES=cores OMP_PROC_BIND=close
export OMP_MAX_ACTIVE_LEVELS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export LD_LIBRARY_PATH="$deps/lib:${LD_LIBRARY_PATH:-}"
hostname > "$out/node.txt"
lscpu > "$out/cpu.txt"
nvidia-smi -q > "$out/gpu.txt"
module list > "$out/modules.txt" 2>&1
find "$source_dir/src/cuda" "$source_dir/src/vbscf/structures/assembly" \
  "$source_dir/src/input/loading" "$source_dir/src/integrals/ao" \
  "$source_dir/src/libcint" "$source_dir/tools/cuvbscf" \
  -type f -print0 | sort -z | xargs -0 sha256sum > "$out/sources.sha256"
cmake --build "$source_dir/build-cuvbscf" --target cuvbscf -j 16 > "$out/build.log" 2>&1
cp "$source_dir/build-cuvbscf/src/cuvbscf" "$out/"
sha256sum "$out/cuvbscf" > "$out/binary.sha256"
bundle="${CUVB_CASE_BUNDLE:-$source_dir/cuvb-c14-bundle}/cov"
printf '%s\n' "$bundle" > "$out/bundle.txt"
cp "$bundle/input.xmi" "$bundle/cc-pvdz.gbs" "$out/"
cd "$out"
sha256sum input.xmi cc-pvdz.gbs > input.sha256
ncu --version > ncu_version.txt
# One later accepted-point rank-one exchange, not the sparse initial point.
# Replay metrics attribute memory traffic; they are not native SCF timings.
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  ncu --nvtx --nvtx-include 'ao.rank_one_exchange/' \
    --launch-skip 1 --launch-count 1 --cache-control none \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --force-overwrite --export "$out/ncu" \
    "$out/cuvbscf" input.xmi > profile.out 2> profile.err
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores \
  ncu --import "$out/ncu.ncu-rep" --page details --csv \
    > ncu.csv 2> report.err
