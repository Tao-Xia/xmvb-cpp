#!/usr/bin/env bash
#SBATCH --job-name=cuvb-scf
#SBATCH --partition=GPU-8A100
#SBATCH --qos=gpu_8a100
#SBATCH --account=taoxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=cuvb-scf-%j.out
#SBATCH --error=cuvb-scf-%j.err

set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 64
. /etc/profile
module purge
module load gcc/14.3.0 cuda/13.0 cmake/3.31.9
set -u
source_dir=$(realpath -e "$SLURM_SUBMIT_DIR")
deps=/home/guqqgroup/taoxia/software/conda-envs/xmvb-cpp-89c74a6
out="$source_dir/cuvb-scf-$SLURM_JOB_ID"
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
  "$source_dir/tools/cuvbscf" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$out/sources.sha256"
cmake -S "$source_dir" -B "$source_dir/build-cuvbscf" > "$out/configure.log" 2>&1
cmake --build "$source_dir/build-cuvbscf" --target cuvbscf -j 16 > "$out/build.log" 2>&1
cp "$source_dir/build-cuvbscf/src/cuvbscf" "$out/"
sha256sum "$out/cuvbscf" > "$out/binary.sha256"
bundle="${CUVB_CASE_BUNDLE:-$source_dir/cuvbscf-c12-297979/bundle/cases}"
printf '%s\n' "$bundle" > "$out/bundle.txt"
status=0
read -ra cases <<< "${CUVB_CASES:-mixed20 cov ion02 full}"
for case in "${cases[@]}"; do
  [[ "$case" == mixed20 || "$case" == cov || "$case" == ion02 || "$case" == full ]] || exit 64
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
      /usr/bin/time -v -o gpu.time bash -c \
        'printf "%s\n" "$$" > gpu.pid; exec "$1" input.xmi' _ "$out/cuvbscf" \
        > gpu.out 2> gpu.err
  ) || status=1
done
exit "$status"
