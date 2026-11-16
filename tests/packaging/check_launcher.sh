#!/usr/bin/env bash
set -euo pipefail
launcher=$(realpath "$1")
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/program/share" "$scratch/bin"
printf 'fixture\n' > "$scratch/program/share/RUNTIME_ID"
cat > "$scratch/bin/apptainer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
test "$1" = exec
test "$2" = --cleanenv
env | sort > "$CAPTURE"
EOF
chmod +x "$scratch/bin/apptainer"
invoke() {
    env -i PATH="$scratch/bin:$PATH" HOME="$HOME" CAPTURE="$scratch/environment" \
        XMVB_RUNTIME_IMAGE=fixture.sif XMVB_PROGRAM_DIR="$scratch/program" "$@" \
        bash "$launcher" input.xmi
}
invoke SLURM_CPUS_PER_TASK=32 SLURM_CPUS_ON_NODE=32 SLURM_MEM_PER_NODE=98304
grep -Fx 'APPTAINERENV_SLURM_CPUS_PER_TASK=32' "$scratch/environment"
grep -Fx 'APPTAINERENV_SLURM_MEM_PER_NODE=98304' "$scratch/environment"
grep -Fx 'APPTAINERENV_OMP_PROC_BIND=close' "$scratch/environment"
! grep -q '^APPTAINERENV_OMP_NUM_THREADS=' "$scratch/environment"
invoke OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 'OMP_PLACES={0,1},{2,3}' OMP_PROC_BIND=spread
grep -Fx 'APPTAINERENV_OMP_NUM_THREADS=8' "$scratch/environment"
grep -Fx 'APPTAINERENV_OPENBLAS_NUM_THREADS=2' "$scratch/environment"
grep -Fx 'APPTAINERENV_OMP_PLACES={0,1},{2,3}' "$scratch/environment"
grep -Fx 'APPTAINERENV_OMP_PROC_BIND=spread' "$scratch/environment"
invoke APPTAINERENV_OMP_NUM_THREADS=6 OMP_NUM_THREADS=8
grep -Fx 'APPTAINERENV_OMP_NUM_THREADS=6' "$scratch/environment"
