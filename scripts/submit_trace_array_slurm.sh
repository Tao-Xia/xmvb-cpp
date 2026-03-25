#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runner_script="${RUNNER_SCRIPT:-$repo_root/scripts/run_trace_manifest_entry_slurm.sh}"
input_dir="${INPUT_DIR:-$repo_root/data/training_xmi/6e6o_full}"
manifest_file="${MANIFEST_FILE:-$input_dir/manifest.txt}"
data_root="${DATA_ROOT:-$repo_root/data}"
dataset_root="${DATASET_ROOT:-$data_root/traces}"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
algorithm="${ALGORITHM:-original}"
cpus_per_task="${CPUS_PER_TASK:-1}"
omp_threads="${OMP_NUM_THREADS:-$cpus_per_task}"
xmvb_cpp_num_threads="${XMVB_CPP_NUM_THREADS:-$omp_threads}"
force_rerun="${FORCE_RERUN:-0}"
job_name="${JOB_NAME:-xmvb_batch}"
job_script="${JOB_SCRIPT:-/export/home/xiatao/dpvbh_data/xmi_file/xt-xmvb.sh}"
max_running="${MAX_RUNNING:-8}"
partition="${PARTITION:-}"
account="${ACCOUNT:-}"
qos="${QOS:-}"
time_limit="${TIME_LIMIT:-}"
onnx_model="${ONNX_MODEL:-}"
dry_run="${DRY_RUN:-0}"

if [[ ! -x "$runner_script" ]]; then
  echo "missing runner script: $runner_script" >&2
  exit 1
fi

if [[ ! -x "$job_script" ]]; then
  echo "missing job script: $job_script" >&2
  exit 1
fi

if [[ ! -d "$input_dir" ]]; then
  echo "missing input directory: $input_dir" >&2
  exit 1
fi

if [[ ! -f "$manifest_file" ]]; then
  echo "missing manifest file: $manifest_file" >&2
  exit 1
fi

if ! [[ "$cpus_per_task" =~ ^[0-9]+$ ]] || (( cpus_per_task < 1 )); then
  echo "CPUS_PER_TASK must be a positive integer" >&2
  exit 1
fi

if ! [[ "$omp_threads" =~ ^[0-9]+$ ]] || (( omp_threads < 1 )); then
  echo "OMP_NUM_THREADS must be a positive integer" >&2
  exit 1
fi

if ! [[ "$xmvb_cpp_num_threads" =~ ^[0-9]+$ ]] || (( xmvb_cpp_num_threads < 1 )); then
  echo "XMVB_CPP_NUM_THREADS must be a positive integer" >&2
  exit 1
fi

if ! [[ "$max_running" =~ ^[0-9]+$ ]] || (( max_running < 1 )); then
  echo "MAX_RUNNING must be a positive integer" >&2
  exit 1
fi

mapfile -t manifest_entries < <(grep -v '^[[:space:]]*$' "$manifest_file")
if (( ${#manifest_entries[@]} == 0 )); then
  echo "manifest is empty: $manifest_file" >&2
  exit 1
fi

array_spec="0-$(( ${#manifest_entries[@]} - 1 ))%${max_running}"
sbatch_args=(
  sbatch
  "--job-name=${job_name}"
  "--array=${array_spec}"
  "--cpus-per-task=${cpus_per_task}"
  "--export=ALL,XMVB_CPP_REPO=${repo_root},DATA_ROOT=${data_root},DATASET_ROOT=${dataset_root},OPTIMIZER_BACKEND=${optimizer_backend},ALGORITHM=${algorithm},OMP_NUM_THREADS=${omp_threads},XMVB_CPP_NUM_THREADS=${xmvb_cpp_num_threads},FORCE_RERUN=${force_rerun},JOB_SCRIPT=${job_script},INPUT_DIR=${input_dir},MANIFEST_FILE=${manifest_file}"
)

if [[ -n "$partition" ]]; then
  sbatch_args+=("--partition=${partition}")
fi
if [[ -n "$account" ]]; then
  sbatch_args+=("--account=${account}")
fi
if [[ -n "$qos" ]]; then
  sbatch_args+=("--qos=${qos}")
fi
if [[ -n "$time_limit" ]]; then
  sbatch_args+=("--time=${time_limit}")
fi
if [[ -n "$onnx_model" ]]; then
  sbatch_args[4]="${sbatch_args[4]},ONNX_MODEL=${onnx_model}"
fi

sbatch_args+=("$runner_script")

echo "repo_root = $repo_root"
echo "runner_script = $runner_script"
echo "job_script = $job_script"
echo "input_dir = $input_dir"
echo "manifest = $manifest_file"
echo "dataset_root = $dataset_root"
echo "optimizer_backend = $optimizer_backend"
echo "algorithm = $algorithm"
echo "cpus_per_task = $cpus_per_task"
echo "omp_threads = $omp_threads"
echo "xmvb_cpp_num_threads = $xmvb_cpp_num_threads"
echo "max_running = $max_running"
echo "manifest_entries = ${#manifest_entries[@]}"

if (( dry_run != 0 )); then
  printf 'DRY_RUN '
  printf '%q ' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi

"${sbatch_args[@]}"
