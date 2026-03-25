#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="${INPUT_DIR:-$repo_root/data/training_xmi/6e6o_full}"
manifest_file="${MANIFEST_FILE:-$input_dir/manifest.txt}"
job_script="${JOB_SCRIPT:-/export/home/xiatao/dpvbh_data/xmi_file/xt-xmvb.sh}"
data_root="${DATA_ROOT:-$repo_root/data}"
dataset_root="${DATASET_ROOT:-$data_root/traces}"
log_dir="${LOG_DIR:-$dataset_root/logs}"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
algorithm="${ALGORITHM:-original}"
cpus_per_task="${CPUS_PER_TASK:-1}"
omp_threads="${OMP_NUM_THREADS:-$cpus_per_task}"
xmvb_cpp_num_threads="${XMVB_CPP_NUM_THREADS:-$omp_threads}"
force_rerun="${FORCE_RERUN:-0}"
dry_run="${DRY_RUN:-0}"
job_name_prefix="${JOB_NAME_PREFIX:-xmvb}"
partition="${PARTITION:-}"
account="${ACCOUNT:-}"
qos="${QOS:-}"
time_limit="${TIME_LIMIT:-}"
onnx_model="${ONNX_MODEL:-}"

if [[ ! -d "$input_dir" ]]; then
  echo "missing input directory: $input_dir" >&2
  exit 1
fi

if [[ ! -f "$manifest_file" ]]; then
  echo "missing manifest file: $manifest_file" >&2
  exit 1
fi

if [[ ! -x "$job_script" ]]; then
  echo "missing job script: $job_script" >&2
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

mkdir -p "$dataset_root" "$log_dir"
submit_log="$log_dir/submit_$(date +%Y%m%d_%H%M%S).log"

mapfile -t manifest_entries < <(grep -v '^[[:space:]]*$' "$manifest_file")
if (( ${#manifest_entries[@]} == 0 )); then
  echo "manifest is empty: $manifest_file" >&2
  exit 1
fi

echo "repo_root = $repo_root" | tee -a "$submit_log"
echo "job_script = $job_script" | tee -a "$submit_log"
echo "input_dir = $input_dir" | tee -a "$submit_log"
echo "manifest = $manifest_file" | tee -a "$submit_log"
echo "dataset_root = $dataset_root" | tee -a "$submit_log"
echo "optimizer_backend = $optimizer_backend" | tee -a "$submit_log"
echo "algorithm = $algorithm" | tee -a "$submit_log"
echo "cpus_per_task = $cpus_per_task" | tee -a "$submit_log"
echo "omp_threads = $omp_threads" | tee -a "$submit_log"
echo "xmvb_cpp_num_threads = $xmvb_cpp_num_threads" | tee -a "$submit_log"
echo "manifest_entries = ${#manifest_entries[@]}" | tee -a "$submit_log"

submitted=0
missing=0

for sample_name in "${manifest_entries[@]}"; do
  xmi_path="$input_dir/$sample_name"
  sample_stem="${sample_name%.xmi}"

  if [[ ! -f "$xmi_path" ]]; then
    echo "missing source file: $xmi_path" | tee -a "$submit_log" >&2
    ((missing += 1))
    continue
  fi

  sbatch_args=(
    sbatch
    "--job-name=${job_name_prefix}_${sample_stem}"
    "--cpus-per-task=${cpus_per_task}"
    "--export=ALL,XMVB_CPP_REPO=${repo_root},DATA_ROOT=${data_root},DATASET_ROOT=${dataset_root},OPTIMIZER_BACKEND=${optimizer_backend},ALGORITHM=${algorithm},OMP_NUM_THREADS=${omp_threads},XMVB_CPP_NUM_THREADS=${xmvb_cpp_num_threads},FORCE_RERUN=${force_rerun}"
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
    sbatch_args[3]="${sbatch_args[3]},ONNX_MODEL=${onnx_model}"
  fi

  sbatch_args+=("$job_script" "$xmi_path")

  if (( dry_run != 0 )); then
    printf 'DRY_RUN ' | tee -a "$submit_log"
    printf '%q ' "${sbatch_args[@]}" | tee -a "$submit_log"
    printf '\n' | tee -a "$submit_log"
  else
    "${sbatch_args[@]}" | tee -a "$submit_log"
  fi
  ((submitted += 1))
done

echo "summary submitted=$submitted missing=$missing submit_log=$submit_log"
