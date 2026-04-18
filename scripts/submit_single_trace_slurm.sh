#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_script="${JOB_SCRIPT:-$repo_root/scripts/xt-xmvb.sh}"
data_root="${DATA_ROOT:-$repo_root/data}"
dataset_root="${DATASET_ROOT:-$data_root/matrix_data}"
log_dir="${LOG_DIR:-$dataset_root/logs}"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
algorithm="${ALGORITHM:-original}"
cpus_per_task="${CPUS_PER_TASK:-32}"
omp_threads="${OMP_NUM_THREADS:-$cpus_per_task}"
xmvb_cpp_num_threads="${XMVB_CPP_NUM_THREADS:-$omp_threads}"
force_rerun="${FORCE_RERUN:-0}"
partition="${PARTITION:-6226r}"
account="${ACCOUNT:-weiwu}"
qos="${QOS:-}"
time_limit="${TIME_LIMIT:-24:00:00}"
mem="${MEM:-}"
onnx_model="${ONNX_MODEL:-}"
dry_run="${DRY_RUN:-0}"
runtime_progress_log="${XMVB_CPP_LOG_RUNTIME_PROGRESS:-}"
objective_progress_log="${XMVB_CPP_LOG_OBJECTIVE_PROGRESS:-}"
echo_input_flag="${XMVB_CPP_ECHO_INPUT:-}"
dump_trace="${DUMP_TRACE:-1}"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/submit_single_trace_slurm.sh <input.xmi>

Environment overrides:
  JOB_SCRIPT                Default: scripts/xt-xmvb.sh
  DATA_ROOT                 Default: <repo>/data
  DATASET_ROOT              Default: <repo>/data/matrix_data
  LOG_DIR                   Default: <dataset_root>/logs
  OPTIMIZER_BACKEND         Default: lbfgspp
  ALGORITHM                 Default: original
  CPUS_PER_TASK             Default: 32
  OMP_NUM_THREADS           Default: CPUS_PER_TASK
  XMVB_CPP_NUM_THREADS      Default: OMP_NUM_THREADS
  FORCE_RERUN               Default: 0
  PARTITION                 Default: 6226r
  ACCOUNT                   Default: weiwu
  QOS                       Default: empty
  TIME_LIMIT                Default: 24:00:00
  MEM                       Default: unset
  XMVB_CPP_LOG_RUNTIME_PROGRESS
  XMVB_CPP_LOG_OBJECTIVE_PROGRESS
  XMVB_CPP_ECHO_INPUT
  DUMP_TRACE                Default: 1
  DRY_RUN                   Default: 0
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

input_file="$(readlink -f "$1")"
if [[ ! -f "$input_file" ]]; then
  echo "missing input file: $input_file" >&2
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

sample_stem="$(basename "${input_file%.xmi}")"
job_name="${JOB_NAME:-xmvb_${sample_stem}}"

mkdir -p "$dataset_root" "$log_dir"
submit_log="${log_dir}/submit_single_${sample_stem}_$(date +%Y%m%d_%H%M%S).log"

export_args=(
  "ALL"
  "DATA_ROOT=${data_root}"
  "DATASET_ROOT=${dataset_root}"
  "OPTIMIZER_BACKEND=${optimizer_backend}"
  "ALGORITHM=${algorithm}"
  "OMP_NUM_THREADS=${omp_threads}"
  "XMVB_CPP_NUM_THREADS=${xmvb_cpp_num_threads}"
  "FORCE_RERUN=${force_rerun}"
)

if [[ -n "$onnx_model" ]]; then
  export_args+=("ONNX_MODEL=${onnx_model}")
fi
if [[ -n "$runtime_progress_log" ]]; then
  export_args+=("XMVB_CPP_LOG_RUNTIME_PROGRESS=${runtime_progress_log}")
fi
if [[ -n "$objective_progress_log" ]]; then
  export_args+=("XMVB_CPP_LOG_OBJECTIVE_PROGRESS=${objective_progress_log}")
fi
if [[ -n "$echo_input_flag" ]]; then
  export_args+=("XMVB_CPP_ECHO_INPUT=${echo_input_flag}")
fi

export_spec=""
for export_arg in "${export_args[@]}"; do
  if [[ -n "$export_spec" ]]; then
    export_spec+=","
  fi
  export_spec+="$export_arg"
done

sbatch_args=(
  sbatch
  "--job-name=${job_name}"
  "--cpus-per-task=${cpus_per_task}"
  "--export=${export_spec}"
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
if [[ -n "$mem" ]]; then
  sbatch_args+=("--mem=${mem}")
fi

sbatch_args+=("$job_script" "$input_file")

{
  echo "repo_root = $repo_root"
  echo "job_script = $job_script"
  echo "input_file = $input_file"
  echo "sample_stem = $sample_stem"
  echo "dataset_root = $dataset_root"
  echo "log_dir = $log_dir"
  echo "optimizer_backend = $optimizer_backend"
  echo "algorithm = $algorithm"
  echo "cpus_per_task = $cpus_per_task"
  echo "omp_threads = $omp_threads"
  echo "xmvb_cpp_num_threads = $xmvb_cpp_num_threads"
  echo "partition = ${partition:-<unset>}"
  echo "account = ${account:-<unset>}"
  echo "qos = ${qos:-<unset>}"
  echo "time_limit = ${time_limit:-<unset>}"
  echo "mem = ${mem:-<unset>}"
  echo "dump_trace = ${dump_trace}"
  echo "submit_log = $submit_log"
  echo "sample_dir = ${dataset_root}/${sample_stem}"
  echo "stdout_log = ${log_dir}/${sample_stem}.<jobid>.stdout.log"
  echo "stderr_log = ${log_dir}/${sample_stem}.<jobid>.stderr.log"
  echo "slurm_stdout = ${job_name}-<jobid>.slog"
  echo "slurm_stderr = ${job_name}-<jobid>.serr"
} | tee -a "$submit_log"

if (( dry_run != 0 )); then
  printf 'DRY_RUN ' | tee -a "$submit_log"
  printf '%q ' "${sbatch_args[@]}" | tee -a "$submit_log"
  printf '\n' | tee -a "$submit_log"
  exit 0
fi

"${sbatch_args[@]}" | tee -a "$submit_log"
