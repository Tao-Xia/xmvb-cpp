#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
batch_script="${BATCH_SCRIPT:-$repo_root/scripts/export_trace_batch.sh}"
dataset_root="${DATASET_ROOT:-$repo_root/artifacts/6e6o_full_training_trace}"
launch_log_dir="$dataset_root/launcher_logs"
launcher_stdout="$launch_log_dir/launcher.stdout.log"
launcher_stderr="$launch_log_dir/launcher.stderr.log"
pid_file="$launch_log_dir/pids.txt"
shard_count="${SHARD_COUNT:-$(nproc)}"
omp_threads="${OMP_NUM_THREADS:-1}"

if [[ ! -x "$batch_script" ]]; then
  echo "missing batch script: $batch_script" >&2
  exit 1
fi

if ! [[ "$shard_count" =~ ^[0-9]+$ ]] || (( shard_count < 1 )); then
  echo "SHARD_COUNT must be a positive integer" >&2
  exit 1
fi

mkdir -p "$launch_log_dir"
: > "$launcher_stdout"
: > "$launcher_stderr"
: > "$pid_file"

echo "repo_root = $repo_root" | tee -a "$launcher_stdout"
echo "batch_script = $batch_script" | tee -a "$launcher_stdout"
echo "dataset_root = $dataset_root" | tee -a "$launcher_stdout"
echo "shard_count = $shard_count" | tee -a "$launcher_stdout"
echo "omp_threads = $omp_threads" | tee -a "$launcher_stdout"

for (( shard_index=0; shard_index<shard_count; ++shard_index )); do
  shard_stdout="$launch_log_dir/shard_${shard_index}.stdout.log"
  shard_stderr="$launch_log_dir/shard_${shard_index}.stderr.log"
  nohup env \
      OMP_NUM_THREADS="$omp_threads" \
      SHARD_COUNT="$shard_count" \
      SHARD_INDEX="$shard_index" \
      DATASET_ROOT="$dataset_root" \
      "$batch_script" \
      >"$shard_stdout" 2>"$shard_stderr" < /dev/null &
  pid=$!
  printf '%s\t%s\t%s\t%s\n' "$pid" "$shard_index" "$shard_stdout" "$shard_stderr" \
      | tee -a "$pid_file" >> "$launcher_stdout"
done

echo "pid_file = $pid_file" | tee -a "$launcher_stdout"
