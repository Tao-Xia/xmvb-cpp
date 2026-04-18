#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
input_dir="${INPUT_DIR:-$repo_root/data/training_xmi/6e6o_full}"
manifest_file="${MANIFEST_FILE:-$input_dir/manifest.txt}"
dataset_root="${DATASET_ROOT:-$repo_root/artifacts/6e6o_full_training_trace}"
binary_path="${XMVB_CPP_BIN:-$repo_root/build/src/xmvb-cpp.exe}"
optimizer_backend="${OPTIMIZER_BACKEND:-lbfgspp}"
algorithm="${ALGORITHM:-original}"
omp_threads="${OMP_NUM_THREADS:-1}"
xmvb_cpp_num_threads="${XMVB_CPP_NUM_THREADS:-$omp_threads}"
shard_index="${SHARD_INDEX:-0}"
shard_count="${SHARD_COUNT:-1}"
stop_on_error="${STOP_ON_ERROR:-1}"
dry_run="${DRY_RUN:-0}"

if [[ ! -x "$binary_path" ]]; then
  echo "missing executable: $binary_path" >&2
  exit 1
fi

runtime_env_helper="$repo_root/scripts/xmvb_cpp_runtime_env.sh"
if [[ ! -f "$runtime_env_helper" ]]; then
  echo "missing runtime env helper: $runtime_env_helper" >&2
  exit 1
fi
source "$runtime_env_helper"
prepare_xmvb_cpp_runtime_env "$binary_path" "$repo_root"
prepare_xmvb_cpp_thread_env "$omp_threads"

if [[ ! -d "$input_dir" ]]; then
  echo "missing input directory: $input_dir" >&2
  exit 1
fi

if [[ ! -f "$manifest_file" ]]; then
  echo "missing manifest file: $manifest_file" >&2
  exit 1
fi

if ! [[ "$shard_index" =~ ^[0-9]+$ && "$shard_count" =~ ^[0-9]+$ ]]; then
  echo "SHARD_INDEX and SHARD_COUNT must be non-negative integers" >&2
  exit 1
fi

if (( shard_count < 1 || shard_index >= shard_count )); then
  echo "invalid shard configuration: SHARD_INDEX=$shard_index SHARD_COUNT=$shard_count" >&2
  exit 1
fi

mkdir -p "$dataset_root"
log_dir="$dataset_root/logs"
mkdir -p "$log_dir"

mapfile -t manifest_entries < <(grep -v '^[[:space:]]*$' "$manifest_file")

selected=0
completed=0
skipped=0
failed=0

echo "repo_root = $repo_root"
echo "binary = $binary_path"
echo "input_dir = $input_dir"
echo "manifest = $manifest_file"
echo "dataset_root = $dataset_root"
echo "optimizer_backend = $optimizer_backend"
echo "algorithm = $algorithm"
log_xmvb_cpp_thread_env
echo "xmvb_cpp_num_threads = $xmvb_cpp_num_threads"
echo "ld_library_path = ${LD_LIBRARY_PATH:-<unset>}"
echo "shard = $shard_index / $shard_count"
echo "manifest_entries = ${#manifest_entries[@]}"

for index in "${!manifest_entries[@]}"; do
  if (( index % shard_count != shard_index )); then
    continue
  fi

  sample_name="${manifest_entries[$index]}"
  xmi_path="$input_dir/$sample_name"
  stem="${sample_name%.xmi}"
  sample_dir="$dataset_root/$stem"
  stdout_log="$log_dir/$stem.stdout.log"
  stderr_log="$log_dir/$stem.stderr.log"
  ((selected += 1))

  if [[ ! -f "$xmi_path" ]]; then
    echo "missing source file: $xmi_path" >&2
    ((failed += 1))
    if (( stop_on_error != 0 )); then
      exit 1
    fi
    continue
  fi

  if [[ -f "$sample_dir/metadata.json" ]]; then
    if grep -q '"status": "completed"' "$sample_dir/metadata.json"; then
      echo "skip completed: $stem"
      ((skipped += 1))
      continue
    fi
    echo "incomplete existing sample directory: $sample_dir" >&2
    ((failed += 1))
    if (( stop_on_error != 0 )); then
      exit 1
    fi
    continue
  fi

  echo "run [$((index + 1))/${#manifest_entries[@]}] $sample_name"
  if (( dry_run != 0 )); then
    continue
  fi

  if XMVB_CPP_NUM_THREADS="$xmvb_cpp_num_threads" \
      "$binary_path" \
      "$xmi_path" \
      --optimizer-backend "$optimizer_backend" \
      --algorithm "$algorithm" \
      --dump-trace-dir "$dataset_root" \
      >"$stdout_log" 2>"$stderr_log"; then
    ((completed += 1))
  else
    echo "failed: $sample_name" >&2
    ((failed += 1))
    if (( stop_on_error != 0 )); then
      exit 1
    fi
  fi
done

echo "summary selected=$selected completed=$completed skipped=$skipped failed=$failed"
