#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_script="${JOB_SCRIPT:-/export/home/xiatao/dpvbh_data/xmi_file/xt-xmvb.sh}"
input_dir="${INPUT_DIR:-$repo_root/data/training_xmi/6e6o_full}"
manifest_file="${MANIFEST_FILE:-$input_dir/manifest.txt}"
task_id="${SLURM_ARRAY_TASK_ID:-}"

if [[ -z "$task_id" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set" >&2
  exit 1
fi

if [[ ! "$task_id" =~ ^[0-9]+$ ]]; then
  echo "invalid SLURM_ARRAY_TASK_ID: $task_id" >&2
  exit 1
fi

if [[ ! -x "$job_script" ]]; then
  echo "missing job script: $job_script" >&2
  exit 1
fi

if [[ ! -f "$manifest_file" ]]; then
  echo "missing manifest file: $manifest_file" >&2
  exit 1
fi

sample_name="$(sed -n "$((task_id + 1))p" "$manifest_file" | tr -d '\r')"
if [[ -z "$sample_name" ]]; then
  echo "manifest entry not found for task index $task_id" >&2
  exit 1
fi

xmi_path="$input_dir/$sample_name"
if [[ ! -f "$xmi_path" ]]; then
  echo "missing source file: $xmi_path" >&2
  exit 1
fi

exec "$job_script" "$xmi_path"
