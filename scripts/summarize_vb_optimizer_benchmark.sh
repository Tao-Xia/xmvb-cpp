#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${script_dir}/.." && pwd -P)"

usage() {
  cat >&2 <<'EOF'
usage: bash scripts/summarize_vb_optimizer_benchmark.sh <benchmark_dir|jobs.tsv> [summary.tsv]

Reads the jobs.tsv produced by submit_vb_optimizer_benchmark_triplet.sh and
extracts the main convergence and timing metrics for cpp_tnhvp, cpp_lbfgs, and
legacy_xmvb into one summary table.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

input_path="$1"
if [[ -d "${input_path}" ]]; then
  jobs_tsv="${input_path%/}/jobs.tsv"
  summary_tsv="${2:-${input_path%/}/summary.tsv}"
else
  jobs_tsv="${input_path}"
  summary_tsv="${2:-$(dirname "${jobs_tsv}")/summary.tsv}"
fi

if [[ ! -f "${jobs_tsv}" ]]; then
  echo "missing jobs.tsv: ${jobs_tsv}" >&2
  exit 1
fi

mkdir -p "$(dirname "${summary_tsv}")"

declare -A sacct_state_by_job=()
declare -A sacct_elapsed_by_job=()

collect_sacct_data() {
  if ! command -v sacct >/dev/null 2>&1; then
    return
  fi

  local job_id_list=""
  local mode=""
  local job_id=""
  local _job_script=""
  local _primary_log=""
  local _secondary_log=""
  local _work_dir=""
  local _slurm_dir=""
  while IFS=$'\t' read -r mode job_id _job_script _primary_log _secondary_log _work_dir _slurm_dir; do
    if [[ -z "${mode}" || "${mode}" == "mode" ]]; then
      continue
    fi
    if ! [[ "${job_id}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if [[ -n "${job_id_list}" ]]; then
      job_id_list+=","
    fi
    job_id_list+="${job_id}"
  done < "${jobs_tsv}"

  if [[ -z "${job_id_list}" ]]; then
    return
  fi

  local raw_job_id=""
  local state=""
  local elapsed_raw=""
  while IFS='|' read -r raw_job_id state elapsed_raw; do
    if [[ -z "${raw_job_id}" || "${raw_job_id}" == *.* ]]; then
      continue
    fi
    if [[ -z "${sacct_state_by_job[${raw_job_id}]:-}" ]]; then
      sacct_state_by_job["${raw_job_id}"]="${state}"
      sacct_elapsed_by_job["${raw_job_id}"]="${elapsed_raw}"
    fi
  done < <(
    sacct -X -P -n -j "${job_id_list}" --format JobIDRaw,State,ElapsedRaw 2>/dev/null || true
  )
}

parse_cpp_log() {
  local run_log="$1"
  awk '
    / : / {
      key = $0
      sub(/[[:space:]]*: .*/, "", key)
      gsub(/[[:space:]]+$/, "", key)
      value = $0
      sub(/^.* : /, "", value)
      values[key] = value
    }
    END {
      optimizer_wall = values["Optimizer internal wall time"]
      if (optimizer_wall == "") {
        optimizer_wall = values["SCF iteration wall time"]
      }
      sub(/ s$/, "", optimizer_wall)
      e2e_wall = values["End-to-end wall time"]
      sub(/ s$/, "", e2e_wall)
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        values["Status"],
        values["Termination reason"],
        values["Iterations"],
        values["Final total energy"],
        values["Final gradient |g|_inf"],
        optimizer_wall,
        e2e_wall,
        values["HVP mode"]
    }
  ' "${run_log}"
}

parse_xmvb_xmo() {
  local xmo_file="$1"
  awk '
    BEGIN {
      converged = "false"
      iterations = ""
      final_total_energy = ""
      method = ""
      cpu_time_seconds = ""
    }
    /OPTIMIZATION METHOD:/ {
      line = $0
      sub(/^.*OPTIMIZATION METHOD:[[:space:]]*/, "", line)
      method = line
    }
    /VBSCF converged in/ {
      converged = "true"
      line = $0
      sub(/^.*VBSCF converged in[[:space:]]+/, "", line)
      sub(/[[:space:]]+iterations.*$/, "", line)
      iterations = line
    }
    /Cpu time for the job:/ {
      line = $0
      sub(/^.*Cpu time for the job:[[:space:]]+/, "", line)
      sub(/[[:space:]]+seconds\..*$/, "", line)
      cpu_time_seconds = line
    }
    /Total Energy:/ {
      line = $0
      sub(/^.*Total Energy:[[:space:]]+/, "", line)
      final_total_energy = line
    }
    /TOTAL ENERGY[[:space:]]*:/ {
      if (final_total_energy == "") {
        final_total_energy = $NF
      }
    }
    END {
      if (method == "") {
        method = "legacy_xmvb"
      }
      printf "%s\t%s\t%s\t%s\t%s\n",
        converged,
        iterations,
        final_total_energy,
        method,
        cpu_time_seconds
    }
  ' "${xmo_file}"
}

resolve_legacy_xmo() {
  local primary_log="$1"
  local work_dir="$2"
  local sample_stem

  if [[ -f "${primary_log}" ]]; then
    printf '%s\n' "${primary_log}"
    return 0
  fi

  sample_stem="$(basename "${primary_log}" .xmo)"
  if [[ -n "${work_dir}" && -f "${work_dir}/${sample_stem}.xmo" ]]; then
    printf '%s\n' "${work_dir}/${sample_stem}.xmo"
    return 0
  fi

  if [[ -f "${repo_root}/test/${sample_stem}.xmo" ]]; then
    printf '%s\n' "${repo_root}/test/${sample_stem}.xmo"
    return 0
  fi

  return 1
}

collect_sacct_data

printf 'mode\tjob_id\tscheduler_state\tscheduler_elapsed_seconds\tconverged\ttermination_reason\titerations\tfinal_total_energy\tfinal_gradient_inf_norm\toptimizer_internal_wall_time_seconds\tend_to_end_wall_time_seconds\txmo_cpu_time_seconds\toptimizer_or_method\thvp_mode\tprimary_log\tsecondary_log\n' > "${summary_tsv}"

tail -n +2 "${jobs_tsv}" | while IFS=$'\t' read -r mode job_id job_script primary_log secondary_log work_dir slurm_dir; do
  if [[ -z "${mode}" ]]; then
    continue
  fi

  scheduler_state="${sacct_state_by_job[${job_id}]:-}"
  scheduler_elapsed="${sacct_elapsed_by_job[${job_id}]:-}"
  converged=""
  termination_reason=""
  iterations=""
  final_total_energy=""
  final_gradient_inf_norm=""
  optimizer_internal_wall_time_seconds=""
  end_to_end_wall_time_seconds=""
  xmo_cpu_time_seconds=""
  optimizer_or_method=""
  hvp_mode=""
  resolved_primary_log="${primary_log}"

  if [[ "${mode}" == cpp_* ]]; then
    optimizer_or_method="${mode#cpp_}"
    if [[ -f "${primary_log}" ]]; then
      IFS=$'\t' read -r converged termination_reason iterations final_total_energy \
        final_gradient_inf_norm optimizer_internal_wall_time_seconds \
        end_to_end_wall_time_seconds hvp_mode < <(parse_cpp_log "${primary_log}")
    else
      converged="missing_log"
    fi
  elif [[ "${mode}" == "legacy_xmvb" ]]; then
    optimizer_or_method="legacy_xmvb"
    if resolved_primary_log="$(resolve_legacy_xmo "${primary_log}" "${work_dir}")"; then
      legacy_method=""
      IFS=$'\t' read -r converged iterations final_total_energy legacy_method xmo_cpu_time_seconds \
        < <(parse_xmvb_xmo "${resolved_primary_log}")
      if [[ -n "${legacy_method}" && "${legacy_method}" != "legacy_xmvb" ]]; then
        optimizer_or_method="${legacy_method}"
      fi
      end_to_end_wall_time_seconds="${xmo_cpu_time_seconds}"
    else
      converged="missing_log"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" \
    "${job_id}" \
    "${scheduler_state}" \
    "${scheduler_elapsed}" \
    "${converged}" \
    "${termination_reason}" \
    "${iterations}" \
    "${final_total_energy}" \
    "${final_gradient_inf_norm}" \
    "${optimizer_internal_wall_time_seconds}" \
    "${end_to_end_wall_time_seconds}" \
    "${xmo_cpu_time_seconds}" \
    "${optimizer_or_method}" \
    "${hvp_mode}" \
    "${resolved_primary_log}" \
    "${secondary_log}" \
    >> "${summary_tsv}"
done

echo "summary_tsv = ${summary_tsv}"
