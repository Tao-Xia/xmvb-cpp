#!/usr/bin/env bash
# Submit all benchmark jobs for thesis tables 2.2 and 2.3.
# Usage: bash run_benchmarks.sh
set -euo pipefail

cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"
RESULT_DIR="${REPO_ROOT}/benchmark_results"
mkdir -p "${RESULT_DIR}"

submit_tnhvp() {
  local input="$1"
  local stem
  stem="$(basename "${input%.xmi}")"
  local log="${RESULT_DIR}/${stem}_tnhvp.log"
  echo "[TNHVP] ${input} -> ${log}"
  sbatch -c 32 --job-name="tnhvp_${stem}" \
    vbscf-cpp-tnhvp.sh "${input}" --max-iterations 200
}

submit_lbfgs() {
  local input="$1"
  local stem
  stem="$(basename "${input%.xmi}")"
  echo "[LBFGS] ${input}"
  OPTIMIZER_BACKEND=lbfgspp \
    sbatch -c 32 --job-name="lbfgs_${stem}" \
    vbscf-cpp.sh "${input}" --max-iterations 2000
}

submit_legacy() {
  local input="$1"
  local stem
  stem="$(basename "${input%.xmi}")"
  echo "[Legacy] ${input}"
  sbatch -c 32 --job-name="legacy_${stem}" \
    xmvb.sh "${input}"
}

# --- TNHVP jobs ---
submit_tnhvp test/F2.xmi
submit_tnhvp test/241_iscf6.xmi
submit_tnhvp test/MnF2_tnhvp.xmi
submit_tnhvp test/TiCl.xmi
submit_tnhvp test/FeCl_tnhvp.xmi
submit_tnhvp test/FeCl2_tnhvp.xmi
submit_tnhvp test/82712.xmi
submit_tnhvp test/240.xmi

# --- LBFGS jobs ---
submit_lbfgs test/F2.xmi
submit_lbfgs test/241_iscf6.xmi
submit_lbfgs test/MnF2_lbfgs.xmi
submit_lbfgs test/TiCl.xmi
submit_lbfgs test/FeCl_lbfgs.xmi
submit_lbfgs test/FeCl2_lbfgs.xmi
submit_lbfgs test/82712.xmi
submit_lbfgs test/240.xmi

# --- Legacy XMVB jobs ---
submit_legacy test/F2.xmi
submit_legacy test/241_iscf6.xmi
submit_legacy test/MnF2_lbfgs.xmi
submit_legacy test/TiCl.xmi
submit_legacy test/FeCl_lbfgs.xmi
submit_legacy test/FeCl2_lbfgs.xmi
submit_legacy test/82712.xmi
submit_legacy test/240.xmi

echo ""
echo "All 24 jobs submitted. Use 'squeue -u \$(whoami)' to monitor."
echo "Results will be collected after completion."
