#!/usr/bin/env bash
# Smoke test: build and run F2 with LBFGS and TNHVP optimizers.
# Usage: ./test/smoke_test_f2.sh [--no-build]
# Exit 0 if both converge, exit 1 on build failure or non-convergence.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
exe="${repo_root}/build/src/xmvb-cpp.exe"
input="${repo_root}/test/F2.xmi"
threads="${OMP_NUM_THREADS:-4}"
timeout_s=120

# --- Parse args ---
do_build=1
if [[ "${1:-}" == "--no-build" ]]; then
  do_build=0
fi

# --- Build ---
if (( do_build )); then
  echo "=== Building ==="
  cmake --build "${repo_root}/build" -j"$(nproc)" 2>&1
  if [[ ! -x "${exe}" ]]; then
    echo "FAIL: build succeeded but executable not found: ${exe}" >&2
    exit 1
  fi
  echo "=== Build OK ==="
  echo
fi

if [[ ! -x "${exe}" ]]; then
  echo "FAIL: executable not found: ${exe}" >&2
  echo "Run without --no-build first." >&2
  exit 1
fi

run_test() {
  local backend="$1"
  local label="$2"
  echo "=== Running F2 (${label}) ==="
  local log
  log="$(mktemp /tmp/f2_smoke_${backend}.XXXXXX.log)"
  local rc=0
  OMP_NUM_THREADS="${threads}" \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  timeout "${timeout_s}" \
    "${exe}" "${input}" --optimizer-backend "${backend}" > "${log}" 2>&1 || rc=$?

  if (( rc != 0 )); then
    echo "FAIL: ${label} exited with code ${rc}" >&2
    tail -20 "${log}" >&2
    rm -f "${log}"
    return 1
  fi

  # Check convergence
  if grep -q 'Status.*:.*converged' "${log}"; then
    local energy iterations
    energy="$(grep 'Final total energy' "${log}" | awk '{print $NF}')"
    iterations="$(grep '^Iterations' "${log}" | awk '{print $NF}')"
    echo "PASS: ${label} converged in ${iterations} iters, E = ${energy}"
  else
    echo "FAIL: ${label} did NOT converge" >&2
    tail -20 "${log}" >&2
    rm -f "${log}"
    return 1
  fi

  rm -f "${log}"
  return 0
}

fail=0

if ! run_test lbfgspp "LBFGS"; then
  fail=1
fi
echo

if ! run_test nonredundant_truncated_newton "TNHVP"; then
  fail=1
fi
echo

if (( fail )); then
  echo "=== SMOKE TEST FAILED ===" >&2
  exit 1
else
  echo "=== ALL SMOKE TESTS PASSED ==="
  exit 0
fi
