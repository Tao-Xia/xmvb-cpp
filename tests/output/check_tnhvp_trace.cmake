if (NOT DEFINED XMVB_EXECUTABLE OR
    NOT DEFINED XMVB_INPUT OR
    NOT DEFINED XMVB_TRACE_ROOT)
  message(FATAL_ERROR
    "XMVB_EXECUTABLE, XMVB_INPUT, and XMVB_TRACE_ROOT are required")
endif()

file(REMOVE_RECURSE "${XMVB_TRACE_ROOT}")
set(tnhvp_trace "${XMVB_TRACE_ROOT}/F2_tnhvp.tsv")
execute_process(
  COMMAND
    "${XMVB_EXECUTABLE}"
    "${XMVB_INPUT}"
    --optimizer-backend nonredundant_truncated_newton
    --nonredundant-truncated-newton-hvp-mode exact_ctx
    --dump-trace-dir "${XMVB_TRACE_ROOT}"
    --tnhvp-trace "${tnhvp_trace}"
  RESULT_VARIABLE xmvb_status
  OUTPUT_VARIABLE xmvb_report
  ERROR_VARIABLE xmvb_errors)

if (NOT xmvb_status EQUAL 0)
  message(FATAL_ERROR
    "F2 TNHVP trace run failed with status ${xmvb_status}:\n${xmvb_errors}")
endif()

set(initial_metadata "${XMVB_TRACE_ROOT}/F2/steps/step_000000/metadata.json")
set(first_step_metadata "${XMVB_TRACE_ROOT}/F2/steps/step_000001/metadata.json")
if (NOT EXISTS "${initial_metadata}" OR NOT EXISTS "${first_step_metadata}")
  message(FATAL_ERROR "F2 TNHVP trace is missing initial or accepted-step metadata")
endif()

file(READ "${initial_metadata}" initial_json)
file(READ "${first_step_metadata}" first_step_json)
if (initial_json MATCHES "\"tnhvp\"")
  message(FATAL_ERROR "Initial-point metadata must not contain TNHVP step data")
endif()

set(required_tnhvp_patterns
  "\"tnhvp\""
  "\"reduced_dimension\": 42"
  "\"hvp_direction_count\": [1-9][0-9]*"
  "\"source_gradient_l2_norm\""
  "\"accepted_gradient_l2_norm\""
  "\"forcing_term\""
  "\"has_kkt_residual\": true"
  "\"initial_trust_radius\""
  "\"accepted_trial_radius\""
  "\"next_trust_radius\""
  "\"predicted_decrease\""
  "\"actual_decrease\""
  "\"trust_ratio\""
  "\"model_spectral_radius\""
  "\"trust_region_shift\""
  "\"encountered_negative_curvature\""
  "\"used_krylov_rescue\""
  "\"reused_krylov_subspace\"")

foreach(pattern IN LISTS required_tnhvp_patterns)
  if (NOT first_step_json MATCHES "${pattern}")
    message(FATAL_ERROR
      "F2 TNHVP trace is missing required pattern: ${pattern}")
  endif()
endforeach()

if (NOT EXISTS "${tnhvp_trace}")
  message(FATAL_ERROR "F2 lightweight TNHVP trace was not written")
endif()
file(READ "${tnhvp_trace}" tnhvp_table)
if (NOT tnhvp_table MATCHES "^iteration" OR
    NOT tnhvp_table MATCHES "reduced_dimension" OR
    NOT tnhvp_table MATCHES "reused_krylov_subspace")
  message(FATAL_ERROR "F2 lightweight TNHVP trace has an invalid header")
endif()
if (NOT tnhvp_table MATCHES "\n1" OR NOT tnhvp_table MATCHES "42")
  message(FATAL_ERROR "F2 lightweight TNHVP trace is missing its first step")
endif()
