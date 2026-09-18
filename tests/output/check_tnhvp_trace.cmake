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
    --dump-trace-dir "${XMVB_TRACE_ROOT}"
    --tnhvp-trace "${tnhvp_trace}"
  RESULT_VARIABLE xmvb_status
  OUTPUT_VARIABLE xmvb_report
  ERROR_VARIABLE xmvb_errors)

if (NOT xmvb_status EQUAL 0)
  message(FATAL_ERROR
    "F2 TNHVP trace run failed with status ${xmvb_status}:\n${xmvb_errors}")
endif()

if (NOT xmvb_report MATCHES
    "VBSCF algorithm: TNHVP [(]matrix-free truncated Newton[)]")
  message(FATAL_ERROR "ISCF=7 did not select TNHVP")
endif()

file(READ "${XMVB_INPUT}" iscf5_input_text)
string(REPLACE "ISCF=7" "ISCF=5" iscf5_input_text "${iscf5_input_text}")
set(iscf5_input "${XMVB_TRACE_ROOT}/F2_iscf5.xmi")
file(WRITE "${iscf5_input}" "${iscf5_input_text}")
execute_process(
  COMMAND
    "${XMVB_EXECUTABLE}"
    "${iscf5_input}"
    --gradient-tolerance 1e20
    --verbose false
  RESULT_VARIABLE iscf5_status
  OUTPUT_VARIABLE iscf5_report
  ERROR_VARIABLE iscf5_errors)
if (NOT iscf5_status EQUAL 0)
  message(FATAL_ERROR
    "ISCF=5 selection run failed with status ${iscf5_status}:\n${iscf5_errors}")
endif()
if (NOT iscf5_report MATCHES "VBSCF algorithm: nonredundant L-BFGS")
  message(FATAL_ERROR "ISCF=5 did not select nonredundant L-BFGS")
endif()

string(
  REPLACE
  "EIGENSOLVER=DAVIDSON"
  "EIGENSOLVER=DENSE"
  dense_input_text
  "${iscf5_input_text}")
set(dense_input "${XMVB_TRACE_ROOT}/F2_dense.xmi")
file(WRITE "${dense_input}" "${dense_input_text}")
execute_process(
  COMMAND
    "${XMVB_EXECUTABLE}"
    "${dense_input}"
    --gradient-tolerance 1e20
    --verbose false
  RESULT_VARIABLE dense_status
  OUTPUT_VARIABLE dense_report
  ERROR_VARIABLE dense_errors)
if (NOT dense_status EQUAL 0)
  message(FATAL_ERROR
    "dense eigensolver selection run failed with status ${dense_status}:\n${dense_errors}")
endif()
if (NOT dense_report MATCHES "Structure eigensolver[ ]+:[ ]+dense")
  message(FATAL_ERROR "EIGENSOLVER=DENSE did not select the dense reference")
endif()

set(initial_metadata "${XMVB_TRACE_ROOT}/F2/steps/step_000000/metadata.json")
set(first_step_metadata "${XMVB_TRACE_ROOT}/F2/steps/step_000001/metadata.json")
if (NOT EXISTS "${initial_metadata}" OR NOT EXISTS "${first_step_metadata}")
  message(FATAL_ERROR "F2 TNHVP trace is missing initial or accepted-step metadata")
endif()

foreach(matrix_name IN ITEMS overlap_matrix_f64.bin hamiltonian_matrix_f64.bin)
  set(matrix_path "${XMVB_TRACE_ROOT}/F2/steps/step_000001/${matrix_name}")
  if (EXISTS "${matrix_path}")
    message(FATAL_ERROR
      "F2 Davidson trace unexpectedly materialized ${matrix_name}")
  endif()
endforeach()

file(READ "${initial_metadata}" initial_json)
file(READ "${first_step_metadata}" first_step_json)
if (initial_json MATCHES "\"tnhvp\"")
  message(FATAL_ERROR "Initial-point metadata must not contain TNHVP step data")
endif()

set(required_tnhvp_patterns
  "\"tnhvp\""
  "\"reduced_dimension\": 42"
  "\"coupled_orbital_subspace_dimension\": [1-9][0-9]*"
  "\"coupled_response_subspace_dimension\": [1-9][0-9]*"
  "\"coupled_orbital_hessian_block_actions\": [1-9][0-9]*"
  "\"coupled_orbital_to_response_block_actions\": [1-9][0-9]*"
  "\"coupled_response_to_orbital_block_actions\": [1-9][0-9]*"
  "\"coupled_response_hessian_block_actions\": [1-9][0-9]*"
  "\"coupled_orbital_metric_block_actions\": [1-9][0-9]*"
  "\"preconditioner_history_size\": [0-9]+"
  "\"outer_iteration_wall_time_seconds\": [0-9]"
  "\"source_gradient_l2_norm\""
  "\"accepted_gradient_l2_norm\""
  "\"forcing_term\""
  "\"orbital_backward_error\""
  "\"response_backward_error\""
  "\"initial_trust_radius\""
  "\"accepted_trial_radius\""
  "\"next_trust_radius\""
  "\"linear_decrease\""
  "\"predicted_decrease\""
  "\"actual_decrease\""
  "\"trust_ratio\""
  "\"minimum_ritz_value\""
  "\"minimum_shifted_ritz_value\""
  "\"trust_region_shift\""
  "\"encountered_negative_curvature\""
  "\"reused_subspace\"")

foreach(pattern IN LISTS required_tnhvp_patterns)
  if (NOT first_step_json MATCHES "${pattern}")
    message(FATAL_ERROR
      "F2 TNHVP trace is missing required pattern: ${pattern}")
  endif()
endforeach()

foreach(retired_field IN ITEMS
    "hvp_direction_count"
    "core_hvp_direction_count"
    "response_probe_performed"
    "response_scale_affordable"
    "used_full_hvp"
    "kkt_relative_residual"
    "model_spectral_radius")
  if (first_step_json MATCHES "\"${retired_field}\"")
    message(FATAL_ERROR
      "F2 TNHVP trace retained obsolete field: ${retired_field}")
  endif()
endforeach()

if (NOT EXISTS "${tnhvp_trace}")
  message(FATAL_ERROR "F2 lightweight TNHVP trace was not written")
endif()
file(READ "${tnhvp_trace}" tnhvp_table)
if (NOT tnhvp_table MATCHES "^iteration" OR
    NOT tnhvp_table MATCHES "reduced_dimension" OR
    NOT tnhvp_table MATCHES "coupled_orbital_subspace_dimension" OR
    NOT tnhvp_table MATCHES "coupled_response_subspace_dimension" OR
    NOT tnhvp_table MATCHES "coupled_orbital_hessian_block_actions" OR
    NOT tnhvp_table MATCHES "orbital_backward_error" OR
    NOT tnhvp_table MATCHES "response_backward_error" OR
    NOT tnhvp_table MATCHES "minimum_shifted_ritz_value" OR
    NOT tnhvp_table MATCHES "reused_subspace")
  message(FATAL_ERROR "F2 lightweight TNHVP trace has an invalid header")
endif()
if (NOT tnhvp_table MATCHES "\n1" OR NOT tnhvp_table MATCHES "42")
  message(FATAL_ERROR "F2 lightweight TNHVP trace is missing its first step")
endif()

# The production TNHVP path is now the coupled orbital--structure workspace.
# Its independently counted A/B/B^T/C/G actions above guard against silently
# reverting to the retired orbital-only HVP admission path.
