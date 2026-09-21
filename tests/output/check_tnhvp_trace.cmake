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
    --verbose false
    --gradient-tolerance 1e20
  RESULT_VARIABLE iscf5_status
  OUTPUT_VARIABLE iscf5_report
  ERROR_VARIABLE iscf5_errors)
if (NOT iscf5_status EQUAL 0)
  message(FATAL_ERROR
    "ISCF=5 selection run failed with status ${iscf5_status}:\n${iscf5_errors}")
endif()
if (NOT iscf5_report MATCHES
    "VBSCF algorithm: L-BFGS [(]XMVB-compatible raw-coordinate method[)]")
  message(FATAL_ERROR "ISCF=5 did not select XMVB-compatible L-BFGS")
endif()

string(REPLACE "ISCF=7" "ISCF=8" iscf8_input_text "${iscf5_input_text}")
# The preceding replacement starts from an ISCF=5 deck.
string(REPLACE "ISCF=5" "ISCF=8" iscf8_input_text "${iscf8_input_text}")
set(iscf8_input "${XMVB_TRACE_ROOT}/F2_iscf8.xmi")
file(WRITE "${iscf8_input}" "${iscf8_input_text}")
execute_process(
  COMMAND
    "${XMVB_EXECUTABLE}"
    "${iscf8_input}"
    --gradient-tolerance 1e20
    --verbose false
  RESULT_VARIABLE iscf8_status
  OUTPUT_VARIABLE iscf8_report
  ERROR_VARIABLE iscf8_errors)
if (NOT iscf8_status EQUAL 0)
  message(FATAL_ERROR
    "ISCF=8 selection run failed with status ${iscf8_status}:\n${iscf8_errors}")
endif()
if (NOT iscf8_report MATCHES
    "VBSCF algorithm: block-LBFGS [(]nonredundant orbital-block inverse[)]")
  message(FATAL_ERROR "ISCF=8 did not select block-LBFGS")
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
  "\"curvature_subspace_dimension\": [0-9]+"
  "\"exact_hvp_block_actions\": [0-9]+"
  "\"structure_response_block_actions\": [0-9]+"
  "\"response_low_rank_model_rank\": [0-9]+"
  "\"response_low_rank_effective_rank\": [0-9]"
  "\"response_low_rank_rank_90\": [0-9]+"
  "\"response_low_rank_rank_99\": [0-9]+"
  "\"response_low_rank_top_10_fraction\": [0-9]"
  "\"preconditioner_history_size\": [0-9]+"
  "\"outer_iteration_wall_time_seconds\": [0-9]"
  "\"source_gradient_l2_norm\""
  "\"accepted_gradient_l2_norm\""
  "\"forcing_term\""
  "\"model_kkt_relative_residual\""
  "\"max_structure_response_relative_residual\""
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
  "\"encountered_negative_curvature\"")

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
    "model_spectral_radius"
    "coupled_orbital_subspace_dimension"
    "coupled_response_subspace_dimension"
    "coupled_orbital_hessian_block_actions")
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
    NOT tnhvp_table MATCHES "curvature_subspace_dimension" OR
    NOT tnhvp_table MATCHES "exact_hvp_block_actions" OR
    NOT tnhvp_table MATCHES "structure_response_block_actions" OR
    NOT tnhvp_table MATCHES "response_low_rank_effective_rank" OR
    NOT tnhvp_table MATCHES "response_low_rank_top_10_fraction" OR
    NOT tnhvp_table MATCHES "accepted_point_setup_seconds" OR
    NOT tnhvp_table MATCHES "exact_hvp_seconds" OR
    NOT tnhvp_table MATCHES "outer_response_seconds" OR
    NOT tnhvp_table MATCHES "trial_objective_seconds" OR
    NOT tnhvp_table MATCHES "model_kkt_relative_residual" OR
    NOT tnhvp_table MATCHES "max_structure_response_relative_residual" OR
    NOT tnhvp_table MATCHES "minimum_shifted_ritz_value")
  message(FATAL_ERROR "F2 lightweight TNHVP trace has an invalid header")
endif()
if (NOT tnhvp_table MATCHES "\n1" OR NOT tnhvp_table MATCHES "42")
  message(FATAL_ERROR "F2 lightweight TNHVP trace is missing its first step")
endif()

# The production TNHVP path enriches the transported L-BFGS direction with an
# exact relaxed HVP. These counters guard the second-order and structure-
# response work independently.
