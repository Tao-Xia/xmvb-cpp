# Fixed-point response regression; the executable enforces numerical identities.
foreach(required IN ITEMS AUDIT_EXECUTABLE XMVB_INPUT XMVB_BASIS
    ORBITAL_FIXTURE TRUST_RADIUS RUN_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Missing response regression option: ${required}")
  endif()
endforeach()
file(MAKE_DIRECTORY "${RUN_DIR}")
file(CREATE_LINK "${XMVB_BASIS}" "${RUN_DIR}/basis" SYMBOLIC)
execute_process(
  COMMAND "${AUDIT_EXECUTABLE}" "${XMVB_INPUT}"
    --orbital-value-table-text "${ORBITAL_FIXTURE}"
    --trust-radius "${TRUST_RADIUS}"
    --eigensolver davidson --audit-operator true
  WORKING_DIRECTORY "${RUN_DIR}"
  RESULT_VARIABLE status
  OUTPUT_FILE "${RUN_DIR}/run.out"
  ERROR_FILE "${RUN_DIR}/run.err")
if(NOT status EQUAL 0)
  file(READ "${RUN_DIR}/run.err" diagnostic)
  message(FATAL_ERROR
    "Fixed-point response regression failed (${status}): ${diagnostic}")
endif()
file(READ "${RUN_DIR}/run.out" output)
if(NOT output MATCHES "response_model_invariants = PASS" OR
    NOT output MATCHES "response_model_probe_count = 3" OR
    NOT output MATCHES "response_model_cached_step_error =")
  message(FATAL_ERROR "Fixed-point response numerical checks were not executed")
endif()
