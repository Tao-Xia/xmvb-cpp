# End-to-end regression: a linear-only predictor must not update Newton trust.
file(MAKE_DIRECTORY "${RUN_DIR}")
configure_file("${XMVB_INPUT}" "${RUN_DIR}/input.xmi" COPYONLY)
file(CREATE_LINK "${XMVB_BASIS}" "${RUN_DIR}/basis" SYMBOLIC)
execute_process(
  COMMAND "${XMVB_EXECUTABLE}" "${RUN_DIR}/input.xmi"
    --optimizer-backend nonredundant_truncated_newton --eigensolver dense
    --tnhvp-trace "${RUN_DIR}/tnhvp.tsv"
  WORKING_DIRECTORY "${RUN_DIR}"
  RESULT_VARIABLE status
  OUTPUT_FILE "${RUN_DIR}/run.out"
  ERROR_FILE "${RUN_DIR}/run.err")
if(NOT status EQUAL 0)
  message(FATAL_ERROR "TNHVP observation regression did not converge: ${status}")
endif()
file(STRINGS "${RUN_DIR}/tnhvp.tsv" rows)
list(POP_FRONT rows header)
string(REPLACE "\t" ";" columns "${header}")
foreach(name IN ITEMS newton_trial_evaluated accepted_newton_step
    initial_trust_radius next_trust_radius predicted_decrease
    accepted_trial_radius trust_ratio)
  list(FIND columns "${name}" ${name}_index)
  if(${name}_index LESS 0)
    message(FATAL_ERROR "Missing trace column: ${name}")
  endif()
endforeach()
set(predictor_count 0)
foreach(row IN LISTS rows)
  string(REPLACE "\t" ";" values "${row}")
  foreach(name IN ITEMS newton_trial_evaluated accepted_newton_step
      initial_trust_radius next_trust_radius predicted_decrease
      accepted_trial_radius trust_ratio)
    list(GET values ${${name}_index} ${name})
  endforeach()
  if(NOT newton_trial_evaluated)
    math(EXPR predictor_count "${predictor_count} + 1")
    if(NOT initial_trust_radius STREQUAL next_trust_radius)
      message(FATAL_ERROR "An Armijo-only step changed the Newton trust radius")
    endif()
  endif()
  if(NOT accepted_newton_step)
    if(NOT predicted_decrease EQUAL 0 OR NOT accepted_trial_radius EQUAL 0
        OR NOT trust_ratio EQUAL 0)
      message(FATAL_ERROR "An Armijo predictor was mislabeled as a quadratic trial")
    endif()
  endif()
endforeach()
if(predictor_count EQUAL 0)
  message(FATAL_ERROR "Regression did not exercise an Armijo-only step")
endif()
