if (NOT DEFINED XMVB_EXECUTABLE OR NOT DEFINED XMVB_INPUT)
  message(FATAL_ERROR "XMVB_EXECUTABLE and XMVB_INPUT are required")
endif()

foreach(solver IN ITEMS davidson dense)
  execute_process(
    COMMAND "${XMVB_EXECUTABLE}" "${XMVB_INPUT}" --eigensolver "${solver}"
    RESULT_VARIABLE ${solver}_status
    OUTPUT_VARIABLE ${solver}_report
    ERROR_VARIABLE ${solver}_errors)
  if (NOT ${solver}_status EQUAL 0)
    message(FATAL_ERROR
      "${XMVB_INPUT}: ${solver} run failed with status ${${solver}_status}:\n${${solver}_errors}")
  endif()
  if (NOT ${solver}_report MATCHES
      "Structure eigensolver[ ]+:[ ]+${solver}")
    message(FATAL_ERROR "${XMVB_INPUT}: report did not select ${solver}")
  endif()
  string(
    REGEX MATCH
    "VBSCF converged in[ ]+([0-9]+) iterations"
    ${solver}_iteration_match
    "${${solver}_report}")
  if ("${${solver}_iteration_match}" STREQUAL "")
    message(FATAL_ERROR "${XMVB_INPUT}: ${solver} run did not converge")
  endif()
  set(${solver}_iterations "${CMAKE_MATCH_1}")
  string(
    REGEX MATCH
    "Total Energy:[ ]+(-?[0-9]+\\.[0-9]+)"
    ${solver}_energy_match
    "${${solver}_report}")
  if ("${${solver}_energy_match}" STREQUAL "")
    message(FATAL_ERROR "${XMVB_INPUT}: ${solver} report has no total energy")
  endif()
  set(${solver}_energy "${CMAKE_MATCH_1}")
endforeach()

# Dense and Davidson evaluate the same matrix-free model to the requested
# accuracy, but floating-point Ritz vectors can select different equally valid
# trust-region steps.  Convergence and final energy are invariants; the exact
# number of accepted outer steps is not.
string(
  REGEX MATCH
  "^-?[0-9]+\\.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]"
  davidson_energy_seven_decimals
  "${davidson_energy}")
string(
  REGEX MATCH
  "^-?[0-9]+\\.[0-9][0-9][0-9][0-9][0-9][0-9][0-9]"
  dense_energy_seven_decimals
  "${dense_energy}")
if (NOT davidson_energy_seven_decimals STREQUAL dense_energy_seven_decimals)
  message(FATAL_ERROR
    "${XMVB_INPUT}: energy mismatch: Davidson=${davidson_energy}, dense=${dense_energy}")
endif()
