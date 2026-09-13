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
  set(${solver}_iterations "${CMAKE_MATCH_1}")
  string(
    REGEX MATCH
    "Total Energy:[ ]+(-?[0-9]+\\.[0-9]+)"
    ${solver}_energy_match
    "${${solver}_report}")
  set(${solver}_energy "${CMAKE_MATCH_1}")
endforeach()

if (NOT davidson_iterations STREQUAL dense_iterations)
  message(FATAL_ERROR
    "${XMVB_INPUT}: iteration mismatch: Davidson=${davidson_iterations}, dense=${dense_iterations}")
endif()
if (NOT davidson_energy STREQUAL dense_energy)
  message(FATAL_ERROR
    "${XMVB_INPUT}: energy mismatch: Davidson=${davidson_energy}, dense=${dense_energy}")
endif()
