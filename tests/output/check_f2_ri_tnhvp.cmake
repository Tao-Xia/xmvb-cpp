if (NOT DEFINED XMVB_EXECUTABLE OR
    NOT DEFINED XMVB_INPUT OR
    NOT DEFINED XMVB_SCRATCH)
  message(FATAL_ERROR
    "XMVB_EXECUTABLE, XMVB_INPUT, and XMVB_SCRATCH are required")
endif()

file(READ "${XMVB_INPUT}" exact_input)
string(REPLACE "INT=LIBCINT" "INT=RI" ri_input "${exact_input}")
if (ri_input STREQUAL exact_input)
  message(FATAL_ERROR "F2 reference input does not select INT=LIBCINT")
endif()

file(MAKE_DIRECTORY "${XMVB_SCRATCH}")
set(ri_input_path "${XMVB_SCRATCH}/F2_RI.xmi")
file(WRITE "${ri_input_path}" "${ri_input}")

execute_process(
  COMMAND "${XMVB_EXECUTABLE}" "${ri_input_path}"
  WORKING_DIRECTORY "${XMVB_SCRATCH}"
  RESULT_VARIABLE xmvb_status
  OUTPUT_VARIABLE xmvb_report
  ERROR_VARIABLE xmvb_errors)

if (NOT xmvb_status EQUAL 0)
  message(FATAL_ERROR
    "F2 RI-TNHVP run failed with status ${xmvb_status}:\n${xmvb_errors}")
endif()

foreach(pattern IN ITEMS
    "Initial total energy[ ]+:[ ]+-198[.]488[0-9]+"
    "VBSCF converged in[ ]+[1-8] iterations"
    "Final total energy[ ]+:[ ]+-198[.]7509[0-9]+"
    "Exact HVP block actions[ ]+:[ ]+[1-9][0-9]*")
  if (NOT xmvb_report MATCHES "${pattern}")
    message(FATAL_ERROR
      "F2 RI-TNHVP report is missing required pattern: ${pattern}")
  endif()
endforeach()
