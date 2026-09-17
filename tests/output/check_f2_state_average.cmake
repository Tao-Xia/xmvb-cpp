if (NOT DEFINED XMVB_EXECUTABLE OR NOT DEFINED XMVB_INPUT)
  message(FATAL_ERROR "XMVB_EXECUTABLE and XMVB_INPUT are required")
endif()

execute_process(
  COMMAND "${XMVB_EXECUTABLE}" "${XMVB_INPUT}"
  RESULT_VARIABLE xmvb_status
  OUTPUT_VARIABLE xmvb_report
  ERROR_VARIABLE xmvb_errors)

if (NOT xmvb_status EQUAL 0)
  message(FATAL_ERROR
    "F2 state-average run failed with status ${xmvb_status}:\n${xmvb_errors}")
endif()

foreach(pattern IN ITEMS
    "Number of equally averaged states[ ]+:[ ]+2"
    "EQUAL-WEIGHT STATE-AVERAGED VBSCF"
    "1[ ]+0[.]50000000"
    "2[ ]+0[.]50000000"
    "STATE-AVERAGED TOTAL ENERGY"
    "STATE 1 STRUCTURE ANALYSIS"
    "STATE 2 STRUCTURE ANALYSIS")
  if (NOT xmvb_report MATCHES "${pattern}")
    message(FATAL_ERROR
      "F2 state-average report is missing required pattern: ${pattern}")
  endif()
endforeach()
