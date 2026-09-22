if (NOT DEFINED XMVB_EXE OR NOT DEFINED XMVB_INPUT OR NOT DEFINED TRACE_FILE)
  message(FATAL_ERROR "NEO PySCF reference check requires XMVB_EXE, XMVB_INPUT, and TRACE_FILE")
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    OMP_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=1
    GOTO_NUM_THREADS=1
    MKL_NUM_THREADS=1
    "${XMVB_EXE}" "${XMVB_INPUT}"
    --optimizer-backend neo
    --eigensolver davidson
    --max-iterations 10
    --gradient-tolerance 1e-3
    --energy-tolerance 1e-7
    --neo-trace "${TRACE_FILE}"
  RESULT_VARIABLE run_status
  OUTPUT_VARIABLE report
  ERROR_VARIABLE run_error)

if (NOT run_status EQUAL 0)
  message(FATAL_ERROR "matched-seed F2 NEO run failed: ${run_error}")
endif()
if (NOT report MATCHES "VBSCF converged in[ ]+[34] iterations")
  message(FATAL_ERROR
    "matched-seed F2 NEO no longer converges in at most four macro iterations")
endif()
if (NOT report MATCHES "Final total energy[ ]+:[ ]+-198[.]761111[0-9]+")
  message(FATAL_ERROR "matched-seed F2 NEO energy no longer agrees with the PySCF CASSCF root")
endif()

file(STRINGS "${TRACE_FILE}" trace_lines)
list(POP_FRONT trace_lines trace_header)
set(total_actions 0)
set(orbital_hvp_actions 0)
foreach (line IN LISTS trace_lines)
  string(REPLACE "\t" ";" fields "${line}")
  list(GET fields 4 actions)
  list(GET fields 5 orbital_actions)
  math(EXPR total_actions "${total_actions} + ${actions}")
  math(EXPR orbital_hvp_actions
    "${orbital_hvp_actions} + ${orbital_actions}")
endforeach()
if (total_actions GREATER 30)
  message(FATAL_ERROR
    "matched-seed F2 NEO used ${total_actions} coupled actions; expected at most 30")
endif()
if (orbital_hvp_actions GREATER 25)
  message(FATAL_ERROR
    "matched-seed F2 NEO used ${orbital_hvp_actions} orbital HVPs; expected at most 25")
endif()
