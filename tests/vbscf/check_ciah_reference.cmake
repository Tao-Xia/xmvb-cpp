if (NOT DEFINED XMVB_EXE OR NOT DEFINED XMVB_INPUT OR NOT DEFINED TRACE_FILE)
  message(FATAL_ERROR "CIAH PySCF reference check requires XMVB_EXE, XMVB_INPUT, and TRACE_FILE")
endif()

execute_process(
  COMMAND
    "${CMAKE_COMMAND}" -E env
    --unset=XMVB_CIAH_SMALL_CI
    --unset=XMVB_CIAH_SPLIT_BUDGET
    OMP_NUM_THREADS=4
    OPENBLAS_NUM_THREADS=1
    GOTO_NUM_THREADS=1
    MKL_NUM_THREADS=1
    "${XMVB_EXE}" "${XMVB_INPUT}"
    --optimizer-backend ciah
    --eigensolver davidson
    --max-iterations 10
    --gradient-tolerance 1e-3
    --energy-tolerance 1e-7
    --ciah-trace "${TRACE_FILE}"
  RESULT_VARIABLE run_status
  OUTPUT_VARIABLE report
  ERROR_VARIABLE run_error)

if (NOT run_status EQUAL 0)
  message(FATAL_ERROR "matched-seed F2 CIAH run failed: ${run_error}")
endif()
if (NOT report MATCHES "Orbital coordinates[ ]+:[ ]+unwhitened quotient")
  message(FATAL_ERROR "CIAH must use the unwhitened quotient by default")
endif()
if (NOT report MATCHES "CIAH response model[ ]+:[ ]+adaptive CI")
  message(FATAL_ERROR "eligible FULL-OEO CIAH must use adaptive CI without environment switches")
endif()
# The report counts every accepted exact-gradient keyframe.  The fifth point is
# the independent energy confirmation after projected-gradient convergence;
# older grouped-macro reporting hid that point inside the preceding macro.
if (NOT report MATCHES "VBSCF converged in[ ]+[345] iterations")
  message(FATAL_ERROR
    "matched-seed F2 CIAH no longer converges in at most five accepted keyframes")
endif()
if (NOT report MATCHES "Final total energy[ ]+:[ ]+-198[.]761111[0-9]+")
  message(FATAL_ERROR "matched-seed F2 CIAH energy no longer agrees with the PySCF CASSCF root")
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
    "matched-seed F2 CIAH used ${total_actions} coupled actions; expected at most 30")
endif()
if (orbital_hvp_actions GREATER 25)
  message(FATAL_ERROR
    "matched-seed F2 CIAH used ${orbital_hvp_actions} orbital HVPs; expected at most 25")
endif()
