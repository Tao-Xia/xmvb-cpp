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
    "F2 text-report run failed with status ${xmvb_status}:\n${xmvb_errors}")
endif()

set(required_report_patterns
  "Developed by: Tao Xia"
  "Software development assistance: OpenAI Codex"
  "VBSCF algorithm: nonredundant_truncated_newton"
  "Structure eigensolver[ ]+:[ ]+davidson"
  "Full structure Hamiltonian and overlap matrices are omitted"
  "COEFFICIENTS OF DETERMINANTS WITHOUT NORMALIZED"
  "Coulson-Chirgwin Weights"
  "Renormalized Weights"
  "Lowdin and inverse weights are omitted"
  "ORBITALS IN PRIMITIVE BASIS FUNCTIONS"
  "[ ]PX[ ]"
  "COMPUTED NATURAL ORBITALS"
  "1\\.87[0-9]+[ ]+0\\.12[0-9]+"
  "POPULATION AND CHARGE"
  "F[ ]+9\\.000000"
  "BOND ORDER"
  "1\\.400[ ]+0\\.773"
  "VALENCE ANALYSIS"
  "F[ ]+1\\.000[ ]+0\\.773[ ]+0\\.227"
  "DIPOLE MOMENT ANALYSIS"
  "0\\.000000[ ]+0\\.000000[ ]+-?0\\.00000[0-9]"
  "VIRIAL THEOREM ANALYSIS"
  "VIRIAL THEOREM VALUE[ ]+:[ ]+1\\.999")

foreach(pattern IN LISTS required_report_patterns)
  if (NOT xmvb_report MATCHES "${pattern}")
    message(FATAL_ERROR
      "F2 report is missing required pattern: ${pattern}")
  endif()
endforeach()

foreach(forbidden_pattern IN ITEMS
    "OVERLAP OF VB STRUCTURES"
    "HAMILTONIAN OF VB STRUCTURES"
    "Lowdin Weights"
    "Inverse Weights")
  if (xmvb_report MATCHES "${forbidden_pattern}")
    message(FATAL_ERROR
      "F2 Davidson report unexpectedly contains: ${forbidden_pattern}")
  endif()
endforeach()

execute_process(
  COMMAND "${XMVB_EXECUTABLE}" "${XMVB_INPUT}" --eigensolver dense
  RESULT_VARIABLE dense_status
  OUTPUT_VARIABLE dense_report
  ERROR_VARIABLE dense_errors)
if (NOT dense_status EQUAL 0)
  message(FATAL_ERROR
    "F2 dense report run failed with status ${dense_status}:\n${dense_errors}")
endif()
foreach(dense_pattern IN ITEMS
    "OVERLAP OF VB STRUCTURES"
    "HAMILTONIAN OF VB STRUCTURES"
    "Lowdin Weights"
    "Inverse Weights")
  if (NOT dense_report MATCHES "${dense_pattern}")
    message(FATAL_ERROR "F2 dense report is missing: ${dense_pattern}")
  endif()
endforeach()
