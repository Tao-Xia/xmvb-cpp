set(XMVB_TNVB_PYTHON "" CACHE FILEPATH "Configured Block2 Python interpreter")
set(XMVB_TNVB_BLOCK2_MODULE "" CACHE PATH "Block2 compiled Python module directory")
set(XMVB_TNVB_BLOCK2_SOURCE "" CACHE PATH "Directory containing pyblock2")
set(XMVB_TNVB_LIBRARY_DIR "" CACHE PATH "Matched GCC C++/OpenMP runtime directory")
foreach(dependency PYTHON BLOCK2_MODULE BLOCK2_SOURCE LIBRARY_DIR)
  if (NOT EXISTS "${XMVB_TNVB_${dependency}}")
    message(FATAL_ERROR "XMVB_ENABLE_TNVB requires XMVB_TNVB_${dependency}")
  endif()
endforeach()

set(XMVB_TNVB_WORKER "${CMAKE_BINARY_DIR}/libexec/xmvb-cpp/tnvb/evaluate.py")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/libexec/xmvb-cpp/tnvb")
configure_file(worker/evaluate.py "${XMVB_TNVB_WORKER}" COPYONLY)
configure_file(worker/pair_core.py "${CMAKE_BINARY_DIR}/libexec/xmvb-cpp/tnvb/pair_core.py" COPYONLY)
configure_file(worker/hs_shared.py "${CMAKE_BINARY_DIR}/libexec/xmvb-cpp/tnvb/hs_shared.py" COPYONLY)
foreach(module davidson common_ci fixed_family shared_effective forward_state rdm_full)
  configure_file(worker/${module}.py "${CMAKE_BINARY_DIR}/libexec/xmvb-cpp/tnvb/${module}.py" COPYONLY)
endforeach()
configure_file(config.hpp.in "${CMAKE_CURRENT_BINARY_DIR}/tnvb/config.hpp" @ONLY)

add_library(xmvb_tnvb STATIC
  communication/process.cpp
  electronic/evaluator.cpp
  gradient/adapter.cpp)
target_include_directories(xmvb_tnvb PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/.."
                                         "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(xmvb_tnvb PUBLIC xmvb_vbscf)
install(TARGETS xmvb_tnvb ARCHIVE DESTINATION lib)
install(FILES worker/evaluate.py worker/pair_core.py worker/hs_shared.py DESTINATION libexec/xmvb-cpp/tnvb)
install(FILES worker/davidson.py worker/common_ci.py worker/fixed_family.py worker/shared_effective.py worker/forward_state.py worker/rdm_full.py
        DESTINATION libexec/xmvb-cpp/tnvb)

if (BUILD_TESTING)
  add_test(NAME tnvb_worker_surface COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_worker_surface.py")
  add_test(NAME tnvb_orthogonal_frame COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_orthogonal_frame.py")
  add_test(NAME tnvb_full_rdm COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_full_rdm.py")
  add_test(NAME tnvb_forward_state COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_forward_state.py")
  add_executable(handoff_tnvb "${CMAKE_SOURCE_DIR}/tests/tnvb/handoff.cpp")
  target_link_libraries(handoff_tnvb PRIVATE xmvb_tnvb xmvb_input)
  add_test(NAME tnvb_davidson COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_davidson.py")
  add_test(NAME tnvb_shared_hs_policy COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_shared_hs_policy.py")
  add_test(NAME tnvb_rdm_policy COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_rdm_policy.py")
  add_test(NAME tnvb_acceptance_summary COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_acceptance_summary.py")
  add_test(NAME tnvb_cli COMMAND "${XMVB_TNVB_PYTHON}"
    "${CMAKE_SOURCE_DIR}/tests/tnvb/test_cli.py" "$<TARGET_FILE:xmvb>"
    "${CMAKE_SOURCE_DIR}/testdata/tnvb/C6_random5_hao.xmi")
  add_executable(test_tnvb_trials "${CMAKE_SOURCE_DIR}/tests/tnvb/test_trials.cpp")
  target_link_libraries(test_tnvb_trials PRIVATE xmvb_tnvb xmvb_input)
  add_test(NAME tnvb_trials COMMAND test_tnvb_trials "${CMAKE_SOURCE_DIR}/testdata/tnvb/C6_random5_hao.xmi")
  add_test(NAME tnvb_davidson_trials COMMAND test_tnvb_trials "${CMAKE_SOURCE_DIR}/testdata/tnvb/C6_random5_hao.xmi" davidson)
  set_tests_properties(tnvb_trials tnvb_davidson_trials PROPERTIES ENVIRONMENT
    "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;XMVB_BLOCK2_WORKERS=1;SLURM_CPUS_PER_TASK=2")
  add_executable(export_tnvb_request EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tests/tnvb/export_request.cpp")
  target_link_libraries(export_tnvb_request PRIVATE xmvb_input)
  add_executable(test_tnvb_input "${CMAKE_SOURCE_DIR}/tests/tnvb/test_input.cpp")
  target_link_libraries(test_tnvb_input PRIVATE xmvb_input)
  add_test(NAME tnvb_input COMMAND test_tnvb_input "${CMAKE_SOURCE_DIR}/testdata/tnvb/C6_random5_hao.xmi")
  add_executable(optimize_tnvb EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tests/tnvb/optimization.cpp")
  target_link_libraries(optimize_tnvb PRIVATE xmvb_tnvb xmvb_input)
  add_executable(check_tnvb_directions EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tests/tnvb/directions.cpp")
  target_link_libraries(check_tnvb_directions PRIVATE xmvb_tnvb xmvb_input)
  add_executable(prepare_tnvb_oeo EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tests/tnvb/prepare_oeo.cpp")
  target_link_libraries(prepare_tnvb_oeo PRIVATE xmvb_input)
  add_executable(compare_tnvb EXCLUDE_FROM_ALL "${CMAKE_SOURCE_DIR}/tests/tnvb/compare.cpp")
  target_link_libraries(compare_tnvb PRIVATE xmvb_tnvb xmvb_input)
  add_executable(test_tnvb_adapter "${CMAKE_SOURCE_DIR}/tests/tnvb/test_adapter.cpp")
  target_link_libraries(test_tnvb_adapter PRIVATE xmvb_tnvb)
  add_test(NAME tnvb_adapter COMMAND test_tnvb_adapter)
  # Numerical acceptance binaries can be snapshotted beside an installed
  # worker. Match their parent C++/OpenMP runtime too, not a conda fallback.
  get_filename_component(_tnvb_lapack_runtime_dir "${LAPACK_LIB}" DIRECTORY)
  foreach(tool test_tnvb_trials export_tnvb_request test_tnvb_input optimize_tnvb
               check_tnvb_directions prepare_tnvb_oeo compare_tnvb test_tnvb_adapter handoff_tnvb)
    set_target_properties(${tool} PROPERTIES
      INSTALL_RPATH "${XMVB_TNVB_LIBRARY_DIR};${_tnvb_lapack_runtime_dir}"
      BUILD_WITH_INSTALL_RPATH TRUE)
  endforeach()
endif()
