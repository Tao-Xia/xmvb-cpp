file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" XMVB_CONDA_PREFIX)

option(
  XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
  "Enable the legacy Fortran L-BFGS backend inside the C++ VB optimizer."
  OFF)

option(
  XMVB_CPP_BUILD_DEV_TARGETS
  "Build C++ comparison, diagnostic, and test targets in addition to xmvb-cpp.exe."
  ON)

find_path(EIGEN3_INCLUDE_DIR Eigen/Core
  HINTS
    ${EIGEN3_ROOT_DIR}
    ${EIGEN3_ROOT_DIR}/include
    ${EIGEN3_ROOT_DIR}/include/eigen3
    ${XMVB_CONDA_PREFIX}
    ${XMVB_CONDA_PREFIX}/include
    ${XMVB_CONDA_PREFIX}/include/eigen3
    /usr/include/eigen3
    /usr/local/include/eigen3
  PATH_SUFFIXES
    eigen3
    include/eigen3
  REQUIRED)

set(XMVB_CPP_CORE_SOURCES
  vb/matrices/biorthogonal_spin_pair.cpp
  core/linear_algebra/generalized_eigensolver.cpp
  vb/matrices/determinant_hamiltonian_resolver.cpp
  vb/matrices/determinant_overlap_resolver.cpp
  vb/matrices/full_determinant_structure_expander.cpp
  vb/matrices/full_determinant_structure_hamiltonian_overlap_builder.cpp
  vb/matrices/hamiltonian_overlap_builder.cpp
  vb/matrices/structure_matrix_evaluator.cpp
  vb/matrices/structure_subspace_builder.cpp
  vb/matrices/structure_hamiltonian_overlap_builder.cpp
  vb/matrices/structure_pair_accumulator.cpp
  vb/matrices/two_electron_indexer.cpp
  vb/model/deepvbh_jax_inference_runner.cpp
  vb/orbital/active_space_matrix_backpropagator.cpp
  vb/orbital/active_space_one_electron_builder.cpp
  vb/orbital/active_space_orbital_backpropagator.cpp
  vb/orbital/active_space_orbital_preparer.cpp
  vb/orbital/active_space_two_electron_backpropagator.cpp
  vb/orbital/active_space_two_electron_builder.cpp
  vb/orbital/ao_effective_one_electron_backpropagator.cpp
  vb/orbital/ao_effective_one_electron_builder.cpp
  vb/orbital/legacy_style_orbital_gradient_projector.cpp
  vb/orbital/orbital_parameter_codec.cpp
  vb/scf/cpp_active_space_gradient_evaluator.cpp
  vb/scf/deepvbh_onnx_direct_final_optimizer.cpp
  vb/scf/deepvbh_onnx_hybrid_optimizer.cpp
  vb/scf/cpp_orbital_gradient_evaluator.cpp
  vb/scf/cpp_vb_scf_evaluator.cpp
  vb/scf/cpp_vb_scf_optimizer.cpp
  vb/scf/vb_scf_evaluator.cpp)

set(XMVB_CPP_RUNTIME_SOURCES
  runtime/cpp_vb_input_loader.cpp
  runtime_c/local_runtime_api.c
  runtime_c/local_runtime/cal_1eint.c
  runtime_c/local_runtime/cal_2eint.c
  runtime_c/local_runtime/char_util.c
  runtime_c/local_runtime/checkkeywords.c
  runtime_c/local_runtime/check_dir2e.c
  runtime_c/local_runtime/cvitra.c
  runtime_c/local_runtime/data.c
  runtime_c/local_runtime/data_struct.c
  runtime_c/local_runtime/del_dfvb_str.c
  runtime_c/local_runtime/del_vb_str.c
  runtime_c/local_runtime/detect_blocks.c
  runtime_c/local_runtime/detect_guess.c
  runtime_c/local_runtime/detect_ndb.c
  runtime_c/local_runtime/diis.c
  runtime_c/local_runtime/dft.c
  runtime_c/local_runtime/dftd3_disp.c
  runtime_c/local_runtime/fvbsec.c
  runtime_c/local_runtime/genstr.c
  runtime_c/local_runtime/get_total_mem.c
  runtime_c/local_runtime/getfrg.c
  runtime_c/local_runtime/getorb.c
  runtime_c/local_runtime/getpoints.c
  runtime_c/local_runtime/getstr.c
  runtime_c/local_runtime/hf.c
  runtime_c/local_runtime/hgp_3c2e.c
  runtime_c/local_runtime/init_bovb.c
  runtime_c/local_runtime/init_vb_param.c
  runtime_c/cpp_runtime_extractor.c
  runtime_c/local_runtime/int_data_trans.c
  runtime_c/local_runtime/lab.c
  runtime_c/local_runtime/mol.c
  runtime_c/local_runtime/coulomb.c
  runtime_c/local_runtime/exchange.c
  runtime_c/local_runtime/boys.c
  runtime_c/local_runtime/global.c
  runtime_c/local_runtime/nb3idx.c
  runtime_c/local_runtime/normalize.c
  runtime_c/local_runtime/pair.c
  runtime_c/local_runtime/print_crd.c
  runtime_c/local_runtime/print_init_guess.c
  runtime_c/local_runtime/print_vb_comp_info.c
  runtime_c/local_runtime/readinp.c
  runtime_c/local_runtime/readint.c
  runtime_c/local_runtime/readint_head.c
  runtime_c/local_runtime/readorb.c
  runtime_c/local_runtime/readinp_vb.c
  runtime_c/local_runtime/readstr.c
  runtime_c/local_runtime/sad.c
  runtime_c/local_runtime/split_string.c
  runtime_c/local_runtime/vb_autoguess.c
  runtime_c/local_runtime/vbguess.c
  runtime_c/local_runtime/vb_readguess.c
  runtime_c/local_runtime/vb_unitguess.c
  runtime_c/local_runtime/value_ao.c
  runtime_c/local_runtime/vbprep.c
  runtime_c/local_runtime/build_g1d.c
  runtime_c/local_runtime/getvars.c
  runtime_c/local_runtime/xint.c
  runtime_c/local_runtime/xint_2c1e.c
  runtime_c/local_runtime/xint_2c2e.c
  runtime_c/local_runtime/xint_3c1e.c
  runtime_c/local_runtime/xint_3c2e.c
  runtime_c/local_runtime/xint_hrr.c
  runtime_c/local_runtime/xint_overlap.c
  runtime_c/local_runtime/xgrids.c
)

set(XMVB_CPP_TEST_SUPPORT_SOURCES
  tests/support/ao_integral_data_loader.cpp
  tests/support/ao_integral_extractor.c
  tests/support/cpp_vb_input_data_loader.cpp
  tests/support/legacy_full_structure_data_loader.cpp
  tests/support/legacy_full_structure_matrix_extractor.c
  tests/support/legacy_determinant_pair_extractor.c
  tests/support/legacy_gradient_layers_data_loader.cpp
  tests/support/legacy_gradient_layers_extractor.c
  tests/support/legacy_orbital_gradient_data_loader.cpp
  tests/support/legacy_orbital_gradient_extractor.c
  tests/support/legacy_raw_structure_extractor.c
  tests/support/orbital_preparation_data_loader.cpp
  tests/support/orbital_preparation_extractor.c
  tests/support/raw_structure_data_loader.cpp
  tests/support/legacy_determinant_pair_test_helper.F90
  tests/support/legacy_gradient_layers_helper.F90)

set(XMVB_CPP_ORIGINAL_RUNTIME_STAGING_SOURCES
  runtime_c/original_runtime/guess/vbguess.c
  runtime_c/original_runtime/inpout/checkkeywords.c
  runtime_c/original_runtime/inpout/readinp.c
  runtime_c/original_runtime/scf/hf.c
  runtime_c/original_runtime/vbint/cal_1eint.c
  runtime_c/original_runtime/vbint/cal_2eint.c
  runtime_c/original_runtime/vbint/int_data_trans.c
  runtime_c/original_runtime/vbutil/readinp_vb.c
  runtime_c/original_runtime/vbutil/vbprep.c)

add_library(xmvb_cpp_openmp INTERFACE)
target_link_libraries(xmvb_cpp_openmp
  INTERFACE
    OpenMP::OpenMP_C
    OpenMP::OpenMP_CXX)

add_library(xmvb_cpp_vb STATIC ${XMVB_CPP_CORE_SOURCES})
set_target_properties(xmvb_cpp_vb PROPERTIES OUTPUT_NAME xmvb-vbscf)
target_include_directories(xmvb_cpp_vb
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/LBFGSpp/include
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${LIBXC_INCLUDE}
    ${CINT_INCLUDE})
if (XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND=1)
endif()
if (XMVB_CPP_ENABLE_ONNX_RUNTIME)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC XMVB_CPP_ENABLE_ONNX_RUNTIME=1)
  target_include_directories(
    xmvb_cpp_vb
    PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(
    xmvb_cpp_vb
    PUBLIC ${ONNXRUNTIME_LIB})
endif()
target_link_libraries(xmvb_cpp_vb PUBLIC ${LAPACKE_LIB} xmvb_cpp_openmp)
if (LAPACK_TYPE STREQUAL "OpenBLAS")
  target_link_libraries(xmvb_cpp_vb PUBLIC ${LAPACK_LIB})
elseif (LAPACK_TYPE STREQUAL "refBLAS")
  target_link_libraries(xmvb_cpp_vb PUBLIC lapacke cblas lapack)
else()
  target_link_libraries(xmvb_cpp_vb PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

set(
  XMVB_CPP_RUNTIME_EXECUTABLE_PATH
  "${CMAKE_BINARY_DIR}/src/driver/xmvb.exe"
  CACHE FILEPATH
  "Executable path used by the standalone runtime to locate basis/data assets.")

add_library(xmvb_cpp_runtime STATIC ${XMVB_CPP_RUNTIME_SOURCES})
target_compile_definitions(xmvb_cpp_runtime
  PRIVATE
    PATH_MAX=4096
    XMVB_CPP_LEGACY_EXECUTABLE_PATH=\"${XMVB_CPP_RUNTIME_EXECUTABLE_PATH}\")
target_include_directories(xmvb_cpp_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/src/vbutil
    ${CMAKE_BINARY_DIR}/src/vbscf
    ${CMAKE_BINARY_DIR}/src/vbci
    ${CMAKE_BINARY_DIR}/src/vbpt2
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${LIBXC_INCLUDE}
    ${CINT_INCLUDE})
target_link_libraries(xmvb_cpp_runtime PUBLIC xmvb_cpp_vb xmvb_cpp_openmp)

if (XMVB_CPP_BUILD_DEV_TARGETS)
  add_library(xmvb_cpp_test_support STATIC ${XMVB_CPP_TEST_SUPPORT_SOURCES})
  target_compile_definitions(xmvb_cpp_test_support PRIVATE PATH_MAX=4096)
  target_include_directories(xmvb_cpp_test_support
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_BINARY_DIR}/src/vbutil
      ${CMAKE_BINARY_DIR}/src/vbscf
      ${CMAKE_BINARY_DIR}/src/vbci
      ${CMAKE_BINARY_DIR}/src/vbpt2
      ${EIGEN3_INCLUDE_DIR}
      ${LAPACK_INCLUDE}
      ${LAPACKE_INCLUDE}
      ${LIBXC_INCLUDE}
      ${CINT_INCLUDE})
  target_link_libraries(xmvb_cpp_test_support PUBLIC xmvb_cpp_vb xmvb_cpp_runtime)
endif()

add_library(
  xmvb_cpp_original_runtime_staging
  STATIC
  EXCLUDE_FROM_ALL
  ${XMVB_CPP_ORIGINAL_RUNTIME_STAGING_SOURCES})
target_compile_definitions(xmvb_cpp_original_runtime_staging PRIVATE PATH_MAX=4096)
target_include_directories(xmvb_cpp_original_runtime_staging
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/src/vbutil
    ${CMAKE_BINARY_DIR}/src/vbscf
    ${CMAKE_BINARY_DIR}/src/vbci
    ${CMAKE_BINARY_DIR}/src/vbpt2
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${LIBXC_INCLUDE}
    ${CINT_INCLUDE})
target_link_libraries(xmvb_cpp_original_runtime_staging PUBLIC xmvb_cpp_openmp)

add_library(xmvb_cpp_legacy_stubs STATIC
  vb/matrices/deepvb_stubs.c)
target_include_directories(xmvb_cpp_legacy_stubs
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${LIBXC_INCLUDE}
    ${CINT_INCLUDE})

function(xmvb_link_legacy_runtime target_name)
  target_link_options(${target_name} PRIVATE -Wl,--allow-multiple-definition)
  target_link_libraries(${target_name}
    PRIVATE
      "-Wl,--start-group"
      symm
      gopt
      vbdrv
      vbpt2
      dfvb
      vbscf
      vbci
      vbgus
      inpout
      vbint
      vbmath
      vbutil
      eda
      scf
      solver
      fock
      mol
      "-Wl,--end-group")
  target_link_libraries(${target_name}
    PRIVATE
      ${LIBXCF_LIB}
      ${LIBXC_LIB}
      ${CINT_LIB}
      xmvb_cpp_openmp
      xmvb_cpp_legacy_stubs
      xmvb_cpp_vb
      gfortran
      quadmath
      m)
  if (LAPACK_TYPE STREQUAL "OpenBLAS")
    target_link_libraries(${target_name} PRIVATE ${LAPACKE_LIB} ${LAPACK_LIB})
  elseif (LAPACK_TYPE STREQUAL "refBLAS")
    target_link_libraries(${target_name} PRIVATE lapacke cblas lapack)
  else()
    target_link_libraries(${target_name} PRIVATE ${LAPACKE_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endfunction()

function(xmvb_link_standalone_runtime target_name)
  target_link_libraries(${target_name}
    PRIVATE
      ${LIBXCF_LIB}
      ${LIBXC_LIB}
      ${CINT_LIB}
      xmvb_cpp_openmp
      xmvb_cpp_vb
      m)
  if (LAPACK_TYPE STREQUAL "OpenBLAS")
    target_link_libraries(${target_name} PRIVATE ${LAPACKE_LIB} ${LAPACK_LIB})
  elseif (LAPACK_TYPE STREQUAL "refBLAS")
    target_link_libraries(${target_name} PRIVATE lapacke cblas lapack)
  else()
    target_link_libraries(${target_name} PRIVATE ${LAPACKE_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endfunction()

add_executable(run_cpp_vbscf
  tools/run_cpp_vbscf.cpp)
set_target_properties(run_cpp_vbscf PROPERTIES OUTPUT_NAME xmvb-cpp.exe)
target_link_libraries(run_cpp_vbscf
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(run_cpp_vbscf)

add_executable(check_cpp_orbital_gradient
  tools/check_cpp_orbital_gradient.cpp)
target_link_libraries(check_cpp_orbital_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_orbital_gradient)

add_executable(check_cpp_auxiliary_gradient
  tools/check_cpp_auxiliary_gradient.cpp)
target_link_libraries(check_cpp_auxiliary_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_auxiliary_gradient)

add_executable(check_cpp_active_space_gradient
  tools/check_cpp_active_space_gradient.cpp)
target_link_libraries(check_cpp_active_space_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_active_space_gradient)

add_executable(check_same_spin_overlap_gradient
  tools/check_same_spin_overlap_gradient.cpp)
target_link_libraries(check_same_spin_overlap_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_same_spin_overlap_gradient)

add_executable(check_structure_overlap_weight
  tools/check_structure_overlap_weight.cpp)
target_link_libraries(check_structure_overlap_weight
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_structure_overlap_weight)

add_executable(check_active_overlap_split
  tools/check_active_overlap_split.cpp)
target_link_libraries(check_active_overlap_split
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_active_overlap_split)

add_executable(compare_deepvbh_jax_inference
  tools/compare_deepvbh_jax_inference.cpp)
target_link_libraries(compare_deepvbh_jax_inference
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_deepvbh_jax_inference)

add_custom_target(
  xmvb_cpp_runtime_assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/basis" "${CMAKE_BINARY_DIR}/basis"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/data" "${CMAKE_BINARY_DIR}/data"
  COMMENT "Copying standalone xmvb-cpp runtime assets into the build tree")
add_dependencies(run_cpp_vbscf xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_orbital_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_auxiliary_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_active_space_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_same_spin_overlap_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_structure_overlap_weight xmvb_cpp_runtime_assets)
add_dependencies(check_active_overlap_split xmvb_cpp_runtime_assets)
add_dependencies(compare_deepvbh_jax_inference xmvb_cpp_runtime_assets)

if (XMVB_CPP_BUILD_DEV_TARGETS)
  function(xmvb_add_cpp_test test_name source_path)
    add_executable(${test_name} ${source_path})
    target_link_libraries(${test_name} PRIVATE xmvb_cpp_test_support xmvb_cpp_runtime xmvb_cpp_vb)
    xmvb_link_legacy_runtime(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endfunction()

  add_executable(generalized_eigensolver_test
    tests/unit/generalized_eigensolver_test.cpp)
  target_link_libraries(generalized_eigensolver_test PRIVATE xmvb_cpp_vb)
  add_test(NAME generalized_eigensolver_test COMMAND generalized_eigensolver_test)
  set_tests_properties(generalized_eigensolver_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(print_cpp_orbital_gradient
    tools/print_cpp_orbital_gradient.cpp)
  target_link_libraries(print_cpp_orbital_gradient
    PRIVATE
      xmvb_cpp_vb
      xmvb_cpp_runtime)
  xmvb_link_legacy_runtime(print_cpp_orbital_gradient)

  add_executable(print_cpp_structure_matrices
    tools/print_cpp_structure_matrices.cpp)
  target_link_libraries(print_cpp_structure_matrices
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(print_cpp_structure_matrices)

  add_executable(compare_cpp_vs_legacy_vbscf
    tools/compare_cpp_vs_legacy_vbscf.cpp)
  target_link_libraries(compare_cpp_vs_legacy_vbscf
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  target_link_options(compare_cpp_vs_legacy_vbscf PRIVATE -Wl,--allow-multiple-definition)
  xmvb_link_legacy_runtime(compare_cpp_vs_legacy_vbscf)

  add_executable(compare_initial_orbital_gradient
    tools/compare_initial_orbital_gradient.cpp)
  target_link_libraries(compare_initial_orbital_gradient
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  target_link_options(compare_initial_orbital_gradient PRIVATE -Wl,--allow-multiple-definition)
  xmvb_link_legacy_runtime(compare_initial_orbital_gradient)

  add_executable(compare_first_alpha_pair_hamiltonian
    tools/compare_first_alpha_pair_hamiltonian.cpp)
  target_link_libraries(compare_first_alpha_pair_hamiltonian
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  target_link_options(compare_first_alpha_pair_hamiltonian PRIVATE -Wl,--allow-multiple-definition)
  xmvb_link_legacy_runtime(compare_first_alpha_pair_hamiltonian)

  add_executable(scan_alpha_pair_hamiltonian_differences
    tools/scan_alpha_pair_hamiltonian_differences.cpp)
  target_link_libraries(scan_alpha_pair_hamiltonian_differences
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  target_link_options(scan_alpha_pair_hamiltonian_differences PRIVATE -Wl,--allow-multiple-definition)
  xmvb_link_legacy_runtime(scan_alpha_pair_hamiltonian_differences)

  add_executable(legacy_orbital_gradient_comparison
    tests/unit/legacy_orbital_gradient_comparison.cpp)
  target_link_libraries(legacy_orbital_gradient_comparison
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  target_link_options(legacy_orbital_gradient_comparison PRIVATE -Wl,--allow-multiple-definition)
  xmvb_link_legacy_runtime(legacy_orbital_gradient_comparison)

  add_executable(legacy_orbital_gradient_fd_diagnostic
    tests/unit/legacy_orbital_gradient_fd_diagnostic.cpp)
  target_link_libraries(legacy_orbital_gradient_fd_diagnostic
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(legacy_orbital_gradient_fd_diagnostic)

  add_executable(legacy_gradient_layers_consistency
    tests/unit/legacy_gradient_layers_consistency.cpp)
  target_link_libraries(legacy_gradient_layers_consistency
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(legacy_gradient_layers_consistency)

  add_executable(legacy_style_orbital_gradient_projector_diagnostic
    tests/unit/legacy_style_orbital_gradient_projector_diagnostic.cpp)
  target_link_libraries(legacy_style_orbital_gradient_projector_diagnostic
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(legacy_style_orbital_gradient_projector_diagnostic)

  add_executable(legacy_orbprep_intermediate_comparison
    tests/unit/legacy_orbprep_intermediate_comparison.cpp)
  target_link_libraries(legacy_orbprep_intermediate_comparison
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(legacy_orbprep_intermediate_comparison)

  add_executable(legacy_orbprep_inactive_response_diagnostic
    tests/unit/legacy_orbprep_inactive_response_diagnostic.cpp)
  target_link_libraries(legacy_orbprep_inactive_response_diagnostic
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(legacy_orbprep_inactive_response_diagnostic)

  add_executable(structure_response_inactive_diagnostic
    tests/unit/structure_response_inactive_diagnostic.cpp)
  target_link_libraries(structure_response_inactive_diagnostic
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(structure_response_inactive_diagnostic)

  add_executable(real_case_inactive_orbital_backpropagator_diagnostic
    tests/unit/real_case_inactive_orbital_backpropagator_diagnostic.cpp)
  target_link_libraries(real_case_inactive_orbital_backpropagator_diagnostic
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(real_case_inactive_orbital_backpropagator_diagnostic)

  add_executable(downstream_energy_adjoint_diagnostic
    tests/unit/downstream_energy_adjoint_diagnostic.cpp)
  target_link_libraries(downstream_energy_adjoint_diagnostic
    PRIVATE
      xmvb_cpp_test_support
      xmvb_cpp_runtime
      xmvb_cpp_vb)
  xmvb_link_legacy_runtime(downstream_energy_adjoint_diagnostic)

  xmvb_add_cpp_test(cpp_vb_scf_evaluator_test
    tests/unit/cpp_vb_scf_evaluator_test.cpp)

  xmvb_add_cpp_test(cpp_active_space_gradient_evaluator_test
    tests/unit/cpp_active_space_gradient_evaluator_test.cpp)

  xmvb_add_cpp_test(cpp_orbital_gradient_evaluator_test
    tests/unit/cpp_orbital_gradient_evaluator_test.cpp)

  xmvb_add_cpp_test(active_space_orbital_backpropagator_test
    tests/unit/active_space_orbital_backpropagator_test.cpp)

  xmvb_add_cpp_test(cpp_vb_scf_optimizer_test
    tests/unit/cpp_vb_scf_optimizer_test.cpp)

  add_executable(structure_subspace_builder_test
    tests/unit/structure_subspace_builder_test.cpp)
  target_link_libraries(structure_subspace_builder_test PRIVATE xmvb_cpp_vb)
  add_test(NAME structure_subspace_builder_test COMMAND structure_subspace_builder_test)
  set_tests_properties(structure_subspace_builder_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(cpp_vb_scf_subspace_test
    tests/unit/cpp_vb_scf_subspace_test.cpp)
  target_link_libraries(cpp_vb_scf_subspace_test PRIVATE xmvb_cpp_vb)
  add_test(NAME cpp_vb_scf_subspace_test COMMAND cpp_vb_scf_subspace_test)
  set_tests_properties(cpp_vb_scf_subspace_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(structure_candidate_coupling_test
    tests/unit/structure_candidate_coupling_test.cpp)
  target_link_libraries(structure_candidate_coupling_test PRIVATE xmvb_cpp_vb)
  add_test(NAME structure_candidate_coupling_test COMMAND structure_candidate_coupling_test)
  set_tests_properties(structure_candidate_coupling_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
