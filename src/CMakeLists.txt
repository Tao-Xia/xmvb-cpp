file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" XMVB_CONDA_PREFIX)

option(
  XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
  "Enable the legacy Fortran L-BFGS backend inside the C++ VB optimizer."
  OFF)

option(
  XMVB_CPP_BUILD_DEV_TARGETS
  "Build C++ comparison, diagnostic, and test targets in addition to xmvb-cpp.exe."
  ON)

set(
  XMVB_CPP_HAVE_TEST_SOURCES
  FALSE)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
  set(XMVB_CPP_HAVE_TEST_SOURCES TRUE)
endif()

if (XMVB_CPP_BUILD_DEV_TARGETS AND
    NOT XMVB_CPP_HAVE_TEST_SOURCES)
  message(WARNING
    "XMVB_CPP_BUILD_DEV_TARGETS=ON but ${CMAKE_CURRENT_SOURCE_DIR}/tests is absent in "
    "this repository export. Building standalone diagnostic tools only; skipping "
    "targets that depend on omitted test sources.")
endif()

option(
  XMVB_CPP_BUILD_LEGACY_HF_RUNTIME
  "Build the legacy standalone HF/XINT runtime used by legacy orbital guesses and HF diagnostics."
  OFF)

option(
  XMVB_CPP_ENABLE_EIGEN_BLAS_BACKEND
  "Route Eigen dense linear algebra through the configured BLAS/LAPACKE backends when supported."
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
  core/linear_algebra/generalized_eigensolver.cpp
  vb/matrices/determinant_hamiltonian_resolver.cpp
  vb/matrices/determinant_overlap_resolver.cpp
  vb/matrices/full_determinant_pair_evaluator.cpp
  vb/matrices/full_structure_expander.cpp
  vb/matrices/full_structure_builder.cpp
  vb/matrices/legacy_structure_overlap.cpp
  vb/matrices/cpp_vb_input_ri_cache.cpp
  vb/matrices/prepared_active_space_context.cpp
  vb/matrices/same_spin_pair_cache.cpp
  vb/matrices/union_graph_rank_predictor.cpp
  vb/matrices/raw_structure_subspace_selector.cpp
  vb/matrices/spin_pair_utils.cpp
  vb/matrices/structure_matrix_evaluator.cpp
  vb/matrices/structure_subspace_builder.cpp
  vb/matrices/two_electron_indexer.cpp
  vb/matrices/union_graph_screening.cpp
  vb/exact_separator/bundle.cpp
  vb/exact_separator/bundle_channels.cpp
  vb/exact_separator/bundle_merge.cpp
  vb/exact_separator/bundle_root_helpers.cpp
  vb/exact_separator/boundary_sector.cpp
  vb/exact_separator/component_terms.cpp
  vb/exact_separator/component_tree.cpp
  vb/exact_separator/hamiltonian_bundle.cpp
  vb/exact_separator/leaf_boundary_bundle.cpp
  vb/exact_separator/leaf_boundary_bundle_table.cpp
  vb/exact_separator/one_leaf_packed_bundle.cpp
  vb/exact_separator/one_electron.cpp
  vb/exact_separator/leaf_coefficient_operator.cpp
  vb/exact_separator/spin_state_aggregate.cpp
  vb/exact_separator/two_electron.cpp
  vb/exact_separator/two_electron_opposite_spin.cpp
  vb/exact_separator/two_electron_star.cpp
  vb/model/deepvbh_jax_inference_runner.cpp
  vb/biorthogonal_vbscf/biorthogonal_orbital_frame.cpp
  vb/biorthogonal_vbscf/biorthogonal_orbital_integrals.cpp
  vb/biorthogonal_vbscf/biorthogonal_determinant_hamiltonian.cpp
  vb/biorthogonal_vbscf/biorthogonal_structure_expansion.cpp
  vb/biorthogonal_vbscf/biorthogonal_prepared_input.cpp
  vb/biorthogonal_vbscf/biorthogonal_spin_pair_tiles.cpp
  vb/biorthogonal_vbscf/biorthogonal_structure_hamiltonian_builder.cpp
  vb/biorthogonal_vbscf/biorthogonal_selected_structure_space.cpp
  vb/biorthogonal_vbscf/biorthogonal_projected_structure_problem.cpp
  vb/biorthogonal_vbscf/biorthogonal_projected_solver.cpp
  vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure.cpp
  vb/biorthogonal_vbscf/biorthogonal_selected_state_matrices.cpp
  vb/biorthogonal_vbscf/biorthogonal_one_electron_transform_backward.cpp
  vb/biorthogonal_vbscf/biorthogonal_two_electron_transform_backward.cpp
  vb/biorthogonal_vbscf/biorthogonal_opposite_spin_matrix_backward.cpp
  vb/biorthogonal_vbscf/biorthogonal_same_spin_matrix_backward.cpp
  vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_gradient.cpp
  vb/biorthogonal_vbscf/biorthogonal_exact_selected_structure_scf.cpp
  vb/biorthogonal_vbscf/biorthogonal_forward_evaluator.cpp
  vb/pdft/bragg_slater_radii.cpp
  vb/pdft/becke_partition.cpp
  vb/pdft/lebedev_grid.cpp
  vb/pdft/libcint_ao_grid_evaluator.cpp
  vb/pdft/libxc_functional.cpp
  vb/pdft/molecular_grid.cpp
  vb/pdft/radial_grid.cpp
  vb/pdft/real_space_density_builder.cpp
  vb/pdft/selected_state_exact_physical_one_rdm_builder.cpp
  vb/pdft/selected_state_exact_physical_on_top_pair_density_builder.cpp
  vb/pdft/selected_state_matrix_form_one_rdm_builder.cpp
  vb/pdft/selected_state_matrix_form_two_rdm_builder.cpp
  vb/pdft/translated_spin_density.cpp
  vb/pdft/vb_pdft_energy_evaluator.cpp
  vb/orbital/active_space_matrix_backpropagator.cpp
  vb/orbital/active_space_one_electron_builder.cpp
  vb/orbital/active_space_orbital_backpropagator.cpp
  vb/orbital/active_space_orbital_preparer.cpp
  vb/orbital/ao_two_electron_pair_index_utils.cpp
  vb/orbital/active_space_two_electron_backpropagator.cpp
  vb/orbital/active_space_two_electron_builder.cpp
  vb/orbital/active_space_two_electron_utils.cpp
  vb/orbital/ao_effective_one_electron_backpropagator.cpp
  vb/orbital/ao_effective_one_electron_builder.cpp
  vb/orbital/ao_effective_one_electron_graph_operator.cpp
  vb/orbital/ao_effective_one_electron_ri_operator.cpp
  vb/orbital/legacy_style_orbital_gradient_projector.cpp
  vb/orbital/legacy_jacobi_diagonalizer.cpp
  vb/orbital/localized_representative_selector.cpp
  vb/orbital/nonredundant_optimizer_input_adapter.cpp
  vb/orbital/nonredundant_orbital_space.cpp
  vb/orbital/orbital_parameter_codec.cpp
  vb/orbital/sparse_orbital_parameter_view.cpp
  vb/orbital/support_aware_mo_gauge_fix.cpp
  vb/orbital/ri_active_space_two_electron_builder.cpp
  vb/scf/selected_state_determinant_matrices.cpp
  vb/scf/opposite_spin_matrix_channels.cpp
  vb/scf/opposite_spin_matrix_backward.cpp
  vb/scf/same_spin_matrix_backward.cpp
  vb/scf/exact_ctx_strategy_profile.cpp
  vb/scf/exact_orbital_second_order_operator.cpp
  vb/scf/cpp_active_space_gradient_result_utils.cpp
  vb/scf/cpp_active_space_gradient_evaluator.cpp
  vb/scf/adaptive_structure_space_optimizer.cpp
  vb/scf/deepvbh_onnx_direct_final_optimizer.cpp
  vb/scf/deepvbh_onnx_hybrid_optimizer.cpp
  vb/scf/cpp_orbital_gradient_evaluator.cpp
  vb/scf/cpp_vb_scf_evaluator.cpp
  vb/scf/cpp_vb_scf_optimizer.cpp
  )

set(XMVB_CPP_RUNTIME_CORE_SOURCES
  runtime/cpp_block_guess_builder.cpp
  runtime/cpp_closed_shell_fock_builder.cpp
  runtime/cpp_initial_guess_builder.cpp
  runtime/molden_file_writer.cpp
  runtime/cpp_restricted_hartree_fock.cpp
  runtime/cpp_vb_input_loader.cpp
  runtime/libcint_auxiliary_basis_builder.cpp
  runtime/libcint_direct_shell_evaluator.cpp
  runtime/libcint_materialized_integral_provider.cpp
  runtime/libcint_ri_integral_provider.cpp
  runtime/materialized_ao_integral_input_builder.cpp
  runtime_c/local_runtime_api.c
  runtime_c/local_runtime_grid_api.c
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
  runtime_c/local_runtime/hgp_3c2e.c
  runtime_c/local_runtime/init_bovb.c
  runtime_c/local_runtime/init_vb_param.c
  runtime_c/cpp_runtime_extractor.c
  runtime_c/local_runtime/int_data_trans.c
  runtime_c/local_runtime/lab.c
  runtime_c/local_runtime/mol.c
  runtime_c/local_runtime/boys.c
  runtime_c/local_runtime/nb3idx.c
  runtime_c/local_runtime/normalize.c
  runtime_c/local_runtime/pair.c
  runtime_c/local_runtime/print_crd.c
  runtime_c/local_runtime/print_vb_comp_info.c
  runtime_c/local_runtime/readinp.c
  runtime_c/local_runtime/readint.c
  runtime_c/local_runtime/readint_head.c
  runtime_c/local_runtime/readorb.c
  runtime_c/local_runtime/readinp_vb.c
  runtime_c/local_runtime/readstr.c
  runtime_c/local_runtime/split_string.c
  runtime_c/local_runtime/value_ao.c
  runtime_c/local_runtime/vbprep.c
  runtime_c/local_runtime/build_g1d.c
  runtime_c/local_runtime/getvars.c
)

set(XMVB_CPP_RUNTIME_LEGACY_HF_SOURCES
  runtime_c/local_runtime_hf_api.c
  runtime_c/local_runtime/hf.c
  runtime_c/local_runtime/coulomb.c
  runtime_c/local_runtime/exchange.c
  runtime_c/local_runtime/global.c
  runtime_c/local_runtime/print_init_guess.c
  runtime_c/local_runtime/sad.c
  runtime_c/local_runtime/vb_autoguess.c
  runtime_c/local_runtime/vbguess.c
  runtime_c/local_runtime/vb_readguess.c
  runtime_c/local_runtime/vb_unitguess.c
  runtime_c/local_runtime/value_ao.c
  runtime_c/local_runtime/xint.c
  runtime_c/local_runtime/xint_2c1e.c
  runtime_c/local_runtime/xint_2c2e.c
  runtime_c/local_runtime/xint_3c1e.c
  runtime_c/local_runtime/xint_3c2e.c
  runtime_c/local_runtime/xint_hrr.c
  runtime_c/local_runtime/xint_overlap.c
  runtime_c/local_runtime/xgrids.c)

set(XMVB_CPP_RUNTIME_SOURCES ${XMVB_CPP_RUNTIME_CORE_SOURCES})
if (XMVB_CPP_BUILD_LEGACY_HF_RUNTIME)
  list(APPEND XMVB_CPP_RUNTIME_SOURCES ${XMVB_CPP_RUNTIME_LEGACY_HF_SOURCES})
endif()

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
    ${CMAKE_SOURCE_DIR}/include
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
if (XMVB_CPP_ENABLE_EIGEN_BLAS_BACKEND)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC
      EIGEN_USE_BLAS=1
      EIGEN_USE_LAPACKE=1)
endif()
# Enable libxc support for VB-PDFT
if (LIBXC_INCLUDE)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC XMVB_CPP_HAS_LIBXC=1)
  # Link libxc library (already linked via LIBXC_LIB in main CMakeLists.txt)
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
if (XMVB_CPP_BUILD_LEGACY_HF_RUNTIME)
  target_compile_definitions(xmvb_cpp_runtime PRIVATE XMVB_CPP_HAVE_LEGACY_HF_RUNTIME=1)
endif()
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

if (XMVB_CPP_BUILD_DEV_TARGETS AND XMVB_CPP_HAVE_TEST_SOURCES)
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
      sci
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

function(xmvb_mark_targets_exclude_from_all)
  foreach(target_name IN LISTS ARGN)
    if (TARGET ${target_name})
      set_target_properties(
        ${target_name}
        PROPERTIES
          EXCLUDE_FROM_ALL TRUE
          EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    endif()
  endforeach()
endfunction()

add_executable(run_cpp_vbscf
  tools/run_cpp_vbscf.cpp)
set_target_properties(run_cpp_vbscf PROPERTIES OUTPUT_NAME xmvb-cpp.exe)
target_link_libraries(run_cpp_vbscf
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(run_cpp_vbscf)

add_executable(check_vb_pdft_energy_evaluator
  tools/check_vb_pdft_energy_evaluator.cpp
  runtime_c/local_runtime/xgrids.c)
target_link_libraries(check_vb_pdft_energy_evaluator
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_vb_pdft_energy_evaluator)

file(
  GLOB_RECURSE XMVB_CPP_RUNTIME_BASIS_INPUTS
  "${CMAKE_SOURCE_DIR}/basis/*")
set(XMVB_CPP_RUNTIME_ASSET_STAMP
  "${CMAKE_BINARY_DIR}/CMakeFiles/xmvb_cpp_runtime_assets.stamp")
add_custom_command(
  OUTPUT ${XMVB_CPP_RUNTIME_ASSET_STAMP}
  COMMAND ${CMAKE_COMMAND}
          -DXMVB_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DXMVB_BINARY_DIR=${CMAKE_BINARY_DIR}
          -P ${CMAKE_SOURCE_DIR}/cmake/stage_runtime_assets.cmake
  COMMAND ${CMAKE_COMMAND} -E touch ${XMVB_CPP_RUNTIME_ASSET_STAMP}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/cmake/stage_runtime_assets.cmake
    ${XMVB_CPP_RUNTIME_BASIS_INPUTS}
  # `basis/` is copied into the build tree, so basis-file edits should restage
  # runtime assets. `data/` is linked instead of copied and contains volatile
  # training corpora, logs, and scratch outputs. Do not expand that directory
  # into the Ninja dependency graph, or every dataset churn will force a CMake
  # regenerate and invalidate the whole Ninja deps database.
  COMMENT "Staging standalone xmvb-cpp runtime assets into the build tree")
add_custom_target(
  xmvb_cpp_runtime_assets
  DEPENDS ${XMVB_CPP_RUNTIME_ASSET_STAMP})
add_dependencies(run_cpp_vbscf xmvb_cpp_runtime_assets)
add_dependencies(check_vb_pdft_energy_evaluator xmvb_cpp_runtime_assets)

if (XMVB_CPP_BUILD_DEV_TARGETS)
get_property(_xmvb_targets_before_dev_tools DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
add_executable(check_cpp_orbital_gradient
  tools/check_cpp_orbital_gradient.cpp)
target_link_libraries(check_cpp_orbital_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_orbital_gradient)

add_executable(benchmark_cpp_orbital_eval
  tools/benchmark_cpp_orbital_eval.cpp)
target_link_libraries(benchmark_cpp_orbital_eval
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_cpp_orbital_eval)

add_executable(check_cpp_auxiliary_gradient
  tools/check_cpp_auxiliary_gradient.cpp)
target_link_libraries(check_cpp_auxiliary_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_auxiliary_gradient)

add_executable(check_cpp_reference_orbital_gradient
  tools/check_cpp_reference_orbital_gradient.cpp)
target_link_libraries(check_cpp_reference_orbital_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_reference_orbital_gradient)

add_executable(check_ao_h1e_ri_operator
  tools/check_ao_h1e_ri_operator.cpp)
target_link_libraries(check_ao_h1e_ri_operator
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_ao_h1e_ri_operator)

add_executable(inspect_ri_inactive_density
  tools/inspect_ri_inactive_density.cpp)
target_link_libraries(inspect_ri_inactive_density
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(inspect_ri_inactive_density)

add_executable(check_ri_ao_h1e_modes
  tools/check_ri_ao_h1e_modes.cpp)
target_link_libraries(check_ri_ao_h1e_modes
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_ri_ao_h1e_modes)

add_executable(benchmark_ri_ao_h1e_operator
  tools/benchmark_ri_ao_h1e_operator.cpp)
target_link_libraries(benchmark_ri_ao_h1e_operator
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_ri_ao_h1e_operator)

add_executable(inspect_ao_h1e_backprop_symmetry
  tools/inspect_ao_h1e_backprop_symmetry.cpp)
target_link_libraries(inspect_ao_h1e_backprop_symmetry
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(inspect_ao_h1e_backprop_symmetry)

add_executable(check_cpp_reference_orbital_gradient_modes
  tools/check_cpp_reference_orbital_gradient_modes.cpp)
target_link_libraries(check_cpp_reference_orbital_gradient_modes
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_reference_orbital_gradient_modes)

add_executable(compare_exact_ri_energy_decomposition
  tools/compare_exact_ri_energy_decomposition.cpp)
target_link_libraries(compare_exact_ri_energy_decomposition
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_exact_ri_energy_decomposition)

add_executable(check_cpp_active_space_gradient
  tools/check_cpp_active_space_gradient.cpp)
target_link_libraries(check_cpp_active_space_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_active_space_gradient)

add_executable(check_selected_state_matrix_form_one_rdm
  tools/check_selected_state_matrix_form_one_rdm.cpp)
target_link_libraries(check_selected_state_matrix_form_one_rdm
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_selected_state_matrix_form_one_rdm)

add_executable(check_selected_state_exact_physical_one_rdm
  tools/check_selected_state_exact_physical_one_rdm.cpp)
target_link_libraries(check_selected_state_exact_physical_one_rdm
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_selected_state_exact_physical_one_rdm)

add_executable(check_selected_state_exact_physical_on_top_pair_density
  tools/check_selected_state_exact_physical_on_top_pair_density.cpp)
target_link_libraries(check_selected_state_exact_physical_on_top_pair_density
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_selected_state_exact_physical_on_top_pair_density)

add_executable(check_selected_state_matrix_form_two_rdm
  tools/check_selected_state_matrix_form_two_rdm.cpp)
target_link_libraries(check_selected_state_matrix_form_two_rdm
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_selected_state_matrix_form_two_rdm)

add_executable(check_structure_builder_fast_path
  tools/check_structure_builder_fast_path.cpp)
target_link_libraries(check_structure_builder_fast_path
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_structure_builder_fast_path)

add_executable(check_biorthogonal_determinant_hamiltonian
  tools/check_biorthogonal_determinant_hamiltonian.cpp)
target_link_libraries(check_biorthogonal_determinant_hamiltonian
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_biorthogonal_projected_structure_solver
  tools/check_biorthogonal_projected_structure_solver.cpp)
target_link_libraries(check_biorthogonal_projected_structure_solver
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_biorthogonal_forward_evaluator
  tools/check_biorthogonal_forward_evaluator.cpp)
target_link_libraries(check_biorthogonal_forward_evaluator
  PRIVATE
    xmvb_cpp_vb)

add_executable(compare_biorthogonal_nonorth_forward
  tools/compare_biorthogonal_nonorth_forward.cpp)
target_link_libraries(compare_biorthogonal_nonorth_forward
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_biorthogonal_nonorth_forward)

add_executable(check_biorthogonal_structure_subspaces
  tools/check_biorthogonal_structure_subspaces.cpp)
target_link_libraries(check_biorthogonal_structure_subspaces
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_structure_subspaces)

add_executable(check_biorthogonal_dual_structure_projection
  tools/check_biorthogonal_dual_structure_projection.cpp)
target_link_libraries(check_biorthogonal_dual_structure_projection
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_dual_structure_projection)

add_executable(check_biorthogonal_exact_selected_subspace
  tools/check_biorthogonal_exact_selected_subspace.cpp)
target_link_libraries(check_biorthogonal_exact_selected_subspace
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_exact_selected_subspace)

add_executable(check_biorthogonal_exact_selected_active_space_gradient
  tools/check_biorthogonal_exact_selected_active_space_gradient.cpp)
target_link_libraries(check_biorthogonal_exact_selected_active_space_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_exact_selected_active_space_gradient)

add_executable(check_biorthogonal_exact_selected_orbital_gradient
  tools/check_biorthogonal_exact_selected_orbital_gradient.cpp)
target_link_libraries(check_biorthogonal_exact_selected_orbital_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_exact_selected_orbital_gradient)

add_executable(check_biorthogonal_selected_state_matrices
  tools/check_biorthogonal_selected_state_matrices.cpp)
target_link_libraries(check_biorthogonal_selected_state_matrices
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_selected_state_matrices)

add_executable(check_biorthogonal_same_spin_matrix_backward
  tools/check_biorthogonal_same_spin_matrix_backward.cpp)
target_link_libraries(check_biorthogonal_same_spin_matrix_backward
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_same_spin_matrix_backward)

add_executable(check_biorthogonal_opposite_spin_matrix_backward
  tools/check_biorthogonal_opposite_spin_matrix_backward.cpp)
target_link_libraries(check_biorthogonal_opposite_spin_matrix_backward
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_opposite_spin_matrix_backward)

add_executable(check_biorthogonal_exact_selected_matrix_backward
  tools/check_biorthogonal_exact_selected_matrix_backward.cpp)
target_link_libraries(check_biorthogonal_exact_selected_matrix_backward
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_exact_selected_matrix_backward)

add_executable(check_biorthogonal_one_electron_transform_backward
  tools/check_biorthogonal_one_electron_transform_backward.cpp)
target_link_libraries(check_biorthogonal_one_electron_transform_backward
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_one_electron_transform_backward)

add_executable(check_biorthogonal_two_electron_transform_backward
  tools/check_biorthogonal_two_electron_transform_backward.cpp)
target_link_libraries(check_biorthogonal_two_electron_transform_backward
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_biorthogonal_two_electron_transform_backward)

add_executable(benchmark_biorthogonal_exact_selected_subspace
  tools/benchmark_biorthogonal_exact_selected_subspace.cpp)
target_link_libraries(benchmark_biorthogonal_exact_selected_subspace
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_biorthogonal_exact_selected_subspace)

add_executable(check_exact_active_space_builders
  tools/check_exact_active_space_builders.cpp)
target_link_libraries(check_exact_active_space_builders
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_exact_active_space_builders)

add_executable(check_exact_two_electron_hvp
  tools/check_exact_two_electron_hvp.cpp)
target_link_libraries(check_exact_two_electron_hvp
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_exact_two_electron_hvp)

add_executable(check_exact_ctx_hvp
  tools/check_exact_ctx_hvp.cpp)
target_link_libraries(check_exact_ctx_hvp
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_exact_ctx_hvp)

add_executable(benchmark_exact_ctx_hvp
  tools/benchmark_exact_ctx_hvp.cpp)
target_link_libraries(benchmark_exact_ctx_hvp
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_exact_ctx_hvp)

add_executable(check_exact_ao_h1e_builder
  tools/check_exact_ao_h1e_builder.cpp)
target_link_libraries(check_exact_ao_h1e_builder
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_exact_ao_h1e_builder)

add_executable(check_structure_expansion_signs
  tools/check_structure_expansion_signs.cpp)
target_link_libraries(check_structure_expansion_signs
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_structure_expansion_signs)

add_executable(check_structure_overlap_against_raw_legacy
  tools/check_structure_overlap_against_raw_legacy.cpp)
target_link_libraries(check_structure_overlap_against_raw_legacy
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_structure_overlap_against_raw_legacy)

add_executable(inspect_cpp_vb_load
  tools/inspect_cpp_vb_load.cpp)
target_link_libraries(inspect_cpp_vb_load
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(inspect_cpp_vb_load)

add_executable(compare_molden_active_auxiliary
  tools/compare_molden_active_auxiliary.cpp)
target_link_libraries(compare_molden_active_auxiliary
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_molden_active_auxiliary)

add_executable(dump_loaded_molden
  tools/dump_loaded_molden.cpp)
target_link_libraries(dump_loaded_molden
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(dump_loaded_molden)

add_executable(dump_legacy_readguess_molden
  tools/dump_legacy_readguess_molden.cpp
  runtime_c/local_runtime/vb_readguess.c)
target_link_libraries(dump_legacy_readguess_molden
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(dump_legacy_readguess_molden)

add_executable(inspect_same_spin_reuse
  tools/inspect_same_spin_reuse.cpp)
target_link_libraries(inspect_same_spin_reuse
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(inspect_same_spin_reuse)

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

add_executable(check_geminal_structure_overlap
  tools/check_geminal_structure_overlap.cpp)
target_link_libraries(check_geminal_structure_overlap
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_geminal_structure_overlap)

add_executable(check_union_graph_overlap_blocks
  tools/check_union_graph_overlap_blocks.cpp)
target_link_libraries(check_union_graph_overlap_blocks
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_union_graph_overlap_blocks)

add_executable(analyze_union_graph_rank_dataset
  tools/analyze_union_graph_rank_dataset.cpp)
target_link_libraries(analyze_union_graph_rank_dataset
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_union_graph_rank_dataset)

add_executable(analyze_metric_aware_graph
  tools/analyze_metric_aware_graph.cpp)
target_link_libraries(analyze_metric_aware_graph
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_metric_aware_graph)

add_executable(analyze_metric_aware_graph_dataset
  tools/analyze_metric_aware_graph_dataset.cpp)
target_link_libraries(analyze_metric_aware_graph_dataset
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_metric_aware_graph_dataset)

add_executable(scan_star_separator_pairs
  tools/scan_star_separator_pairs.cpp)
target_link_libraries(scan_star_separator_pairs
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(scan_star_separator_pairs)

add_executable(analyze_exact_disconnected_overlap_dataset
  tools/analyze_exact_disconnected_overlap_dataset.cpp)
target_link_libraries(analyze_exact_disconnected_overlap_dataset
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_exact_disconnected_overlap_dataset)

add_executable(debug_exact_raw_vb_overlap_pair
  tools/debug_exact_raw_vb_overlap_pair.cpp)
target_link_libraries(debug_exact_raw_vb_overlap_pair
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(debug_exact_raw_vb_overlap_pair)

add_executable(analyze_star_separator_overlap_dataset
  tools/analyze_star_separator_overlap_dataset.cpp)
target_link_libraries(analyze_star_separator_overlap_dataset
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_star_separator_overlap_dataset)

add_executable(analyze_star_separator_one_electron_dataset
  tools/analyze_star_separator_one_electron_dataset.cpp)
target_link_libraries(analyze_star_separator_one_electron_dataset
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_star_separator_one_electron_dataset)

add_executable(check_exact_separator_two_electron
  tools/check_exact_separator_two_electron.cpp)
target_link_libraries(check_exact_separator_two_electron
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_exact_separator_two_electron_opposite_spin
  tools/check_exact_separator_two_electron_opposite_spin.cpp)
target_link_libraries(check_exact_separator_two_electron_opposite_spin
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_exact_separator_two_electron_one_leaf
  tools/check_exact_separator_two_electron_one_leaf.cpp)
target_link_libraries(check_exact_separator_two_electron_one_leaf
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_exact_separator_one_leaf_boundary_message
  tools/check_exact_separator_one_leaf_boundary_message.cpp)
target_link_libraries(check_exact_separator_one_leaf_boundary_message
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_exact_separator_two_electron_multi_leaf
  tools/check_exact_separator_two_electron_multi_leaf.cpp)
target_link_libraries(check_exact_separator_two_electron_multi_leaf
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_exact_separator_two_electron_component_tree
  tools/check_exact_separator_two_electron_component_tree.cpp)
target_link_libraries(check_exact_separator_two_electron_component_tree
  PRIVATE
    xmvb_cpp_vb)

add_executable(check_exact_separator_component_tree_gradient
  tools/check_exact_separator_component_tree_gradient.cpp)
target_link_libraries(check_exact_separator_component_tree_gradient
  PRIVATE
    xmvb_cpp_vb)

add_executable(validate_rooted_component_tree_exact_hamiltonian
  tools/validate_rooted_component_tree_exact_hamiltonian.cpp)
target_link_libraries(validate_rooted_component_tree_exact_hamiltonian
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(validate_rooted_component_tree_exact_hamiltonian)

add_executable(debug_rooted_component_tree_star_pair
  tools/debug_rooted_component_tree_star_pair.cpp)
target_link_libraries(debug_rooted_component_tree_star_pair
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(debug_rooted_component_tree_star_pair)

add_executable(validate_one_leaf_open_state_one_electron
  tools/validate_one_leaf_open_state_one_electron.cpp)
target_link_libraries(validate_one_leaf_open_state_one_electron
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(validate_one_leaf_open_state_one_electron)

add_executable(validate_one_leaf_exact_separator_two_electron
  tools/validate_one_leaf_exact_separator_two_electron.cpp)
target_link_libraries(validate_one_leaf_exact_separator_two_electron
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(validate_one_leaf_exact_separator_two_electron)

add_executable(debug_one_leaf_boundary_state_pairs
  tools/debug_one_leaf_boundary_state_pairs.cpp)
target_link_libraries(debug_one_leaf_boundary_state_pairs
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(debug_one_leaf_boundary_state_pairs)

add_executable(validate_multi_leaf_exact_separator_two_electron
  tools/validate_multi_leaf_exact_separator_two_electron.cpp)
target_link_libraries(validate_multi_leaf_exact_separator_two_electron
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(validate_multi_leaf_exact_separator_two_electron)

add_executable(validate_multi_leaf_open_state_one_electron
  tools/validate_multi_leaf_open_state_one_electron.cpp)
target_link_libraries(validate_multi_leaf_open_state_one_electron
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(validate_multi_leaf_open_state_one_electron)

add_executable(benchmark_union_graph_single_step
  tools/benchmark_union_graph_single_step.cpp)
target_link_libraries(benchmark_union_graph_single_step
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_union_graph_single_step)

add_executable(benchmark_rooted_component_tree_pairs
  tools/benchmark_rooted_component_tree_pairs.cpp)
target_link_libraries(benchmark_rooted_component_tree_pairs
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_rooted_component_tree_pairs)

add_executable(check_covalent_pair_cluster_hamiltonian
  tools/check_covalent_pair_cluster_hamiltonian.cpp)
target_link_libraries(check_covalent_pair_cluster_hamiltonian
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_covalent_pair_cluster_hamiltonian)

add_executable(check_direct_libcint_smoke
  tools/check_direct_libcint_smoke.cpp)
target_link_libraries(check_direct_libcint_smoke
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_direct_libcint_smoke)

add_executable(check_libcint_ri_smoke
  tools/check_libcint_ri_smoke.cpp)
target_link_libraries(check_libcint_ri_smoke
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_libcint_ri_smoke)

add_executable(compare_libcint_materialized_provider
  tools/compare_libcint_materialized_provider.cpp)
target_link_libraries(compare_libcint_materialized_provider
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_libcint_materialized_provider)

add_executable(compare_ri_active_space_builder
  tools/compare_ri_active_space_builder.cpp)
target_link_libraries(compare_ri_active_space_builder
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_ri_active_space_builder)

if (XMVB_CPP_BUILD_LEGACY_HF_RUNTIME)
  add_executable(compare_cpp_rhf_legacy_hf
    tools/compare_cpp_rhf_legacy_hf.cpp)
  target_link_libraries(compare_cpp_rhf_legacy_hf
    PRIVATE
      xmvb_cpp_vb
      xmvb_cpp_runtime)
  xmvb_link_standalone_runtime(compare_cpp_rhf_legacy_hf)

  add_executable(compare_cpp_legacy_orbital_gradient_layers
    tools/compare_cpp_legacy_orbital_gradient_layers.cpp
    tools/legacy_gradient_layers_helper.F90)
  target_link_libraries(compare_cpp_legacy_orbital_gradient_layers
    PRIVATE
      xmvb_cpp_vb
      xmvb_cpp_runtime)
  xmvb_link_legacy_runtime(compare_cpp_legacy_orbital_gradient_layers)

  add_executable(compare_cpp_legacy_vb_energy
    tools/compare_cpp_legacy_vb_energy.cpp
    vb/matrices/legacy_hamiltonian_overlap_calculator.cpp
    vb/matrices/legacy_hamiltonian_overlap_calculator_c.c)
  target_link_libraries(compare_cpp_legacy_vb_energy
    PRIVATE
      xmvb_cpp_vb
      xmvb_cpp_runtime)
  xmvb_link_legacy_runtime(compare_cpp_legacy_vb_energy)

  add_executable(check_full_structure_expander_legacy
    tools/check_full_structure_expander_legacy.cpp)
  target_link_libraries(check_full_structure_expander_legacy
    PRIVATE
      xmvb_cpp_vb
      xmvb_cpp_runtime)
  xmvb_link_legacy_runtime(check_full_structure_expander_legacy)

  add_executable(compare_legacy_block_guess
    tools/compare_legacy_block_guess.cpp)
  target_link_libraries(compare_legacy_block_guess
    PRIVATE
      xmvb_cpp_vb
      xmvb_cpp_runtime)
  xmvb_link_standalone_runtime(compare_legacy_block_guess)
endif()

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
add_dependencies(check_cpp_orbital_gradient xmvb_cpp_runtime_assets)
add_dependencies(benchmark_cpp_orbital_eval xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_auxiliary_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_reference_orbital_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_ao_h1e_ri_operator xmvb_cpp_runtime_assets)
add_dependencies(inspect_ri_inactive_density xmvb_cpp_runtime_assets)
add_dependencies(check_ri_ao_h1e_modes xmvb_cpp_runtime_assets)
add_dependencies(benchmark_ri_ao_h1e_operator xmvb_cpp_runtime_assets)
add_dependencies(inspect_ao_h1e_backprop_symmetry xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_reference_orbital_gradient_modes xmvb_cpp_runtime_assets)
add_dependencies(compare_exact_ri_energy_decomposition xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_active_space_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_selected_state_matrix_form_one_rdm xmvb_cpp_runtime_assets)
add_dependencies(check_selected_state_exact_physical_one_rdm xmvb_cpp_runtime_assets)
add_dependencies(check_selected_state_exact_physical_on_top_pair_density xmvb_cpp_runtime_assets)
add_dependencies(check_selected_state_matrix_form_two_rdm xmvb_cpp_runtime_assets)
add_dependencies(check_exact_active_space_builders xmvb_cpp_runtime_assets)
add_dependencies(check_exact_two_electron_hvp xmvb_cpp_runtime_assets)
add_dependencies(check_exact_ctx_hvp xmvb_cpp_runtime_assets)
add_dependencies(benchmark_exact_ctx_hvp xmvb_cpp_runtime_assets)
add_dependencies(inspect_cpp_vb_load xmvb_cpp_runtime_assets)
add_dependencies(check_structure_expansion_signs xmvb_cpp_runtime_assets)
add_dependencies(inspect_same_spin_reuse xmvb_cpp_runtime_assets)
add_dependencies(check_same_spin_overlap_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_structure_overlap_weight xmvb_cpp_runtime_assets)
add_dependencies(check_geminal_structure_overlap xmvb_cpp_runtime_assets)
add_dependencies(check_union_graph_overlap_blocks xmvb_cpp_runtime_assets)
add_dependencies(analyze_union_graph_rank_dataset xmvb_cpp_runtime_assets)
add_dependencies(analyze_metric_aware_graph xmvb_cpp_runtime_assets)
add_dependencies(analyze_metric_aware_graph_dataset xmvb_cpp_runtime_assets)
add_dependencies(scan_star_separator_pairs xmvb_cpp_runtime_assets)
add_dependencies(analyze_exact_disconnected_overlap_dataset xmvb_cpp_runtime_assets)
add_dependencies(debug_exact_raw_vb_overlap_pair xmvb_cpp_runtime_assets)
add_dependencies(analyze_star_separator_overlap_dataset xmvb_cpp_runtime_assets)
add_dependencies(analyze_star_separator_one_electron_dataset xmvb_cpp_runtime_assets)
add_dependencies(validate_multi_leaf_open_state_one_electron xmvb_cpp_runtime_assets)
add_dependencies(benchmark_union_graph_single_step xmvb_cpp_runtime_assets)
add_dependencies(check_covalent_pair_cluster_hamiltonian xmvb_cpp_runtime_assets)
add_dependencies(check_direct_libcint_smoke xmvb_cpp_runtime_assets)
add_dependencies(check_libcint_ri_smoke xmvb_cpp_runtime_assets)
add_dependencies(compare_libcint_materialized_provider xmvb_cpp_runtime_assets)
add_dependencies(compare_ri_active_space_builder xmvb_cpp_runtime_assets)
if (XMVB_CPP_BUILD_LEGACY_HF_RUNTIME)
  add_dependencies(compare_cpp_rhf_legacy_hf xmvb_cpp_runtime_assets)
  add_dependencies(compare_cpp_legacy_orbital_gradient_layers xmvb_cpp_runtime_assets)
  add_dependencies(compare_cpp_legacy_vb_energy xmvb_cpp_runtime_assets)
  add_dependencies(compare_legacy_block_guess xmvb_cpp_runtime_assets)
endif()
add_dependencies(check_active_overlap_split xmvb_cpp_runtime_assets)
add_dependencies(compare_deepvbh_jax_inference xmvb_cpp_runtime_assets)
get_property(_xmvb_targets_after_dev_tools DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(_xmvb_dev_tool_targets)
foreach(target_name IN LISTS _xmvb_targets_after_dev_tools)
  if (NOT target_name IN_LIST _xmvb_targets_before_dev_tools)
    list(APPEND _xmvb_dev_tool_targets ${target_name})
  endif()
endforeach()
xmvb_mark_targets_exclude_from_all(${_xmvb_dev_tool_targets})
if (_xmvb_dev_tool_targets)
  add_custom_target(xmvb_cpp_dev_tools DEPENDS ${_xmvb_dev_tool_targets})
endif()
endif()

if (XMVB_CPP_BUILD_DEV_TARGETS AND XMVB_CPP_HAVE_TEST_SOURCES)
  get_property(_xmvb_targets_before_dev_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
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

  get_property(_xmvb_targets_after_dev_tests DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  set(_xmvb_dev_test_targets)
  foreach(target_name IN LISTS _xmvb_targets_after_dev_tests)
    if (NOT target_name IN_LIST _xmvb_targets_before_dev_tests)
      list(APPEND _xmvb_dev_test_targets ${target_name})
    endif()
  endforeach()
  xmvb_mark_targets_exclude_from_all(${_xmvb_dev_test_targets})
  if (_xmvb_dev_test_targets)
    add_custom_target(xmvb_cpp_dev_tests DEPENDS ${_xmvb_dev_test_targets})
  endif()
endif()
