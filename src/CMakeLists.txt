file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" XMVB_CONDA_PREFIX)

if (NOT DEFINED XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND)
  set(
    XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
    OFF
    CACHE BOOL
    "Enable the legacy Fortran L-BFGS backend inside the C++ VB optimizer."
    FORCE)
endif()

if (NOT DEFINED XMVB_CPP_BUILD_DEV_TARGETS)
  set(
    XMVB_CPP_BUILD_DEV_TARGETS
    OFF
    CACHE BOOL
    "Build C++ comparison, diagnostic, and test targets in addition to xmvb-cpp.exe."
    FORCE)
endif()

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

# The standalone path is now pure C++. Keep the old cache variables pinned off
# so existing build directories cannot silently pull runtime_c back in.
set(
  XMVB_CPP_BUILD_LEGACY_HF_RUNTIME
  OFF
  CACHE BOOL
  "Build the legacy standalone HF/XINT runtime used by legacy orbital guesses and HF diagnostics."
  FORCE)

set(
  XMVB_CPP_BUILD_RUNTIME_C_TOOLS
  OFF
  CACHE BOOL
  "Build legacy comparison/diagnostic tools that still depend on runtime_c."
  FORCE)

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
  vb/approx/approx_vbscf_cluster_quotient.cpp
  vb/approx/approx_vbscf_evaluator.cpp
  vb/approx/approx_vbscf_metric.cpp
  vb/approx/approx_vbscf_pair_cluster.cpp
  vb/approx/approx_vbscf_resonance.cpp
  vb/approx/approx_vbscf_resonance_functional.cpp
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
  vb/matrices/two_electron_indexer.cpp
  vb/matrices/union_graph_screening.cpp
  vb/model/deepvbh_jax_inference_runner.cpp
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
  vb/orbital/libcint_input_utils.cpp
  vb/orbital/legacy_style_orbital_gradient_projector.cpp
  vb/orbital/legacy_jacobi_diagonalizer.cpp
  vb/orbital/localized_representative_selector.cpp
  vb/orbital/nonredundant_optimizer_input_adapter.cpp
  vb/orbital/nonredundant_orbital_space.cpp
  vb/orbital/sparse_orbital_gauge_audit.cpp
  vb/orbital/orbital_parameter_codec.cpp
  vb/orbital/sparse_orbital_parameter_view.cpp
  vb/orbital/support_aware_mo_gauge_fix.cpp
  vb/orbital/ri_active_space_two_electron_builder.cpp
  vb/scf/selected_state_determinant_matrices.cpp
  vb/scf/opposite_spin_matrix_channels.cpp
  vb/scf/opposite_spin_matrix_backward.cpp
  vb/scf/same_spin_matrix_backward.cpp
  vb/scf/exact_ctx_memory_accounting.cpp
  vb/scf/exact_orbital_second_order_operator.cpp
  vb/scf/exact_orbital_second_order_operator_outer_response_cache.cpp
  vb/scf/cpp_active_space_gradient_result_utils.cpp
  vb/scf/cpp_active_space_gradient_evaluator.cpp
  vb/scf/adaptive_structure_space_optimizer.cpp
  vb/scf/deepvbh_onnx_direct_final_optimizer.cpp
  vb/scf/deepvbh_onnx_hybrid_optimizer.cpp
  vb/scf/cpp_orbital_gradient_evaluator.cpp
  vb/scf/cpp_vb_scf_evaluator.cpp
  vb/scf/orbital_objective.cpp
  vb/scf/cpp_vb_scf_optimizer.cpp
  )

set(XMVB_CPP_RUNTIME_CORE_SOURCES
  runtime/cpp_block_guess_builder.cpp
  runtime/cpp_closed_shell_fock_builder.cpp
  runtime/cpp_initial_guess_builder.cpp
  runtime/generated_raw_structure_builder.cpp
  runtime/input_deck_model.cpp
  runtime/input_deck_metadata.cpp
  runtime/input_deck_primary_basis_builder.cpp
  runtime/input_deck_orbital_support_builder.cpp
  runtime/input_deck_raw_structures.cpp
  runtime/molden_file_writer.cpp
  runtime/cpp_restricted_hartree_fock.cpp
  runtime/cpp_vb_input_loader.cpp
  runtime/libcint_auxiliary_basis_builder.cpp
  runtime/libcint_direct_shell_evaluator.cpp
  runtime/libcint_materialized_integral_provider.cpp
  runtime/libcint_ri_integral_provider.cpp
  runtime/materialized_ao_integral_input_builder.cpp
)

set(XMVB_CPP_RUNTIME_SOURCES ${XMVB_CPP_RUNTIME_CORE_SOURCES})

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
    ${CMAKE_BINARY_DIR}/src/vbutil
    ${CMAKE_BINARY_DIR}/src/vbscf
    ${CMAKE_BINARY_DIR}/src/vbci
    ${CMAKE_BINARY_DIR}/src/vbpt2
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${CINT_INCLUDE})
target_link_libraries(xmvb_cpp_runtime PUBLIC xmvb_cpp_vb xmvb_cpp_openmp)

function(xmvb_link_standalone_runtime target_name)
  target_link_libraries(${target_name}
    PRIVATE
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

add_executable(dump_approx_vbscf_features
  tools/dump_approx_vbscf_features.cpp)
target_link_libraries(dump_approx_vbscf_features
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(dump_approx_vbscf_features)

add_executable(validate_approx_vbscf_batch
  tools/validate_approx_vbscf_batch.cpp)
target_link_libraries(validate_approx_vbscf_batch
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(validate_approx_vbscf_batch)

add_executable(run_approx_vbscf_pair_scf
  tools/run_approx_vbscf_pair_scf.cpp)
target_link_libraries(run_approx_vbscf_pair_scf
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(run_approx_vbscf_pair_scf)

add_executable(benchmark_approx_vbscf_scaling
  tools/benchmark_approx_vbscf_scaling.cpp)
target_link_libraries(benchmark_approx_vbscf_scaling
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(benchmark_approx_vbscf_scaling)

add_executable(compare_approx_structure_importance
  tools/compare_approx_structure_importance.cpp)
target_link_libraries(compare_approx_structure_importance
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(compare_approx_structure_importance)

add_executable(analyze_approx_vbscf_resonance
  tools/analyze_approx_vbscf_resonance.cpp)
target_link_libraries(analyze_approx_vbscf_resonance
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(analyze_approx_vbscf_resonance)

add_executable(check_cpp_active_space_gradient
  tools/check_cpp_active_space_gradient.cpp)
target_link_libraries(check_cpp_active_space_gradient
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_cpp_active_space_gradient)

add_executable(check_structure_builder_fast_path
  tools/check_structure_builder_fast_path.cpp)
target_link_libraries(check_structure_builder_fast_path
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_structure_builder_fast_path)

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

add_executable(audit_sparse_orbital_gauge
  tools/audit_sparse_orbital_gauge.cpp)
target_link_libraries(audit_sparse_orbital_gauge
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(audit_sparse_orbital_gauge)

add_executable(test_davidson core/linear_algebra/test_davidson.cpp)
target_link_libraries(test_davidson PRIVATE xmvb_cpp_vb xmvb_cpp_runtime)
xmvb_link_standalone_runtime(test_davidson)

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

add_executable(inspect_input_deck_model
  tools/inspect_input_deck_model.cpp)
target_link_libraries(inspect_input_deck_model
  PRIVATE
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(inspect_input_deck_model)

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

add_executable(check_ri_low_rank_unique_spin_pair
  tools/check_ri_low_rank_unique_spin_pair.cpp)
target_link_libraries(check_ri_low_rank_unique_spin_pair
  PRIVATE
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(check_ri_low_rank_unique_spin_pair)

add_executable(benchmark_low_rank_synthetic
  tools/benchmark_low_rank_synthetic.cpp)
target_include_directories(benchmark_low_rank_synthetic
  PRIVATE
    ${EIGEN3_INCLUDE_DIR})

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
add_dependencies(dump_approx_vbscf_features xmvb_cpp_runtime_assets)
add_dependencies(validate_approx_vbscf_batch xmvb_cpp_runtime_assets)
add_dependencies(run_approx_vbscf_pair_scf xmvb_cpp_runtime_assets)
add_dependencies(benchmark_approx_vbscf_scaling xmvb_cpp_runtime_assets)
add_dependencies(compare_approx_structure_importance xmvb_cpp_runtime_assets)
add_dependencies(analyze_approx_vbscf_resonance xmvb_cpp_runtime_assets)
add_dependencies(check_cpp_active_space_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_exact_active_space_builders xmvb_cpp_runtime_assets)
add_dependencies(check_exact_two_electron_hvp xmvb_cpp_runtime_assets)
add_dependencies(check_exact_ctx_hvp xmvb_cpp_runtime_assets)
add_dependencies(benchmark_exact_ctx_hvp xmvb_cpp_runtime_assets)

if (BUILD_TESTING)
  add_test(
    NAME exact_ctx_hvp_f2_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/src/test_molecule/F2.xmi
      --step 1e-4
      --probe full
      --max-rel-error 1e-7)
  set_tests_properties(
    exact_ctx_hvp_f2_finite_difference
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
endif()
add_dependencies(check_structure_expansion_signs xmvb_cpp_runtime_assets)
add_dependencies(check_same_spin_overlap_gradient xmvb_cpp_runtime_assets)
add_dependencies(check_structure_overlap_weight xmvb_cpp_runtime_assets)
add_dependencies(check_geminal_structure_overlap xmvb_cpp_runtime_assets)
add_dependencies(check_union_graph_overlap_blocks xmvb_cpp_runtime_assets)
add_dependencies(analyze_union_graph_rank_dataset xmvb_cpp_runtime_assets)
add_dependencies(analyze_metric_aware_graph xmvb_cpp_runtime_assets)
add_dependencies(analyze_metric_aware_graph_dataset xmvb_cpp_runtime_assets)
add_dependencies(analyze_exact_disconnected_overlap_dataset xmvb_cpp_runtime_assets)
add_dependencies(debug_exact_raw_vb_overlap_pair xmvb_cpp_runtime_assets)
add_dependencies(validate_multi_leaf_open_state_one_electron xmvb_cpp_runtime_assets)
add_dependencies(benchmark_union_graph_single_step xmvb_cpp_runtime_assets)
add_dependencies(check_direct_libcint_smoke xmvb_cpp_runtime_assets)
add_dependencies(check_libcint_ri_smoke xmvb_cpp_runtime_assets)
add_dependencies(compare_libcint_materialized_provider xmvb_cpp_runtime_assets)
add_dependencies(compare_ri_active_space_builder xmvb_cpp_runtime_assets)
add_dependencies(check_ri_low_rank_unique_spin_pair xmvb_cpp_runtime_assets)
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

  add_executable(generalized_eigensolver_test
    tests/unit/generalized_eigensolver_test.cpp)
  target_link_libraries(generalized_eigensolver_test PRIVATE xmvb_cpp_vb)
  add_test(NAME generalized_eigensolver_test COMMAND generalized_eigensolver_test)
  set_tests_properties(generalized_eigensolver_test PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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
