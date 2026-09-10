# Developer-only diagnostic and benchmark targets.
#
# Included by src/CMakeLists.txt after production libraries and runtime assets
# have been defined.

get_property(_xmvb_targets_before_dev_tools DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
add_executable(check_orbital_gradient
  tools/check_orbital_gradient.cpp)
target_link_libraries(check_orbital_gradient
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_orbital_gradient)

add_executable(benchmark_orbital_evaluation
  tools/benchmark_orbital_evaluation.cpp)
target_link_libraries(benchmark_orbital_evaluation
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(benchmark_orbital_evaluation)

add_executable(check_auxiliary_gradient
  tools/check_auxiliary_gradient.cpp)
target_link_libraries(check_auxiliary_gradient
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_auxiliary_gradient)

add_executable(check_reference_orbital_gradient
  tools/check_reference_orbital_gradient.cpp)
target_link_libraries(check_reference_orbital_gradient
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_reference_orbital_gradient)

add_executable(check_ao_h1e_ri_operator
  tools/check_ao_h1e_ri_operator.cpp)
target_link_libraries(check_ao_h1e_ri_operator
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_ao_h1e_ri_operator)

add_executable(inspect_ri_inactive_density
  tools/inspect_ri_inactive_density.cpp)
target_link_libraries(inspect_ri_inactive_density
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(inspect_ri_inactive_density)

add_executable(check_ri_ao_h1e_modes
  tools/check_ri_ao_h1e_modes.cpp)
target_link_libraries(check_ri_ao_h1e_modes
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_ri_ao_h1e_modes)

add_executable(benchmark_ri_ao_h1e_operator
  tools/benchmark_ri_ao_h1e_operator.cpp)
target_link_libraries(benchmark_ri_ao_h1e_operator
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(benchmark_ri_ao_h1e_operator)

add_executable(inspect_ao_h1e_backprop_symmetry
  tools/inspect_ao_h1e_backprop_symmetry.cpp)
target_link_libraries(inspect_ao_h1e_backprop_symmetry
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(inspect_ao_h1e_backprop_symmetry)

add_executable(check_reference_orbital_gradient_modes
  tools/check_reference_orbital_gradient_modes.cpp)
target_link_libraries(check_reference_orbital_gradient_modes
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_reference_orbital_gradient_modes)

add_executable(compare_exact_ri_energy_decomposition
  tools/compare_exact_ri_energy_decomposition.cpp)
target_link_libraries(compare_exact_ri_energy_decomposition
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(compare_exact_ri_energy_decomposition)

add_executable(check_active_space_gradient
  tools/check_active_space_gradient.cpp)
target_link_libraries(check_active_space_gradient
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_active_space_gradient)

add_executable(check_structure_builder_fast_path
  tools/check_structure_builder_fast_path.cpp)
target_link_libraries(check_structure_builder_fast_path
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_structure_builder_fast_path)

add_executable(check_exact_active_space_builders
  tools/check_exact_active_space_builders.cpp)
target_link_libraries(check_exact_active_space_builders
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_exact_active_space_builders)

add_executable(check_exact_two_electron_hvp
  tools/check_exact_two_electron_hvp.cpp)
target_link_libraries(check_exact_two_electron_hvp
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_exact_two_electron_hvp)

add_executable(check_exact_ctx_hvp
  tools/check_exact_ctx_hvp.cpp)
target_link_libraries(check_exact_ctx_hvp
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_exact_ctx_hvp)

add_executable(benchmark_exact_ctx_hvp
  tools/benchmark_exact_ctx_hvp.cpp)
target_link_libraries(benchmark_exact_ctx_hvp
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(benchmark_exact_ctx_hvp)

add_executable(audit_sparse_orbital_gauge
  tools/audit_sparse_orbital_gauge.cpp)
target_link_libraries(audit_sparse_orbital_gauge
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(audit_sparse_orbital_gauge)

include(${CMAKE_SOURCE_DIR}/tests/vbscf/targets.cmake)

add_executable(test_davidson core/linear_algebra/test_davidson.cpp)
target_link_libraries(test_davidson PRIVATE xmvb_vbscf xmvb_runtime)
xmvb_link_standalone_runtime(test_davidson)

add_executable(check_exact_ao_h1e_builder
  tools/check_exact_ao_h1e_builder.cpp)
target_link_libraries(check_exact_ao_h1e_builder
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_exact_ao_h1e_builder)

add_executable(check_structure_expansion_signs
  tools/check_structure_expansion_signs.cpp)
target_link_libraries(check_structure_expansion_signs
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_structure_expansion_signs)

add_executable(check_structure_overlap_against_raw_legacy
  tools/check_structure_overlap_against_raw_legacy.cpp)
target_link_libraries(check_structure_overlap_against_raw_legacy
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_structure_overlap_against_raw_legacy)

add_executable(inspect_input_deck_model
  tools/inspect_input_deck_model.cpp)
target_link_libraries(inspect_input_deck_model
  PRIVATE
    xmvb_runtime)
xmvb_link_standalone_runtime(inspect_input_deck_model)

add_executable(compare_molden_active_auxiliary
  tools/compare_molden_active_auxiliary.cpp)
target_link_libraries(compare_molden_active_auxiliary
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(compare_molden_active_auxiliary)

add_executable(dump_loaded_molden
  tools/dump_loaded_molden.cpp)
target_link_libraries(dump_loaded_molden
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(dump_loaded_molden)

add_executable(check_same_spin_overlap_gradient
  tools/check_same_spin_overlap_gradient.cpp)
target_link_libraries(check_same_spin_overlap_gradient
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_same_spin_overlap_gradient)

add_executable(check_structure_overlap_weight
  tools/check_structure_overlap_weight.cpp)
target_link_libraries(check_structure_overlap_weight
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_structure_overlap_weight)

add_executable(check_geminal_structure_overlap
  tools/check_geminal_structure_overlap.cpp)
target_link_libraries(check_geminal_structure_overlap
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_geminal_structure_overlap)

add_executable(check_union_graph_overlap_blocks
  tools/check_union_graph_overlap_blocks.cpp)
target_link_libraries(check_union_graph_overlap_blocks
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_union_graph_overlap_blocks)

add_executable(analyze_union_graph_rank_dataset
  tools/analyze_union_graph_rank_dataset.cpp)
target_link_libraries(analyze_union_graph_rank_dataset
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(analyze_union_graph_rank_dataset)

add_executable(analyze_metric_aware_graph
  tools/analyze_metric_aware_graph.cpp)
target_link_libraries(analyze_metric_aware_graph
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(analyze_metric_aware_graph)

add_executable(analyze_metric_aware_graph_dataset
  tools/analyze_metric_aware_graph_dataset.cpp)
target_link_libraries(analyze_metric_aware_graph_dataset
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(analyze_metric_aware_graph_dataset)

add_executable(analyze_exact_disconnected_overlap_dataset
  tools/analyze_exact_disconnected_overlap_dataset.cpp)
target_link_libraries(analyze_exact_disconnected_overlap_dataset
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(analyze_exact_disconnected_overlap_dataset)

add_executable(debug_exact_raw_vb_overlap_pair
  tools/debug_exact_raw_vb_overlap_pair.cpp)
target_link_libraries(debug_exact_raw_vb_overlap_pair
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(debug_exact_raw_vb_overlap_pair)

add_executable(validate_multi_leaf_open_state_one_electron
  tools/validate_multi_leaf_open_state_one_electron.cpp)
target_link_libraries(validate_multi_leaf_open_state_one_electron
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(validate_multi_leaf_open_state_one_electron)

add_executable(benchmark_union_graph_single_step
  tools/benchmark_union_graph_single_step.cpp)
target_link_libraries(benchmark_union_graph_single_step
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(benchmark_union_graph_single_step)

add_executable(check_direct_libcint_smoke
  tools/check_direct_libcint_smoke.cpp)
target_link_libraries(check_direct_libcint_smoke
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_direct_libcint_smoke)

add_executable(check_libcint_ri_smoke
  tools/check_libcint_ri_smoke.cpp)
target_link_libraries(check_libcint_ri_smoke
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_libcint_ri_smoke)

add_executable(compare_libcint_materialized_provider
  tools/compare_libcint_materialized_provider.cpp)
target_link_libraries(compare_libcint_materialized_provider
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(compare_libcint_materialized_provider)

add_executable(compare_ri_active_space_builder
  tools/compare_ri_active_space_builder.cpp)
target_link_libraries(compare_ri_active_space_builder
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(compare_ri_active_space_builder)

add_executable(check_ri_low_rank_unique_spin_pair
  tools/check_ri_low_rank_unique_spin_pair.cpp)
target_link_libraries(check_ri_low_rank_unique_spin_pair
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
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
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(check_active_overlap_split)

add_dependencies(check_orbital_gradient xmvb_runtime_assets)
add_dependencies(benchmark_orbital_evaluation xmvb_runtime_assets)
add_dependencies(check_auxiliary_gradient xmvb_runtime_assets)
add_dependencies(check_reference_orbital_gradient xmvb_runtime_assets)
add_dependencies(check_ao_h1e_ri_operator xmvb_runtime_assets)
add_dependencies(inspect_ri_inactive_density xmvb_runtime_assets)
add_dependencies(check_ri_ao_h1e_modes xmvb_runtime_assets)
add_dependencies(benchmark_ri_ao_h1e_operator xmvb_runtime_assets)
add_dependencies(inspect_ao_h1e_backprop_symmetry xmvb_runtime_assets)
add_dependencies(check_reference_orbital_gradient_modes xmvb_runtime_assets)
add_dependencies(compare_exact_ri_energy_decomposition xmvb_runtime_assets)
add_dependencies(check_active_space_gradient xmvb_runtime_assets)
add_dependencies(check_exact_active_space_builders xmvb_runtime_assets)
add_dependencies(check_exact_two_electron_hvp xmvb_runtime_assets)
add_dependencies(check_exact_ctx_hvp xmvb_runtime_assets)
add_dependencies(benchmark_exact_ctx_hvp xmvb_runtime_assets)

add_dependencies(check_structure_expansion_signs xmvb_runtime_assets)
add_dependencies(check_same_spin_overlap_gradient xmvb_runtime_assets)
add_dependencies(check_structure_overlap_weight xmvb_runtime_assets)
add_dependencies(check_geminal_structure_overlap xmvb_runtime_assets)
add_dependencies(check_union_graph_overlap_blocks xmvb_runtime_assets)
add_dependencies(analyze_union_graph_rank_dataset xmvb_runtime_assets)
add_dependencies(analyze_metric_aware_graph xmvb_runtime_assets)
add_dependencies(analyze_metric_aware_graph_dataset xmvb_runtime_assets)
add_dependencies(analyze_exact_disconnected_overlap_dataset xmvb_runtime_assets)
add_dependencies(debug_exact_raw_vb_overlap_pair xmvb_runtime_assets)
add_dependencies(validate_multi_leaf_open_state_one_electron xmvb_runtime_assets)
add_dependencies(benchmark_union_graph_single_step xmvb_runtime_assets)
add_dependencies(check_direct_libcint_smoke xmvb_runtime_assets)
add_dependencies(check_libcint_ri_smoke xmvb_runtime_assets)
add_dependencies(compare_libcint_materialized_provider xmvb_runtime_assets)
add_dependencies(compare_ri_active_space_builder xmvb_runtime_assets)
add_dependencies(check_ri_low_rank_unique_spin_pair xmvb_runtime_assets)
add_dependencies(check_active_overlap_split xmvb_runtime_assets)
get_property(_xmvb_targets_after_dev_tools DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
set(_xmvb_dev_tool_targets)
foreach(target_name IN LISTS _xmvb_targets_after_dev_tools)
  if (NOT target_name IN_LIST _xmvb_targets_before_dev_tools)
    list(APPEND _xmvb_dev_tool_targets ${target_name})
  endif()
endforeach()
xmvb_mark_targets_exclude_from_all(${_xmvb_dev_tool_targets})
if (_xmvb_dev_tool_targets)
  add_custom_target(xmvb_dev_tools DEPENDS ${_xmvb_dev_tool_targets})
endif()
