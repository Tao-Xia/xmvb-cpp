# VBSCF source ownership manifest.
#
# The physical migration from src/vb is intentionally incremental. Keeping the
# ownership lists here makes module boundaries visible in the build before the
# remaining implementation files move to their final directories.

set(XMVB_VBSCF_APPROXIMATION_SOURCES
  vb/approx/approx_vbscf_cluster_quotient.cpp
  vb/approx/approx_vbscf_evaluator.cpp
  vb/approx/approx_vbscf_metric.cpp
  vb/approx/approx_vbscf_pair_cluster.cpp
  vb/approx/approx_vbscf_resonance.cpp
  vb/approx/approx_vbscf_resonance_functional.cpp
)

set(XMVB_VBSCF_DETERMINANT_AND_STRUCTURE_SOURCES
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
  vb/matrices/cofactor_differential.cpp
  vb/matrices/structure_matrix_evaluator.cpp
  vb/matrices/two_electron_indexer.cpp
  vb/matrices/union_graph_screening.cpp
  vb/scf/selected_state_determinant_matrices.cpp
)

set(XMVB_VBSCF_ORBITAL_SOURCES
  vbscf/orbitals/orbital_parameter_codec.cpp
  vbscf/orbitals/orbital_preparer.cpp
  vbscf/orbitals/orbital_pullback.cpp
  vbscf/orbitals/gauge/localized_representative.cpp
  vbscf/orbitals/charts/orbital_block_partition.cpp
  vbscf/orbitals/charts/support_layout_adapter.cpp
  vbscf/orbitals/charts/orbital_chart.cpp
  vbscf/orbitals/charts/sparse_parameter_layout.cpp
  vbscf/orbitals/gauge/support_preserving_gauge.cpp
)

set(XMVB_VBSCF_AO_INTEGRAL_SOURCES
  vbscf/integrals/ao/ao_effective_one_electron_backpropagator.cpp
  vbscf/integrals/ao/ao_effective_one_electron_builder.cpp
  vbscf/integrals/ao/ao_effective_one_electron_graph_operator.cpp
  vbscf/integrals/ao/ao_effective_one_electron_ri_operator.cpp
  vbscf/integrals/ao/libcint_input_validation.cpp
  vbscf/integrals/ao/two_electron_pair_index.cpp
)

set(XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES
  vbscf/integrals/active/active_space_matrix_backpropagator.cpp
  vbscf/integrals/active/active_space_one_electron_builder.cpp
  vbscf/integrals/active/active_space_two_electron_backpropagator.cpp
  vbscf/integrals/active/active_space_two_electron_builder.cpp
  vbscf/integrals/active/active_two_electron_operator.cpp
  vbscf/integrals/active/ri_active_space_two_electron_builder.cpp
)

set(XMVB_VBSCF_LEGACY_SOURCES
  vbscf/legacy/orbitals/jacobi_diagonalizer.cpp
  vbscf/legacy/orbitals/orbital_gradient_projector.cpp
)

set(XMVB_VBSCF_DERIVATIVE_SOURCES
  vb/scf/opposite_spin_matrix_channels.cpp
  vb/scf/opposite_spin_matrix_backward.cpp
  vb/scf/same_spin_matrix_backward.cpp
  vb/scf/exact_ctx_memory_accounting.cpp
  vb/scf/exact_orbital_second_order_operator.cpp
  vb/scf/exact_orbital_second_order_operator_outer_response_cache.cpp
  vb/scf/cpp_active_space_gradient_result_utils.cpp
  vb/scf/cpp_active_space_gradient_evaluator.cpp
  vb/scf/cpp_orbital_gradient_evaluator.cpp
)

set(XMVB_VBSCF_DIAGNOSTIC_SOURCES
  vbscf/diagnostics/orbital_chart_audit.cpp
)

set(XMVB_VBSCF_OPTIMIZATION_SOURCES
  vbscf/optimization/vbscf_objective.cpp
  vbscf/optimization/vbscf_optimizer.cpp
)

set(XMVB_VBSCF_WORKFLOW_SOURCES
  vbscf/workflow/vbscf_evaluator.cpp
)

set(XMVB_VBSCF_ADAPTIVE_SOURCES
  vb/scf/adaptive_structure_space_optimizer.cpp
)

# Experimental models remain linked for compatibility, but production VBSCF
# modules must never depend on them. They will become an optional target after
# the core dependency cycles have been removed.
set(XMVB_VBSCF_EXPERIMENTAL_SOURCES
  vb/model/deepvbh_jax_inference_runner.cpp
  vb/scf/deepvbh_onnx_direct_final_optimizer.cpp
  vb/scf/deepvbh_onnx_hybrid_optimizer.cpp
)

set(XMVB_CPP_CORE_SOURCES
  core/linear_algebra/generalized_eigensolver.cpp
  ${XMVB_VBSCF_APPROXIMATION_SOURCES}
  ${XMVB_VBSCF_DETERMINANT_AND_STRUCTURE_SOURCES}
  ${XMVB_VBSCF_ORBITAL_SOURCES}
  ${XMVB_VBSCF_AO_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_DERIVATIVE_SOURCES}
  ${XMVB_VBSCF_DIAGNOSTIC_SOURCES}
  ${XMVB_VBSCF_OPTIMIZATION_SOURCES}
  ${XMVB_VBSCF_WORKFLOW_SOURCES}
  ${XMVB_VBSCF_ADAPTIVE_SOURCES}
  ${XMVB_VBSCF_EXPERIMENTAL_SOURCES}
  ${XMVB_VBSCF_LEGACY_SOURCES}
)
