# VBSCF source ownership manifest.
#
# The physical migration from src/vb is intentionally incremental. Keeping the
# ownership lists here makes module boundaries visible in the build before the
# remaining implementation files move to their final directories.

set(XMVB_VBSCF_APPROXIMATION_SOURCES
  vbscf/approx/approx_vbscf_cluster_quotient.cpp
  vbscf/approx/approx_vbscf_evaluator.cpp
  vbscf/approx/approx_vbscf_metric.cpp
  vbscf/approx/approx_vbscf_pair_cluster.cpp
  vbscf/approx/approx_vbscf_resonance.cpp
  vbscf/approx/approx_vbscf_resonance_functional.cpp
)

set(XMVB_VBSCF_DETERMINANT_SOURCES
  vbscf/determinants/cofactor_differential.cpp
  vbscf/determinants/determinant_hamiltonian.cpp
  vbscf/determinants/determinant_overlap.cpp
  vbscf/determinants/determinant_pair_evaluator.cpp
  vbscf/determinants/same_spin_pair_cache.cpp
  vbscf/determinants/spin_pair_contractions.cpp
)

set(XMVB_VBSCF_STRUCTURE_SOURCES
  vbscf/structures/hamiltonian_overlap_builder.cpp
  vbscf/structures/selected_state_coefficients.cpp
  vbscf/structures/structure_evaluator.cpp
  vbscf/structures/structure_expander.cpp
  vbscf/structures/subspace_selector.cpp
  vbscf/structures/union_graph_rank_predictor.cpp
  vbscf/structures/union_graph_screening.cpp
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
  vbscf/integrals/ao/ri_integral_cache.cpp
  vbscf/integrals/ao/two_electron_pair_index.cpp
)

set(XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES
  vbscf/integrals/active/active_space_matrix_backpropagator.cpp
  vbscf/integrals/active/active_space_one_electron_builder.cpp
  vbscf/integrals/active/active_space_two_electron_backpropagator.cpp
  vbscf/integrals/active/active_space_two_electron_builder.cpp
  vbscf/integrals/active/active_space_two_electron_kernel.cpp
  vbscf/integrals/active/active_space_two_electron_response.cpp
  vbscf/integrals/active/ri_active_space_two_electron_builder.cpp
  vbscf/integrals/active/prepared_active_space.cpp
  vbscf/integrals/active/two_electron_indexer.cpp
)

set(XMVB_VBSCF_LEGACY_SOURCES
  vbscf/legacy/orbitals/jacobi_diagonalizer.cpp
  vbscf/legacy/orbitals/orbital_gradient_projector.cpp
  vbscf/legacy/structures/structure_overlap.cpp
)

set(XMVB_VBSCF_DERIVATIVE_SOURCES
  vbscf/derivatives/gradient/active_space_gradient_helpers.cpp
  vbscf/derivatives/gradient/active_space_gradient_evaluator.cpp
  vbscf/derivatives/gradient/orbital_gradient_evaluator.cpp
  vbscf/derivatives/hessian/exact_hvp_operator.cpp
  vbscf/derivatives/hessian/structure_response_cache.cpp
  vbscf/derivatives/hessian/responses/opposite_spin_channels.cpp
  vbscf/derivatives/hessian/responses/opposite_spin_response.cpp
  vbscf/derivatives/hessian/responses/same_spin_response.cpp
)

set(XMVB_VBSCF_DIAGNOSTIC_SOURCES
  vbscf/diagnostics/hvp_memory_report.cpp
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
  vbscf/adaptive/structure_space_optimizer.cpp
)

# DeepVBH is a separate compatibility target. It may depend on VBSCF and the
# standalone runtime, but neither production layer may depend on it.
set(XMVB_DEEPVBH_SOURCES
  vb/model/deepvbh_jax_inference_runner.cpp
  vb/scf/deepvbh_onnx_direct_final_optimizer.cpp
  vb/scf/deepvbh_onnx_hybrid_optimizer.cpp
)

set(XMVB_CPP_CORE_SOURCES
  core/linear_algebra/generalized_eigensolver.cpp
  ${XMVB_VBSCF_APPROXIMATION_SOURCES}
  ${XMVB_VBSCF_DETERMINANT_SOURCES}
  ${XMVB_VBSCF_STRUCTURE_SOURCES}
  ${XMVB_VBSCF_ORBITAL_SOURCES}
  ${XMVB_VBSCF_AO_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_DERIVATIVE_SOURCES}
  ${XMVB_VBSCF_DIAGNOSTIC_SOURCES}
  ${XMVB_VBSCF_OPTIMIZATION_SOURCES}
  ${XMVB_VBSCF_WORKFLOW_SOURCES}
  ${XMVB_VBSCF_ADAPTIVE_SOURCES}
  ${XMVB_VBSCF_LEGACY_SOURCES}
)
