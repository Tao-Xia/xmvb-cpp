# VBSCF source ownership manifest.
#
# Keep ownership lists grouped by domain so the target composition mirrors the
# source tree.

set(XMVB_VBSCF_DETERMINANT_SOURCES
  vbscf/determinants/cofactor_differential.cpp
  vbscf/determinants/determinant_hamiltonian.cpp
  vbscf/determinants/determinant_overlap.cpp
  vbscf/determinants/determinant_pair_evaluator.cpp
  vbscf/determinants/same_spin_pair_cache.cpp
  vbscf/determinants/spin_pair_contractions.cpp
)

set(XMVB_VBSCF_STRUCTURE_SOURCES
  vbscf/structures/assembly/hamiltonian_overlap.cpp
  vbscf/structures/reference/overlap.cpp
  vbscf/structures/assembly/selected_coefficients.cpp
  vbscf/structures/evaluation/evaluator.cpp
  vbscf/structures/expansion/expander.cpp
  vbscf/structures/selection/subspace/selector.cpp
  vbscf/structures/selection/union_graph/rank_predictor.cpp
  vbscf/structures/selection/union_graph/screening.cpp
)

set(XMVB_VBSCF_ORBITAL_SOURCES
  vbscf/orbitals/preparation/preparer.cpp
  vbscf/orbitals/pullback/operator.cpp
  vbscf/orbitals/gauge/localized.cpp
  vbscf/orbitals/charts/partition.cpp
  vbscf/orbitals/charts/canonicalization.cpp
  vbscf/orbitals/charts/support_adapter.cpp
  vbscf/orbitals/charts/chart.cpp
  vbscf/orbitals/charts/layout.cpp
  vbscf/orbitals/gauge/support_preserving.cpp
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
  vbscf/integrals/active/active_space_two_electron_adjoint.cpp
  vbscf/integrals/active/active_space_two_electron_directional.cpp
  vbscf/integrals/active/active_space_pair_transforms.cpp
  vbscf/integrals/active/ao_pair_integral_operator.cpp
  vbscf/integrals/active/ri_active_space_two_electron_builder.cpp
  vbscf/integrals/active/prepared_active_space.cpp
  vbscf/integrals/active/two_electron_indexer.cpp
)

set(XMVB_VBSCF_DERIVATIVE_SOURCES
  vbscf/derivatives/gradient/active_space_gradient_helpers.cpp
  vbscf/derivatives/gradient/active_space_gradient_evaluator.cpp
  vbscf/derivatives/gradient/orbital_gradient_evaluator.cpp
  vbscf/derivatives/hessian/exact/apply.cpp
  vbscf/derivatives/hessian/exact/batch.cpp
  vbscf/derivatives/hessian/exact/ao_one_electron.cpp
  vbscf/derivatives/hessian/exact/operator.cpp
  vbscf/derivatives/hessian/context/response_cache.cpp
  vbscf/derivatives/hessian/responses/active_space/integral_direction.cpp
  vbscf/derivatives/hessian/responses/active_space/outer_response.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/backward.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/channels.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/pair_response.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/contractions.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions.cpp
  vbscf/derivatives/hessian/responses/orbital/preparation.cpp
  vbscf/derivatives/hessian/responses/same_spin/accepted_weights.cpp
  vbscf/derivatives/hessian/responses/same_spin/backward.cpp
  vbscf/derivatives/hessian/responses/same_spin/backward_kernels.cpp
  vbscf/derivatives/hessian/responses/same_spin/directional_weights.cpp
  vbscf/derivatives/hessian/responses/same_spin/directional_pair_cache.cpp
  vbscf/derivatives/hessian/responses/same_spin/local_weights.cpp
  vbscf/derivatives/hessian/responses/same_spin/pair_response.cpp
  vbscf/derivatives/hessian/responses/same_spin/tile_weights.cpp
  vbscf/derivatives/hessian/responses/same_spin/weight_kernels.cpp
  vbscf/derivatives/hessian/responses/structure/directional.cpp
)

set(XMVB_VBSCF_DIAGNOSTIC_SOURCES
  vbscf/diagnostics/orbital_chart_audit.cpp
)

set(XMVB_VBSCF_OPTIMIZATION_SOURCES
  vbscf/optimization/backends/lbfgs.cpp
  vbscf/optimization/backends/projected_gradient.cpp
  vbscf/optimization/backends/truncated_newton.cpp
  vbscf/optimization/globalization/line_search.cpp
  vbscf/optimization/driver/session.cpp
  vbscf/optimization/preconditioners/transported_lbfgs.cpp
  vbscf/optimization/objective/reduced_hvp.cpp
  vbscf/optimization/trust_region/retraction.cpp
  vbscf/optimization/trust_region/truncated_newton.cpp
  vbscf/optimization/objective/function.cpp
  vbscf/optimization/driver/optimizer.cpp
)

set(XMVB_VBSCF_WORKFLOW_SOURCES
  vbscf/workflow/vbscf_evaluator.cpp
)

# Keep the ownership manifest complete. Adding a VBSCF translation unit without
# assigning it to one of the domain lists above is a configuration error.
set(_xmvb_declared_vbscf_sources
  ${XMVB_VBSCF_DETERMINANT_SOURCES}
  ${XMVB_VBSCF_STRUCTURE_SOURCES}
  ${XMVB_VBSCF_ORBITAL_SOURCES}
  ${XMVB_VBSCF_AO_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_DERIVATIVE_SOURCES}
  ${XMVB_VBSCF_DIAGNOSTIC_SOURCES}
  ${XMVB_VBSCF_OPTIMIZATION_SOURCES}
  ${XMVB_VBSCF_WORKFLOW_SOURCES})
file(
  GLOB_RECURSE _xmvb_discovered_vbscf_sources
  CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}/.."
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
list(SORT _xmvb_declared_vbscf_sources)
list(SORT _xmvb_discovered_vbscf_sources)
if (NOT "${_xmvb_declared_vbscf_sources}" STREQUAL
    "${_xmvb_discovered_vbscf_sources}")
  message(FATAL_ERROR
    "src/vbscf/sources.cmake does not own every VBSCF .cpp exactly once.\n"
    "Declared: ${_xmvb_declared_vbscf_sources}\n"
    "Discovered: ${_xmvb_discovered_vbscf_sources}")
endif()
unset(_xmvb_declared_vbscf_sources)
unset(_xmvb_discovered_vbscf_sources)

# The standalone runtime prepares input decks and backend services on top of
# the numerical VBSCF library.  Keep that dependency one-way: canonical VBSCF
# sources may consume injected contracts, but must never include runtime headers.
file(
  GLOB_RECURSE _xmvb_vbscf_dependency_files
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.hpp")
foreach(_xmvb_vbscf_dependency_file IN LISTS _xmvb_vbscf_dependency_files)
  file(
    STRINGS "${_xmvb_vbscf_dependency_file}"
    _xmvb_runtime_includes
    REGEX "^[ \t]*#[ \t]*include[ \t]*\"runtime/")
  if (_xmvb_runtime_includes)
    message(FATAL_ERROR
      "Canonical VBSCF source includes a runtime header: "
      "${_xmvb_vbscf_dependency_file}\n${_xmvb_runtime_includes}")
  endif()
endforeach()
unset(_xmvb_runtime_includes)
unset(_xmvb_vbscf_dependency_file)
unset(_xmvb_vbscf_dependency_files)

set(XMVB_VBSCF_SOURCES
  core/linear_algebra/generalized_eigensolver.cpp
  ${XMVB_VBSCF_DETERMINANT_SOURCES}
  ${XMVB_VBSCF_STRUCTURE_SOURCES}
  ${XMVB_VBSCF_ORBITAL_SOURCES}
  ${XMVB_VBSCF_AO_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES}
  ${XMVB_VBSCF_DERIVATIVE_SOURCES}
  ${XMVB_VBSCF_DIAGNOSTIC_SOURCES}
  ${XMVB_VBSCF_OPTIMIZATION_SOURCES}
  ${XMVB_VBSCF_WORKFLOW_SOURCES}
)
