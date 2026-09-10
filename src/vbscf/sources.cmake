# VBSCF source ownership manifest.
#
# Keep ownership lists grouped by domain so the target composition mirrors the
# source tree.

set(XMVB_VBSCF_DETERMINANT_SOURCES
  vbscf/determinants/algebra/cofactor_differential.cpp
  vbscf/determinants/algebra/hamiltonian.cpp
  vbscf/determinants/algebra/overlap.cpp
  vbscf/determinants/pairs/evaluator.cpp
  vbscf/determinants/pairs/same_spin_cache.cpp
  vbscf/determinants/pairs/contractions.cpp
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
  vbscf/integrals/ao/one_electron/backpropagator.cpp
  vbscf/integrals/ao/one_electron/builder.cpp
  vbscf/integrals/ao/one_electron/graph_operator.cpp
  vbscf/integrals/ao/one_electron/ri_operator.cpp
  vbscf/integrals/ao/libcint/validation.cpp
  vbscf/integrals/ao/ri/cache.cpp
  vbscf/integrals/ao/pairs/two_electron_index.cpp
)

set(XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES
  vbscf/integrals/active/matrix/backpropagator.cpp
  vbscf/integrals/active/one_electron/builder.cpp
  vbscf/integrals/active/two_electron/response/backpropagator.cpp
  vbscf/integrals/active/two_electron/construction/builder.cpp
  vbscf/integrals/active/two_electron/construction/kernel.cpp
  vbscf/integrals/active/two_electron/response/adjoint.cpp
  vbscf/integrals/active/two_electron/response/directional.cpp
  vbscf/integrals/active/two_electron/transformation/pair_transforms.cpp
  vbscf/integrals/active/two_electron/transformation/ao_pair_operator.cpp
  vbscf/integrals/active/two_electron/construction/ri_builder.cpp
  vbscf/integrals/active/preparation/space.cpp
  vbscf/integrals/active/two_electron/construction/indexer.cpp
)

set(XMVB_VBSCF_DERIVATIVE_SOURCES
  vbscf/derivatives/gradient/active_space/helpers.cpp
  vbscf/derivatives/gradient/active_space/evaluator.cpp
  vbscf/derivatives/gradient/orbital/evaluator.cpp
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
  vbscf/diagnostics/orbitals/chart_audit.cpp
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
  vbscf/workflow/evaluator.cpp
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
  core/eigensolver.cpp
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
