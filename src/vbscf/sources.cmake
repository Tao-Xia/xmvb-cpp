# VBSCF source ownership manifest.
#
# Keep ownership lists grouped by domain so the target composition mirrors the
# source tree.

set(XMVB_VBSCF_DETERMINANT_SOURCES
  vbscf/determinants/algebra/cofactor_differential.cpp
  vbscf/determinants/algebra/hamiltonian.cpp
  vbscf/determinants/algebra/overlap.cpp
  vbscf/determinants/pairs/evaluator.cpp
  vbscf/determinants/pairs/accepted_action.cpp
  vbscf/determinants/pairs/accepted_tile.cpp
  vbscf/determinants/pairs/same_spin_cache.cpp
  vbscf/determinants/pairs/contractions.cpp
)

set(XMVB_VBSCF_STRUCTURE_SOURCES
  vbscf/structures/spinor/strings.cpp
  vbscf/structures/spinor/sigma.cpp
  vbscf/structures/spinor/transform.cpp
  vbscf/structures/spinor/basis.cpp
  vbscf/structures/spinor/action.cpp
  vbscf/structures/analysis/spin.cpp
  vbscf/structures/assembly/action.cpp
  vbscf/structures/assembly/states.cpp
  vbscf/structures/assembly/transport_tiles.cpp
  vbscf/structures/assembly/orthogonal_tiles.cpp
  vbscf/structures/assembly/operator_first.cpp
  vbscf/structures/assembly/operator_first_jet.cpp
  vbscf/structures/assembly/product_jet.cpp
  vbscf/structures/assembly/overlap_tiles.cpp
  vbscf/structures/assembly/source_tiles.cpp
  vbscf/structures/assembly/source_response.cpp
  vbscf/structures/assembly/source_direction.cpp
  vbscf/structures/assembly/source_image.cpp
  vbscf/structures/assembly/sources.cpp
  vbscf/structures/assembly/hamiltonian_overlap.cpp
  vbscf/structures/reference/overlap.cpp
  vbscf/structures/assembly/selected_coefficients.cpp
  vbscf/structures/evaluation/evaluator.cpp
  vbscf/structures/expansion/expander.cpp
  vbscf/structures/orthogonal_ci/exterior_transform.cpp
  vbscf/structures/orthogonal_ci/exterior_rows.cpp
  vbscf/structures/orthogonal_ci/metric.cpp
  vbscf/structures/overlap/representation.cpp
  vbscf/structures/overlap/spin_metric.cpp
  vbscf/structures/orthogonal_ci/integrals.cpp
  vbscf/structures/selection/subspace/selector.cpp
  vbscf/structures/selection/union_graph/rank_predictor.cpp
  vbscf/structures/selection/union_graph/screening.cpp
)

set(XMVB_VBSCF_ORBITAL_SOURCES
  vbscf/orbitals/spinor/frame.cpp
  vbscf/orbitals/preparation/preparer.cpp
  vbscf/orbitals/preparation/overlap_pullback.cpp
  vbscf/orbitals/pullback/operator.cpp
  vbscf/orbitals/gauge/localized.cpp
  vbscf/orbitals/charts/partition.cpp
  vbscf/orbitals/charts/canonicalization.cpp
  vbscf/orbitals/charts/support_adapter.cpp
  vbscf/orbitals/charts/chart.cpp
  vbscf/orbitals/charts/whitening.cpp
  vbscf/orbitals/charts/raw.cpp
  vbscf/orbitals/charts/rotation.cpp
  vbscf/orbitals/charts/layout.cpp
  vbscf/orbitals/charts/physical_metric.cpp
  vbscf/orbitals/gauge/support_preserving.cpp
)

set(XMVB_VBSCF_AO_INTEGRAL_SOURCES
  vbscf/integrals/ao/one_electron/backpropagator.cpp
  vbscf/integrals/ao/one_electron/builder.cpp
  vbscf/integrals/ao/ri/cache.cpp
)

set(XMVB_VBSCF_ACTIVE_INTEGRAL_SOURCES
  vbscf/integrals/active/space.cpp
)

set(XMVB_VBSCF_DERIVATIVE_SOURCES
  vbscf/derivatives/gradient/active_space/backend.cpp
  vbscf/derivatives/gradient/active_space/determinant_backend.cpp
  vbscf/derivatives/gradient/active_space/helpers.cpp
  vbscf/derivatives/gradient/active_space/evaluator.cpp
  vbscf/derivatives/gradient/orbital/evaluator.cpp
  vbscf/derivatives/hessian/coupled/coupling.cpp
  vbscf/derivatives/hessian/coupled/operator.cpp
  vbscf/derivatives/hessian/coupled/structure.cpp
  vbscf/derivatives/hessian/exact/apply.cpp
  vbscf/derivatives/hessian/exact/active_gradient.cpp
  vbscf/derivatives/hessian/exact/batch.cpp
  vbscf/derivatives/hessian/exact/direction.cpp
  vbscf/derivatives/hessian/exact/structure_response_schur.cpp
  vbscf/derivatives/hessian/exact/ao_one_electron.cpp
  vbscf/derivatives/hessian/exact/operator.cpp
  vbscf/derivatives/hessian/context/response_cache.cpp
  vbscf/derivatives/hessian/context/metric.cpp
    vbscf/derivatives/hessian/responses/active_space/integral_direction.cpp
    vbscf/derivatives/hessian/responses/active_space/local_tile_accumulator.cpp
  vbscf/derivatives/hessian/responses/active_space/outer_response.cpp
  vbscf/derivatives/hessian/responses/active_space/ri_factor_adjoint.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/backward.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/pair_response.cpp
    vbscf/derivatives/hessian/responses/opposite_spin/contractions.cpp
    vbscf/derivatives/hessian/responses/opposite_spin/local_tile_accumulator.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/overlap_contractions.cpp
  vbscf/derivatives/hessian/responses/opposite_spin/selected_state_pair_graph.cpp
  vbscf/derivatives/hessian/responses/orbital/preparation.cpp
  vbscf/derivatives/hessian/responses/same_spin/backward.cpp
  vbscf/derivatives/hessian/responses/same_spin/backward_kernels.cpp
  vbscf/derivatives/hessian/responses/same_spin/local_tile_accumulator.cpp
  vbscf/derivatives/hessian/responses/same_spin/pair_response.cpp
  vbscf/derivatives/hessian/responses/same_spin/pair_tile_stream.cpp
  vbscf/derivatives/hessian/responses/same_spin/tile_weights.cpp
  vbscf/derivatives/hessian/responses/same_spin/weight_kernels.cpp
  vbscf/derivatives/hessian/responses/structure/directional.cpp
)

set(XMVB_VBSCF_DIAGNOSTIC_SOURCES
  vbscf/diagnostics/orbitals/chart_audit.cpp
  vbscf/diagnostics/orbitals/ciah_keyframe.cpp
  vbscf/diagnostics/orbitals/step_geometry.cpp
)

set(XMVB_VBSCF_OPTIMIZATION_SOURCES
  vbscf/optimization/coupled/ci_transport.cpp
  vbscf/optimization/coupled/affine.cpp
  vbscf/optimization/backends/ciah_step.cpp
  vbscf/optimization/backends/plbfgs.cpp
  vbscf/optimization/backends/ciah.cpp
  vbscf/optimization/backends/projected_gradient.cpp
  vbscf/optimization/backends/truncated_newton.cpp
  vbscf/optimization/globalization/line_search.cpp
  vbscf/optimization/driver/session.cpp
  vbscf/optimization/neo/coordinates.cpp
  vbscf/optimization/neo/curvature.cpp
  vbscf/optimization/preconditioners/hessian_diagonal.cpp
  vbscf/optimization/preconditioners/density.cpp
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

# External input, backend, guess, output, and CLI components sit above the
# numerical VBSCF library. Keep that dependency one-way: canonical VBSCF
# sources may consume injected contracts, but must never include their headers.
file(
  GLOB_RECURSE _xmvb_vbscf_dependency_files
  CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/*.hpp")
foreach(_xmvb_vbscf_dependency_file IN LISTS _xmvb_vbscf_dependency_files)
  file(
    STRINGS "${_xmvb_vbscf_dependency_file}"
    _xmvb_external_component_includes
    REGEX "^[ \t]*#[ \t]*include[ \t]*\"(cli|guess|input|libcint|output)/")
  if (_xmvb_external_component_includes)
    message(FATAL_ERROR
      "Canonical VBSCF source includes an external component header: "
      "${_xmvb_vbscf_dependency_file}\n${_xmvb_external_component_includes}")
  endif()
endforeach()
unset(_xmvb_external_component_includes)
unset(_xmvb_vbscf_dependency_file)
unset(_xmvb_vbscf_dependency_files)

set(XMVB_VBSCF_SOURCES
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
