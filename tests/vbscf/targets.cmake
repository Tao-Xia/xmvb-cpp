# VBSCF unit and numerical-regression targets.
#
# This file is included from src/CMakeLists.txt while developer targets are
# enabled.  Keeping the target ownership beside the tests prevents the main
# production build file from accumulating test-specific wiring.

set(_xmvb_vbscf_unit_targets
  test_active_two_electron_sparse
  test_cofactor_differential
  test_coupled_newton_cauchy
  test_coupled_newton_model_reduction
  test_coupled_newton_operator
  test_coupled_newton_preconditioner
  test_coupled_projection_cache
  test_coupled_projected_model
  test_coupled_projected_trust
  test_coupled_subspace_solver
  test_coupled_workspace
  test_coupled_newton_defect
  test_coupled_newton_spectral
  test_curvature_decomposition
  test_davidson
  test_eigen_response
  test_localized_representative_selector
  test_minres
  test_normalized_orbital_curvature
  test_oeo_normalization_pullback
  test_opposite_spin_pair_graph
  test_orthogonal_direct_ci
  test_orthonormal_hvp_basis
  test_positive_ritz_secants
  test_projected_orbital_surrogate
  test_reduced_hessian_reference
  test_ri_ao_h1e_hvp
  test_ri_active_two_electron_response
  test_sparse_orbital_quotient
  test_support_preserving_gauge
  test_spectral_trust_region
  test_orbital_block_partition)

foreach(target_name IN LISTS _xmvb_vbscf_unit_targets)
  add_executable(
    ${target_name}
    ${CMAKE_SOURCE_DIR}/tests/vbscf/unit/${target_name}.cpp)
  target_link_libraries(${target_name} PRIVATE xmvb_vbscf)
endforeach()

if (BUILD_TESTING)
  add_test(NAME active_two_electron_sparse COMMAND test_active_two_electron_sparse)
  add_test(NAME orbital_block_partition COMMAND test_orbital_block_partition)
  add_test(NAME curvature_decomposition COMMAND test_curvature_decomposition)
  add_test(NAME oeo_normalization_pullback COMMAND test_oeo_normalization_pullback)
  add_test(NAME opposite_spin_pair_graph COMMAND test_opposite_spin_pair_graph)
  add_test(NAME orthogonal_direct_ci COMMAND test_orthogonal_direct_ci)
  add_test(NAME cofactor_differential COMMAND test_cofactor_differential)
  add_test(NAME coupled_newton_cauchy COMMAND test_coupled_newton_cauchy)
  set_tests_properties(coupled_newton_cauchy PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(
    NAME coupled_newton_model_reduction
    COMMAND test_coupled_newton_model_reduction)
  set_tests_properties(coupled_newton_model_reduction PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_newton_operator COMMAND test_coupled_newton_operator)
  set_tests_properties(coupled_newton_operator PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(
    NAME coupled_newton_preconditioner
    COMMAND test_coupled_newton_preconditioner)
  set_tests_properties(coupled_newton_preconditioner PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_projection_cache COMMAND test_coupled_projection_cache)
  set_tests_properties(coupled_projection_cache PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_projected_model COMMAND test_coupled_projected_model)
  set_tests_properties(coupled_projected_model PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_projected_trust COMMAND test_coupled_projected_trust)
  set_tests_properties(coupled_projected_trust PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_subspace_solver COMMAND test_coupled_subspace_solver)
  set_tests_properties(coupled_subspace_solver PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_workspace COMMAND test_coupled_workspace)
  set_tests_properties(coupled_workspace PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_newton_defect COMMAND test_coupled_newton_defect)
  set_tests_properties(coupled_newton_defect PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME coupled_newton_spectral COMMAND test_coupled_newton_spectral)
  set_tests_properties(coupled_newton_spectral PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME davidson COMMAND test_davidson)
  add_test(NAME eigen_response COMMAND test_eigen_response)
  add_test(NAME minres COMMAND test_minres)
  set_tests_properties(minres PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(
    NAME localized_representative_selector
    COMMAND test_localized_representative_selector)
  set_tests_properties(davidson PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  set_tests_properties(oeo_normalization_pullback PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

  add_test(NAME curvature_audit_f2 COMMAND benchmark_exact_ctx_hvp
    ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
    --curvature-audit-directions 6)
  set_tests_properties(curvature_audit_f2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "audit_total_hvp_calls =")

  add_test(NAME audit_newton_step_f2 COMMAND audit_newton_step
    ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
    --subspace-dimension 4 --trust-radius 0.1
    --finite-difference-step 1e-5 --eigensolver davidson)
  set_tests_properties(audit_newton_step_f2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "directional_hvp_fd_relative =")

  add_test(NAME dense_reduced_hessian_f2 COMMAND benchmark_exact_ctx_hvp
    ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
    --repeats 1 --dense-reference-block-width 4)
  set_tests_properties(dense_reduced_hessian_f2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "dense_reference_action_relative_error =")

  add_test(NAME projected_orbital_surrogate COMMAND test_projected_orbital_surrogate)
  set_tests_properties(projected_orbital_surrogate PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME positive_ritz_secants COMMAND test_positive_ritz_secants)
  set_tests_properties(positive_ritz_secants PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME normalized_orbital_curvature COMMAND test_normalized_orbital_curvature)
  set_tests_properties(normalized_orbital_curvature PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME spectral_trust_region COMMAND test_spectral_trust_region)
  set_tests_properties(spectral_trust_region PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME orthonormal_hvp_basis COMMAND test_orthonormal_hvp_basis)
  set_tests_properties(orthonormal_hvp_basis PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME reduced_hessian_reference COMMAND test_reduced_hessian_reference)
  set_tests_properties(reduced_hessian_reference PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME ri_ao_h1e_hvp COMMAND test_ri_ao_h1e_hvp)
  set_tests_properties(ri_ao_h1e_hvp PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(
    NAME ri_active_two_electron_response
    COMMAND test_ri_active_two_electron_response)
  set_tests_properties(ri_active_two_electron_response PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME sparse_orbital_quotient COMMAND test_sparse_orbital_quotient)
  set_tests_properties(sparse_orbital_quotient PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME support_preserving_gauge COMMAND test_support_preserving_gauge)
  set_tests_properties(support_preserving_gauge PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

  add_test(
    NAME nonredundant_lbfgs_f2
    COMMAND
      xmvb
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --optimizer-backend nonredundant_lbfgspp
      --eigensolver davidson
      --max-iterations 12
      --gradient-tolerance 1e-3
      --energy-tolerance 1e-7)
  set_tests_properties(nonredundant_lbfgs_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT
      "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "nonredundant_lbfgspp_dual_tolerance")

  add_test(
    NAME exact_ctx_hvp_f2_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --step 1e-4
      --probe full
      --max-rel-error 1e-7)
  set_tests_properties(
    exact_ctx_hvp_f2_finite_difference
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")

  add_test(
    NAME exact_ctx_hvp_f2_state_average_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_SA2.xmi
      --step 1e-4
      --probe full
      --max-rel-error 1e-7)
  set_tests_properties(
    exact_ctx_hvp_f2_state_average_finite_difference
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")

  add_test(
    NAME exact_ctx_hvp_f2_state_average_davidson_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_SA2.xmi
      --step 1e-4
      --probe full
      --eigensolver davidson
      --max-rel-error 1e-7)
  set_tests_properties(
    exact_ctx_hvp_f2_state_average_davidson_finite_difference
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")

  foreach(coupled_case IN ITEMS
      "coupled_projection_f2_dense|F2.xmi|dense"
      "coupled_projection_f2_state_average_dense|F2_SA2.xmi|dense"
      "coupled_projection_f2_state_average_davidson|F2_SA2.xmi|davidson")
    string(REPLACE "|" ";" coupled_fields "${coupled_case}")
    list(GET coupled_fields 0 coupled_test_name)
    list(GET coupled_fields 1 coupled_input_name)
    list(GET coupled_fields 2 coupled_eigensolver)
    add_test(
      NAME ${coupled_test_name}
      COMMAND
        check_exact_ctx_hvp
        ${CMAKE_SOURCE_DIR}/testdata/vbscf/${coupled_input_name}
        --eigensolver ${coupled_eigensolver}
        --coupled-projection-audit 1)
    set_tests_properties(${coupled_test_name} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
  endforeach()

  foreach(ri_case IN ITEMS
      "ri_ctx_hvp_f2_finite_difference|F2.xmi|dense"
      "ri_ctx_hvp_f2_oeo_finite_difference|F2_OEO.xmi|dense"
      "ri_ctx_hvp_f2_state_average_finite_difference|F2_SA2.xmi|dense"
      "ri_ctx_hvp_f2_state_average_davidson_finite_difference|F2_SA2.xmi|davidson")
    string(REPLACE "|" ";" ri_fields "${ri_case}")
    list(GET ri_fields 0 ri_test_name)
    list(GET ri_fields 1 ri_input_name)
    list(GET ri_fields 2 ri_eigensolver)
    add_test(
      NAME ${ri_test_name}
      COMMAND
        check_exact_ctx_hvp
        ${CMAKE_SOURCE_DIR}/testdata/vbscf/${ri_input_name}
        --standard-two-electron-mode ri
        --eigensolver ${ri_eigensolver}
        --step 1e-4
        --probe full
        --max-rel-error 1e-7)
    set_tests_properties(${ri_test_name} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
  endforeach()

  add_test(
    NAME exact_ctx_hvp_f2_oeo_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_OEO.xmi
      --step 1e-4
      --probe full
      --max-rel-error 1e-7)
  set_tests_properties(exact_ctx_hvp_f2_oeo_finite_difference PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

  add_test(
    NAME exact_ctx_hvp_f2_streamed_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --step 1e-4
      --probe full
      --stream-pair-products 1
      --max-rel-error 1e-7)
  set_tests_properties(exact_ctx_hvp_f2_streamed_finite_difference PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

  add_test(
    NAME exact_ctx_hvp_f2_oeo_streamed_finite_difference
    COMMAND
      check_exact_ctx_hvp
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_OEO.xmi
      --step 1e-4
      --probe full
      --stream-pair-products 1
      --max-rel-error 1e-7)
  set_tests_properties(exact_ctx_hvp_f2_oeo_streamed_finite_difference PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

unset(_xmvb_vbscf_unit_targets)
