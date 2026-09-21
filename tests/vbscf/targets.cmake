# VBSCF unit and numerical-regression targets.
#
# This file is included from src/CMakeLists.txt while developer targets are
# enabled.  Keeping the target ownership beside the tests prevents the main
# production build file from accumulating test-specific wiring.

set(_xmvb_vbscf_unit_targets
  test_active_two_electron_sparse
  test_block_inverse_bfgs
  test_cofactor_differential
  test_coupled_structure
  test_curvature_decomposition
  test_davidson
  test_eigen_response
  test_localized_representative_selector
  test_minres
  test_neo
  test_neo_response_solver
  test_neo_globalization
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
  add_test(NAME block_inverse_bfgs COMMAND test_block_inverse_bfgs)
  set_tests_properties(block_inverse_bfgs PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME orbital_block_partition COMMAND test_orbital_block_partition)
  add_test(NAME curvature_decomposition COMMAND test_curvature_decomposition)
  add_test(NAME oeo_normalization_pullback COMMAND test_oeo_normalization_pullback)
  add_test(NAME opposite_spin_pair_graph COMMAND test_opposite_spin_pair_graph)
  add_test(NAME orthogonal_direct_ci COMMAND test_orthogonal_direct_ci)
  add_test(NAME cofactor_differential COMMAND test_cofactor_differential)
  add_test(NAME coupled_structure COMMAND test_coupled_structure)
  set_tests_properties(coupled_structure PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME davidson COMMAND test_davidson)
  add_test(NAME eigen_response COMMAND test_eigen_response)
  add_test(NAME minres COMMAND test_minres)
  set_tests_properties(minres PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME neo COMMAND test_neo)
  set_tests_properties(neo PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME neo_response_solver COMMAND test_neo_response_solver)
  set_tests_properties(neo_response_solver PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME neo_globalization COMMAND test_neo_globalization)
  set_tests_properties(neo_globalization PROPERTIES
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
    --trust-radius 0.1
    --finite-difference-step 1e-5 --eigensolver davidson)
  set_tests_properties(audit_newton_step_f2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "directional_hvp_fd_relative =")

  foreach(case IN ITEMS 241 7975)
    if(case STREQUAL "241")
      set(input_name 241_VBSCF)
      set(trust_radius 0.397717925496941)
    else()
      set(input_name 7975_vb)
      set(trust_radius 0.00498924840619093)
    endif()
    add_test(NAME response_model_${case} COMMAND
      ${CMAKE_COMMAND}
      -DAUDIT_EXECUTABLE=$<TARGET_FILE:audit_newton_step>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/${input_name}.xmi
      -DXMVB_BASIS=${CMAKE_SOURCE_DIR}/basis
      -DORBITAL_FIXTURE=${CMAKE_SOURCE_DIR}/tests/vbscf/fixtures/response/${case}_orbitals.txt
      -DTRUST_RADIUS=${trust_radius}
      -DRUN_DIR=${CMAKE_BINARY_DIR}/tests/response_model_${case}
      -P ${CMAKE_SOURCE_DIR}/tests/vbscf/check_response_model.cmake)
  endforeach()
  set_tests_properties(response_model_241 response_model_7975 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

  add_test(NAME tnhvp_model_observations_f2 COMMAND
    ${CMAKE_COMMAND}
    -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
    -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
    -DXMVB_BASIS=${CMAKE_SOURCE_DIR}/basis
    -DRUN_DIR=${CMAKE_BINARY_DIR}/tests/tnhvp_model_observations
    -P ${CMAKE_SOURCE_DIR}/tests/vbscf/check_model_observations.cmake)
  set_tests_properties(tnhvp_model_observations_f2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

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
    NAME lbfgs_f2
    COMMAND
      xmvb
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --optimizer-backend lbfgs
      --eigensolver davidson
      --max-iterations 40
      --gradient-tolerance 1e-3
      --energy-tolerance 1e-7)
  set_tests_properties(lbfgs_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT
      "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "xmvb_lbfgs_dual_tolerance")

  add_test(
    NAME block_lbfgs_f2
    COMMAND
      xmvb
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --optimizer-backend block_lbfgs
      --eigensolver davidson
      --max-iterations 40
      --gradient-tolerance 1e-3
      --energy-tolerance 1e-7)
  set_tests_properties(block_lbfgs_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT
      "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "block_lbfgs_dual_tolerance")

  add_test(
    NAME neo_f2
    COMMAND
      xmvb
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --optimizer-backend neo
      --eigensolver davidson
      --max-iterations 30
      --gradient-tolerance 1e-3
      --energy-tolerance 1e-7)
  set_tests_properties(neo_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT
      "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "neo_dual_tolerance")

  add_test(
    NAME neo_pyscf_reference_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_OEO_PYSCF_LOW.xmi
      -DTRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/neo_pyscf_reference_f2.tsv
      -P ${CMAKE_SOURCE_DIR}/tests/vbscf/check_neo_pyscf_reference.cmake)
  set_tests_properties(neo_pyscf_reference_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

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
