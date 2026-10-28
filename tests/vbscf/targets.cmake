# VBSCF unit and numerical-regression targets.
#
# This file is included from src/CMakeLists.txt while developer targets are
# enabled.  Keeping the target ownership beside the tests prevents the main
# production build file from accumulating test-specific wiring.

set(_xmvb_vbscf_unit_targets
  test_exterior_transform
  test_spinor_strings
  test_spinor_projection
  test_active_two_electron_sparse
  test_integral_transform
  test_ri_integral_transform
  test_orthogonal_ri_transform
  test_fused_integral_frame
  test_two_electron_jet
  test_orbital_pair_direction
  test_raw_integral_frame
  test_projected_orbital_pullback
  test_ri_projected_response
  test_accepted_pair_tile
  test_hessian_diagonal
  test_keyframe
  test_cofactor_differential
  test_coupled_structure
  test_ci_transport
  test_representation_overlap
  test_selected_metric
  test_overlap_pullback
  test_selected_ci_transport
  test_orbital_step_geometry
  test_curvature_decomposition
  test_localized_representative_selector
  test_normalized_orbital_curvature
  test_oeo_normalization_pullback
  test_opposite_spin_pair_graph
  test_orthogonal_direct_ci
  test_projected_orbital_surrogate
  test_reduced_hessian_reference
  test_ri_ao_h1e_hvp
  test_ri_pair_contractions
  test_ri_active_two_electron_response
  test_sparse_orbital_quotient
  test_support_preserving_gauge
  test_spectral_trust_region
  test_structure_spin
  test_structure_tiles
  test_orbital_block_partition)

foreach(target_name IN LISTS _xmvb_vbscf_unit_targets)
  add_executable(
    ${target_name}
    ${CMAKE_SOURCE_DIR}/tests/vbscf/unit/${target_name}.cpp)
  target_link_libraries(${target_name} PRIVATE xmvb_vbscf)
endforeach()

# Same-orbital molecular reference: direct-CI versus nonorthogonal cofactors.
add_executable(test_action_parity EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/action_parity.cpp)
target_link_libraries(test_action_parity PRIVATE xmvb_input xmvb_vbscf)

# Independent symbolic schedule census; does not alter a numerical backend.
add_executable(test_metric_order EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/metric_order.cpp)
target_link_libraries(test_metric_order PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_metric_point EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/metric_point.cpp)
target_link_libraries(test_metric_point PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_initial_ci EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/initial_ci.cpp)
target_link_libraries(test_initial_ci PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_contraction_order EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/contraction_order.cpp)
target_link_libraries(test_contraction_order PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_similarity_order EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/similarity_order.cpp)
target_link_libraries(test_similarity_order PRIVATE xmvb_vbscf)

add_executable(test_conditional_sampling EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/conditional_sampling.cpp)
target_link_libraries(test_conditional_sampling PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_sampling EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/sampling.cpp)
target_link_libraries(test_sampling PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_operator_point EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/operator_point.cpp)
target_link_libraries(test_operator_point PRIVATE xmvb_input xmvb_vbscf)

add_executable(test_selected_metric_point EXCLUDE_FROM_ALL
  ${CMAKE_SOURCE_DIR}/tests/vbscf/diagnostics/selected_metric_point.cpp)
target_link_libraries(test_selected_metric_point PRIVATE xmvb_input xmvb_vbscf)

if (BUILD_TESTING)
  add_test(NAME exterior_transform COMMAND test_exterior_transform)
  set_tests_properties(exterior_transform PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1")
  add_test(NAME ci_transport COMMAND test_ci_transport)
  add_test(NAME representation_overlap COMMAND test_representation_overlap)
  add_test(NAME selected_metric COMMAND test_selected_metric)
  add_test(NAME overlap_pullback COMMAND test_overlap_pullback)
  add_test(NAME selected_ci_transport COMMAND test_selected_ci_transport)
  add_test(NAME orbital_step_geometry COMMAND test_orbital_step_geometry)
  add_test(NAME spinor_strings COMMAND test_spinor_strings)
  add_test(NAME spinor_projection COMMAND test_spinor_projection)
  set_tests_properties(spinor_strings spinor_projection PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME active_two_electron_sparse COMMAND test_active_two_electron_sparse)
  add_test(NAME integral_transform COMMAND test_integral_transform)
  add_test(NAME ri_integral_transform COMMAND test_ri_integral_transform)
  add_test(NAME orthogonal_ri_transform COMMAND test_orthogonal_ri_transform)
  add_test(NAME fused_integral_frame COMMAND test_fused_integral_frame)
  foreach(_frame_backend IN ITEMS exact ri)
    add_test(NAME integral_frame_pair_${_frame_backend}
      COMMAND check_exact_ctx_hvp
        ${CMAKE_SOURCE_DIR}/tests/vbscf/fixtures/response/h4_pair.xmi
        --standard-two-electron-mode ${_frame_backend}
        --eigensolver davidson --step 1e-4 --probe full --max-rel-error 1e-7)
    set_tests_properties(integral_frame_pair_${_frame_backend} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
  endforeach()
  add_test(NAME two_electron_jet COMMAND test_two_electron_jet)
  add_test(NAME orbital_pair_direction COMMAND test_orbital_pair_direction)
  add_test(NAME raw_integral_frame COMMAND test_raw_integral_frame)
  set_tests_properties(fused_integral_frame two_electron_jet orbital_pair_direction raw_integral_frame PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME projected_orbital_pullback COMMAND test_projected_orbital_pullback)
  add_test(NAME ri_projected_response COMMAND test_ri_projected_response)
  add_test(NAME accepted_pair_tile COMMAND test_accepted_pair_tile)
  add_test(NAME hessian_diagonal COMMAND test_hessian_diagonal)
  add_test(NAME keyframe COMMAND test_keyframe)
  add_test(NAME orbital_block_partition COMMAND test_orbital_block_partition)
  add_test(NAME curvature_decomposition COMMAND test_curvature_decomposition)
  add_test(NAME oeo_normalization_pullback COMMAND test_oeo_normalization_pullback)
  add_test(NAME opposite_spin_pair_graph COMMAND test_opposite_spin_pair_graph)
  add_test(NAME orthogonal_direct_ci COMMAND test_orthogonal_direct_ci)
  add_test(NAME structure_tiles COMMAND test_structure_tiles)
  add_test(NAME cofactor_differential COMMAND test_cofactor_differential)
  add_test(NAME coupled_structure COMMAND test_coupled_structure)
  set_tests_properties(coupled_structure PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(
    NAME localized_representative_selector
    COMMAND test_localized_representative_selector)
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
  add_test(NAME normalized_orbital_curvature COMMAND test_normalized_orbital_curvature)
  set_tests_properties(normalized_orbital_curvature PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME spectral_trust_region COMMAND test_spectral_trust_region)
  add_test(NAME structure_spin COMMAND test_structure_spin)
  set_tests_properties(spectral_trust_region PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME reduced_hessian_reference COMMAND test_reduced_hessian_reference)
  set_tests_properties(reduced_hessian_reference PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME ri_ao_h1e_hvp COMMAND test_ri_ao_h1e_hvp)
  set_tests_properties(ri_ao_h1e_hvp PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(NAME ri_pair_contractions COMMAND test_ri_pair_contractions)
  set_tests_properties(ri_pair_contractions PROPERTIES
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
    NAME plbfgs_f2
    COMMAND
      xmvb
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --optimizer-backend plbfgs
      --eigensolver davidson
      --max-iterations 40
      --gradient-tolerance 1e-3
      --energy-tolerance 1e-7)
  set_tests_properties(plbfgs_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT
      "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "plbfgs_projected_gradient_tolerance")

  add_test(
    NAME ciah_f2
    COMMAND
      xmvb
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      --optimizer-backend ciah
      --eigensolver davidson
      --max-iterations 30
      --gradient-tolerance 1e-3
      --energy-tolerance 1e-7)
  set_tests_properties(ciah_f2 PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    ENVIRONMENT
      "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
    PASS_REGULAR_EXPRESSION "ciah_projected_gradient_tolerance")

  foreach(_xmvb_orbital_type IN ITEMS hao oeo)
    if (_xmvb_orbital_type STREQUAL "hao")
      set(_xmvb_hessian_diagonal_input
        ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi)
    else()
      set(_xmvb_hessian_diagonal_input
        ${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_OEO_PYSCF_LOW.xmi)
    endif()
    foreach(_xmvb_integral_mode IN ITEMS exact ri)
      if (_xmvb_integral_mode STREQUAL "ri")
        set(_xmvb_hessian_diagonal_test
          ciah_hessian_diagonal_ri_f2_${_xmvb_orbital_type})
        set(_xmvb_hessian_diagonal_mode_args
          --standard-two-electron-mode ri)
      else()
        set(_xmvb_hessian_diagonal_test
          ciah_hessian_diagonal_f2_${_xmvb_orbital_type})
        set(_xmvb_hessian_diagonal_mode_args)
      endif()
      add_test(
        NAME ${_xmvb_hessian_diagonal_test}
        COMMAND
          xmvb
          ${_xmvb_hessian_diagonal_input}
          --optimizer-backend ciah
          --orbital-preconditioner hessian-diagonal
          ${_xmvb_hessian_diagonal_mode_args}
          --eigensolver davidson
          --max-iterations 10
          --gradient-tolerance 1e-3
          --energy-tolerance 1e-7)
      set_tests_properties(
        ${_xmvb_hessian_diagonal_test} PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        ENVIRONMENT
          "OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1"
        PASS_REGULAR_EXPRESSION "ciah_projected_gradient_tolerance")
    endforeach()
  endforeach()
  unset(_xmvb_hessian_diagonal_input)
  unset(_xmvb_hessian_diagonal_mode_args)
  unset(_xmvb_hessian_diagonal_test)
  unset(_xmvb_integral_mode)
  unset(_xmvb_orbital_type)

  add_test(
    NAME ciah_pyscf_reference_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_OEO_PYSCF_LOW.xmi
      -DTRACE_FILE=${CMAKE_CURRENT_BINARY_DIR}/ciah_pyscf_reference_f2.tsv
      -P ${CMAKE_SOURCE_DIR}/tests/vbscf/check_ciah_reference.cmake)
  set_tests_properties(ciah_pyscf_reference_f2 PROPERTIES
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

# Fixed-reference gradient/HVP/keyframe audit at prescribed molecular orbitals.
add_executable(check_ciah_derivatives EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/tests/vbscf/diagnostics/ciah_derivatives.cpp)
target_link_libraries(check_ciah_derivatives PRIVATE xmvb_input xmvb_vbscf)

add_executable(check_ciah_conditioning EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/vbscf/diagnostics/ciah_conditioning.cpp)
target_link_libraries(check_ciah_conditioning PRIVATE xmvb_input xmvb_vbscf)

add_executable(check_structure_accuracy EXCLUDE_FROM_ALL
  ${PROJECT_SOURCE_DIR}/tests/vbscf/diagnostics/structure_accuracy.cpp)
target_link_libraries(check_structure_accuracy PRIVATE xmvb_input xmvb_vbscf)
