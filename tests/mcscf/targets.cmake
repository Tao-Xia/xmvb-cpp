# Common solver tests must link without the VBSCF or CASSCF model libraries.
set(_xmvb_mcscf_tests
  block_inverse_bfgs ciah neo neo_globalization
  positive_ritz_secants shifted_metric_preconditioner state_weights)
foreach(name IN LISTS _xmvb_mcscf_tests)
  add_executable(test_${name} ${CMAKE_CURRENT_LIST_DIR}/unit/test_${name}.cpp)
  target_link_libraries(test_${name} PRIVATE xmvb_mcscf)
  if (BUILD_TESTING)
    add_test(NAME ${name} COMMAND test_${name})
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  endif()
endforeach()
unset(_xmvb_mcscf_tests)

add_executable(test_raw_action ${CMAKE_CURRENT_LIST_DIR}/test_raw_action.cpp)
target_link_libraries(test_raw_action PRIVATE xmvb_mcscf)
if (BUILD_TESTING)
  add_test(NAME raw_action COMMAND test_raw_action)
  set_tests_properties(raw_action PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

add_executable(audit_channel_support ${CMAKE_CURRENT_LIST_DIR}/audit_channel_support.cpp)
target_link_libraries(audit_channel_support PRIVATE xmvb_mcscf)
if (BUILD_TESTING)
  add_test(NAME channel_support COMMAND audit_channel_support --oracle)
  set_tests_properties(channel_support PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

add_executable(test_spin_split ${CMAKE_CURRENT_LIST_DIR}/test_spin_split.cpp)
target_link_libraries(test_spin_split PRIVATE xmvb_mcscf)
if (BUILD_TESTING)
  add_test(NAME spin_split COMMAND test_spin_split)
  set_tests_properties(spin_split PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
endif()

add_executable(test_index_bounds ${CMAKE_CURRENT_LIST_DIR}/test_index_bounds.cpp)
target_link_libraries(test_index_bounds PRIVATE xmvb_mcscf)
if (BUILD_TESTING)
  add_test(NAME index_bounds COMMAND test_index_bounds)
endif()
