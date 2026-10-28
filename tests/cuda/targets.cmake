# CUDA tests are launched inside a GPU allocation by tests/cuda/production.sbatch.
add_executable(test_cuda_structure_action
  ${CMAKE_SOURCE_DIR}/tests/cuda/test_structure_action.cpp)
target_link_libraries(test_cuda_structure_action PRIVATE xmvb_cuda xmvb_vbscf)

add_executable(test_cuda_production_ci
  ${CMAKE_SOURCE_DIR}/tests/cuda/test_production_ci.cpp)
target_link_libraries(test_cuda_production_ci PRIVATE xmvb_cuda xmvb_vbscf)
set(cuda_test_targets test_cuda_structure_action test_cuda_production_ci)

foreach(name IN ITEMS exterior operator_first exact orbitals ciah point tangent chart device_davidson ci_refresh transport coarse diagonal macro ci_ah_inverse selected_metric streamed_moments product_columns metric_residency)
  add_executable(test_cuda_${name} ${CMAKE_SOURCE_DIR}/tests/cuda/test_${name}.cpp)
  target_link_libraries(test_cuda_${name} PRIVATE xmvb_cuda xmvb_vbscf)
  list(APPEND cuda_test_targets test_cuda_${name})
endforeach()
add_custom_target(cuda_tests DEPENDS ${cuda_test_targets})

add_executable(c12_cuda_benchmark
  ${CMAKE_SOURCE_DIR}/tests/cuda/c12_benchmark.cpp)
target_include_directories(c12_cuda_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/tests/cuda)
target_link_libraries(c12_cuda_benchmark PRIVATE xmvb_cuda xmvb_input xmvb_vbscf)

add_executable(test_cuda_index_bounds ${CMAKE_SOURCE_DIR}/tests/cuda/test_index_bounds.cu)
target_link_libraries(test_cuda_index_bounds PRIVATE xmvb_cuda)
set_target_properties(test_cuda_index_bounds PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
add_dependencies(cuda_tests test_cuda_index_bounds)
