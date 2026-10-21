# CUDA tests are launched inside a GPU allocation by tests/cuda/production.sbatch.
add_executable(test_cuda_structure_action
  ${CMAKE_SOURCE_DIR}/tests/cuda/test_structure_action.cpp)
target_link_libraries(test_cuda_structure_action PRIVATE xmvb_cuda xmvb_vbscf)

add_executable(test_cuda_production_ci
  ${CMAKE_SOURCE_DIR}/tests/cuda/test_production_ci.cpp)
target_link_libraries(test_cuda_production_ci PRIVATE xmvb_cuda xmvb_vbscf)

foreach(name IN ITEMS exact orbitals ciah point tangent chart device_davidson ci_refresh transport coarse diagonal macro ci_ah_inverse selected_metric)
  add_executable(test_cuda_${name} ${CMAKE_SOURCE_DIR}/tests/cuda/test_${name}.cpp)
  target_link_libraries(test_cuda_${name} PRIVATE xmvb_cuda xmvb_vbscf)
endforeach()

add_executable(c12_cuda_benchmark
  ${CMAKE_SOURCE_DIR}/tests/cuda/c12_benchmark.cpp)
target_include_directories(c12_cuda_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/tests/cuda)
target_link_libraries(c12_cuda_benchmark PRIVATE xmvb_cuda xmvb_input xmvb_vbscf)
