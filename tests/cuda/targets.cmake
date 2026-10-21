# CUDA tests are launched inside a GPU allocation by tests/cuda/c12.sbatch.
add_executable(test_cuda_structure_action
  ${CMAKE_SOURCE_DIR}/tests/cuda/test_structure_action.cpp)
target_link_libraries(test_cuda_structure_action PRIVATE xmvb_cuda xmvb_vbscf)

add_executable(c12_cuda_benchmark
  ${CMAKE_SOURCE_DIR}/tests/cuda/c12_benchmark.cpp)
target_include_directories(c12_cuda_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/tests/cuda)
target_link_libraries(c12_cuda_benchmark PRIVATE xmvb_cuda xmvb_input xmvb_vbscf)
