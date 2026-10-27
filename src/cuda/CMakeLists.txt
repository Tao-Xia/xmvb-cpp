add_library(xmvb_cuda
  linalg/block_map.cu
  linalg/triangular.cu
  linalg/davidson.cu
  linalg/cholesky.cu
  linalg/spectrum.cu
  linalg/svd.cu
  ci/raw.cu
  ci/similarity.cu
  ci/sigma.cu
  ci/tiles.cu
  ci/moments.cu
  ci/spin_self.cu
  integrals/exact.cu
  integrals/unique.cu
  integrals/packed.cu
  integrals/packed_response.cu
  integrals/exchange.cu
  integrals/core.cu
  integrals/one_electron.cu
  optimization/ciah.cu
  optimization/coarse.cu
  vb/transform.cu
  vb/product.cu
  vb/columns.cu
  vb/metric.cu
  vb/orbitals.cu
  vb/chart.cu
  vb/diagonal.cu
  vb/tangent.cu
  vb/transport.cu
  vb/operator_first.cu
  vb/point.cu
  vb/scf.cu
  vb/ci.cu
  vb/action.cu)
target_sources(xmvb_cuda PRIVATE vb/full_action.cu vb/panel.cpp)
target_include_directories(xmvb_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(xmvb_cuda PUBLIC
  xmvb_linalg xmvb_mcscf xmvb_vbscf
  CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)
target_link_libraries(xmvb_cuda PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(xmvb_cuda PROPERTIES CUDA_STANDARD 20)

add_library(xmvb_gpu_cli ${CMAKE_SOURCE_DIR}/src/cli/cuvbscf.cpp)
target_link_libraries(xmvb_gpu_cli PUBLIC xmvb_cuda xmvb_input xmvb_output)
target_link_libraries(xmvb PRIVATE xmvb_gpu_cli)
target_compile_definitions(xmvb PRIVATE XMVB_ENABLE_CUDA=1)

add_executable(cuvbscf ${CMAKE_SOURCE_DIR}/tools/cuvbscf/main.cpp)
target_link_libraries(cuvbscf PRIVATE xmvb_gpu_cli)
set_target_properties(cuvbscf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src)

add_executable(cuvb_input_audit ${CMAKE_SOURCE_DIR}/tests/cuda/input_audit.cpp)
target_link_libraries(cuvb_input_audit PRIVATE xmvb_input)
set_target_properties(cuvb_input_audit PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src)
