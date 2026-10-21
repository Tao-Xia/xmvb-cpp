add_library(xmvb_cuda
  linalg/davidson.cu
  linalg/cholesky.cu
  ci/sigma.cu
  integrals/exact.cu
  integrals/exchange.cu
  integrals/one_electron.cu
  optimization/ciah.cu
  optimization/coarse.cu
  vb/transform.cu
  vb/metric.cu
  vb/orbitals.cu
  vb/chart.cu
  vb/diagonal.cu
  vb/tangent.cu
  vb/transport.cu
  vb/point.cu
  vb/scf.cu
  vb/ci.cu
  vb/action.cu)
target_include_directories(xmvb_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(xmvb_cuda PUBLIC
  xmvb_linalg xmvb_mcscf xmvb_vbscf
  CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)
target_link_libraries(xmvb_cuda PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(xmvb_cuda PROPERTIES CUDA_STANDARD 20)

add_executable(cuvbscf ${CMAKE_SOURCE_DIR}/tools/cuvbscf/main.cpp)
target_link_libraries(cuvbscf PRIVATE xmvb_cuda xmvb_input xmvb_output)
set_target_properties(cuvbscf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src)
