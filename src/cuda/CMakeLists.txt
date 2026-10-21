add_library(xmvb_cuda
  linalg/davidson.cu
  linalg/cholesky.cu
  ci/sigma.cu
  vb/transform.cu
  vb/metric.cu
  vb/ci.cu
  vb/action.cu)
target_include_directories(xmvb_cuda PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(xmvb_cuda PUBLIC
  xmvb_linalg xmvb_mcscf xmvb_vbscf
  CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)
target_link_libraries(xmvb_cuda PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(xmvb_cuda PROPERTIES CUDA_STANDARD 20)
