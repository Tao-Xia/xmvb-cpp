find_path(LIBRI_INCLUDE_DIR RI/ri/LRI.h HINTS ${LIBRI_ROOT} PATH_SUFFIXES include REQUIRED)
find_path(LIBCOMM_INCLUDE_DIR Comm/Comm_Tools.h HINTS ${LIBCOMM_ROOT} PATH_SUFFIXES include REQUIRED)
find_path(CEREAL_INCLUDE_DIR cereal/cereal.hpp HINTS ${CEREAL_ROOT} PATH_SUFFIXES include REQUIRED)

add_library(xmvb_local_ri STATIC exchange.cpp fit.cpp)
target_compile_features(xmvb_local_ri PUBLIC cxx_std_23)
target_include_directories(xmvb_local_ri PUBLIC ${PROJECT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
target_include_directories(xmvb_local_ri SYSTEM PRIVATE
  ${LIBRI_INCLUDE_DIR} ${LIBCOMM_INCLUDE_DIR} ${CEREAL_INCLUDE_DIR})
target_link_libraries(xmvb_local_ri PRIVATE MPI::MPI_CXX xmvb_openmp ${LAPACK_LIB})

if (BUILD_TESTING AND XMVB_BUILD_DEV_TARGETS)
  add_executable(test_local_ri_exchange ${PROJECT_SOURCE_DIR}/tests/integrals/local_ri_exchange.cpp)
  target_link_libraries(test_local_ri_exchange PRIVATE xmvb_local_ri MPI::MPI_CXX)
  add_test(NAME local_ri_exchange COMMAND test_local_ri_exchange)
  set_tests_properties(local_ri_exchange PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
endif()
