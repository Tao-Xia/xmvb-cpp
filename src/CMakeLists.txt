file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" XMVB_CONDA_PREFIX)

if (NOT DEFINED XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND)
  set(
    XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND
    OFF
    CACHE BOOL
    "Enable the legacy Fortran L-BFGS backend inside the C++ VB optimizer."
    FORCE)
endif()

if (NOT DEFINED XMVB_CPP_BUILD_DEV_TARGETS)
  set(
    XMVB_CPP_BUILD_DEV_TARGETS
    OFF
    CACHE BOOL
    "Build C++ comparison, diagnostic, and test targets in addition to xmvb-cpp.exe."
    FORCE)
endif()

# The standalone path is now pure C++. Keep the old cache variables pinned off
# so existing build directories cannot silently pull runtime_c back in.
set(
  XMVB_CPP_BUILD_LEGACY_HF_RUNTIME
  OFF
  CACHE BOOL
  "Build the legacy standalone HF/XINT runtime used by legacy orbital guesses and HF diagnostics."
  FORCE)

set(
  XMVB_CPP_BUILD_RUNTIME_C_TOOLS
  OFF
  CACHE BOOL
  "Build legacy comparison/diagnostic tools that still depend on runtime_c."
  FORCE)

option(
  XMVB_CPP_ENABLE_EIGEN_BLAS_BACKEND
  "Route Eigen dense linear algebra through the configured BLAS/LAPACKE backends when supported."
  ON)

find_path(EIGEN3_INCLUDE_DIR Eigen/Core
  HINTS
    ${EIGEN3_ROOT_DIR}
    ${EIGEN3_ROOT_DIR}/include
    ${EIGEN3_ROOT_DIR}/include/eigen3
    ${XMVB_CONDA_PREFIX}
    ${XMVB_CONDA_PREFIX}/include
    ${XMVB_CONDA_PREFIX}/include/eigen3
    /usr/include/eigen3
    /usr/local/include/eigen3
  PATH_SUFFIXES
    eigen3
    include/eigen3
  REQUIRED)

include(vbscf/sources.cmake)

set(XMVB_CPP_RUNTIME_CORE_SOURCES
  runtime/cpp_block_guess_builder.cpp
  runtime/cpp_closed_shell_fock_builder.cpp
  runtime/cpp_initial_guess_builder.cpp
  runtime/generated_raw_structure_builder.cpp
  runtime/input_deck_model.cpp
  runtime/input_deck_metadata.cpp
  runtime/input_deck_primary_basis_builder.cpp
  runtime/input_deck_orbital_support_builder.cpp
  runtime/input_deck_raw_structures.cpp
  runtime/molden_file_writer.cpp
  runtime/cpp_restricted_hartree_fock.cpp
  runtime/vbscf_input_loader.cpp
  runtime/libcint_auxiliary_basis_builder.cpp
  runtime/libcint_direct_shell_evaluator.cpp
  runtime/libcint_materialized_integral_provider.cpp
  runtime/libcint_ri_integral_provider.cpp
  runtime/materialized_ao_integral_input_builder.cpp
)

set(XMVB_CPP_RUNTIME_SOURCES ${XMVB_CPP_RUNTIME_CORE_SOURCES})

add_library(xmvb_cpp_openmp INTERFACE)
target_link_libraries(xmvb_cpp_openmp
  INTERFACE
    OpenMP::OpenMP_C
    OpenMP::OpenMP_CXX)

add_library(xmvb_cpp_vb STATIC ${XMVB_CPP_CORE_SOURCES})
set_target_properties(xmvb_cpp_vb PROPERTIES OUTPUT_NAME xmvb-vbscf)
target_include_directories(xmvb_cpp_vb
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/LBFGSpp/include
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${CINT_INCLUDE})
if (XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC XMVB_CPP_ENABLE_LEGACY_FORTRAN_BACKEND=1)
endif()
if (XMVB_CPP_ENABLE_ONNX_RUNTIME)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC XMVB_CPP_ENABLE_ONNX_RUNTIME=1)
endif()
if (XMVB_CPP_ENABLE_EIGEN_BLAS_BACKEND)
  target_compile_definitions(
    xmvb_cpp_vb
    PUBLIC
      EIGEN_USE_BLAS=1
      EIGEN_USE_LAPACKE=1)
endif()
target_link_libraries(xmvb_cpp_vb PUBLIC ${LAPACKE_LIB} xmvb_cpp_openmp)
if (LAPACK_TYPE STREQUAL "OpenBLAS")
  target_link_libraries(xmvb_cpp_vb PUBLIC ${LAPACK_LIB})
elseif (LAPACK_TYPE STREQUAL "refBLAS")
  target_link_libraries(xmvb_cpp_vb PUBLIC lapacke cblas lapack)
else()
  target_link_libraries(xmvb_cpp_vb PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

set(
  XMVB_CPP_RUNTIME_EXECUTABLE_PATH
  "${CMAKE_BINARY_DIR}/src/driver/xmvb.exe"
  CACHE FILEPATH
  "Executable path used by the standalone runtime to locate basis/data assets.")

add_library(xmvb_cpp_runtime STATIC ${XMVB_CPP_RUNTIME_SOURCES})
target_compile_definitions(xmvb_cpp_runtime
  PRIVATE
    PATH_MAX=4096
    XMVB_CPP_LEGACY_EXECUTABLE_PATH=\"${XMVB_CPP_RUNTIME_EXECUTABLE_PATH}\")
target_include_directories(xmvb_cpp_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/src/vbutil
    ${CMAKE_BINARY_DIR}/src/vbscf
    ${CMAKE_BINARY_DIR}/src/vbci
    ${CMAKE_BINARY_DIR}/src/vbpt2
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${CINT_INCLUDE})
target_link_libraries(xmvb_cpp_runtime PUBLIC xmvb_cpp_vb xmvb_cpp_openmp)

function(xmvb_link_standalone_runtime target_name)
  target_link_libraries(${target_name}
    PRIVATE
      ${CINT_LIB}
      xmvb_cpp_openmp
      xmvb_cpp_vb
      m)
  if (LAPACK_TYPE STREQUAL "OpenBLAS")
    target_link_libraries(${target_name} PRIVATE ${LAPACKE_LIB} ${LAPACK_LIB})
  elseif (LAPACK_TYPE STREQUAL "refBLAS")
    target_link_libraries(${target_name} PRIVATE lapacke cblas lapack)
  else()
    target_link_libraries(${target_name} PRIVATE ${LAPACKE_LIB} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endfunction()

function(xmvb_mark_targets_exclude_from_all)
  foreach(target_name IN LISTS ARGN)
    if (TARGET ${target_name})
      set_target_properties(
        ${target_name}
        PROPERTIES
          EXCLUDE_FROM_ALL TRUE
          EXCLUDE_FROM_DEFAULT_BUILD TRUE)
    endif()
  endforeach()
endfunction()

add_library(xmvb_cpp_deepvbh STATIC ${XMVB_DEEPVBH_SOURCES})
set_target_properties(xmvb_cpp_deepvbh PROPERTIES OUTPUT_NAME xmvb-deepvbh)
target_include_directories(xmvb_cpp_deepvbh
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/src/vbutil
    ${CMAKE_BINARY_DIR}/src/vbscf
    ${CMAKE_BINARY_DIR}/src/vbci
    ${CMAKE_BINARY_DIR}/src/vbpt2
    ${EIGEN3_INCLUDE_DIR})
target_link_libraries(xmvb_cpp_deepvbh PUBLIC xmvb_cpp_runtime)
if (XMVB_CPP_ENABLE_ONNX_RUNTIME)
  target_compile_definitions(
    xmvb_cpp_deepvbh
    PUBLIC XMVB_CPP_ENABLE_ONNX_RUNTIME=1)
  target_include_directories(
    xmvb_cpp_deepvbh
    PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(
    xmvb_cpp_deepvbh
    PUBLIC ${ONNXRUNTIME_LIB})
endif()

add_executable(run_cpp_vbscf
  tools/run_cpp_vbscf.cpp)
set_target_properties(run_cpp_vbscf PROPERTIES OUTPUT_NAME xmvb-cpp.exe)
target_link_libraries(run_cpp_vbscf
  PRIVATE
    xmvb_cpp_deepvbh
    xmvb_cpp_vb
    xmvb_cpp_runtime)
xmvb_link_standalone_runtime(run_cpp_vbscf)

file(
  GLOB_RECURSE XMVB_CPP_RUNTIME_BASIS_INPUTS
  "${CMAKE_SOURCE_DIR}/basis/*")
set(XMVB_CPP_RUNTIME_ASSET_STAMP
  "${CMAKE_BINARY_DIR}/CMakeFiles/xmvb_cpp_runtime_assets.stamp")
add_custom_command(
  OUTPUT ${XMVB_CPP_RUNTIME_ASSET_STAMP}
  COMMAND ${CMAKE_COMMAND}
          -DXMVB_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DXMVB_BINARY_DIR=${CMAKE_BINARY_DIR}
          -P ${CMAKE_SOURCE_DIR}/cmake/stage_runtime_assets.cmake
  COMMAND ${CMAKE_COMMAND} -E touch ${XMVB_CPP_RUNTIME_ASSET_STAMP}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/cmake/stage_runtime_assets.cmake
    ${XMVB_CPP_RUNTIME_BASIS_INPUTS}
  # `basis/` is copied into the build tree, so basis-file edits should restage
  # runtime assets. `data/` is linked instead of copied and contains volatile
  # training corpora, logs, and scratch outputs. Do not expand that directory
  # into the Ninja dependency graph, or every dataset churn will force a CMake
  # regenerate and invalidate the whole Ninja deps database.
  COMMENT "Staging standalone xmvb-cpp runtime assets into the build tree")
add_custom_target(
  xmvb_cpp_runtime_assets
  DEPENDS ${XMVB_CPP_RUNTIME_ASSET_STAMP})
add_dependencies(run_cpp_vbscf xmvb_cpp_runtime_assets)

if (XMVB_CPP_BUILD_DEV_TARGETS)
  include(tools/targets.cmake)
endif()
