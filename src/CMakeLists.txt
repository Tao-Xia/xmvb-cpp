file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" XMVB_CONDA_PREFIX)

if (NOT DEFINED XMVB_BUILD_DEV_TARGETS)
  set(
    XMVB_BUILD_DEV_TARGETS
    OFF
    CACHE BOOL
    "Build comparison, diagnostic, and test targets in addition to xmvb-cpp.exe."
    FORCE)
endif()

option(
  XMVB_ENABLE_EIGEN_BLAS_BACKEND
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

set(XMVB_RUNTIME_SOURCES
  guess/block.cpp
  guess/closed_shell_fock.cpp
  guess/initial.cpp
  input/loading/generated_structures.cpp
  input/deck/model.cpp
  input/deck/metadata.cpp
  input/deck/primary_basis.cpp
  input/deck/orbital_support.cpp
  input/deck/raw_structures.cpp
  output/molden/writer.cpp
  guess/restricted_hf.cpp
  input/loading/loader.cpp
  libcint/auxiliary_basis.cpp
  libcint/direct_shell.cpp
  libcint/materialized_provider.cpp
  libcint/ri_provider.cpp
  libcint/materialized_input.cpp
  output/trace/accepted_iteration.cpp
)

# Keep runtime ownership as strict as the canonical VBSCF manifest. A new
# translation unit must be assigned here before it can enter the build.
file(
  GLOB_RECURSE _xmvb_discovered_runtime_sources
  CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/guess/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/input/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/libcint/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/*.cpp")
set(_xmvb_declared_runtime_sources ${XMVB_RUNTIME_SOURCES})
list(SORT _xmvb_declared_runtime_sources)
list(SORT _xmvb_discovered_runtime_sources)
if (NOT "${_xmvb_declared_runtime_sources}" STREQUAL
    "${_xmvb_discovered_runtime_sources}")
  message(FATAL_ERROR
    "src/CMakeLists.txt does not own every runtime .cpp exactly once.\n"
    "Declared: ${_xmvb_declared_runtime_sources}\n"
    "Discovered: ${_xmvb_discovered_runtime_sources}")
endif()
unset(_xmvb_declared_runtime_sources)
unset(_xmvb_discovered_runtime_sources)

add_library(xmvb_openmp INTERFACE)
target_link_libraries(xmvb_openmp
  INTERFACE
    OpenMP::OpenMP_C
    OpenMP::OpenMP_CXX)

add_library(xmvb_vbscf STATIC ${XMVB_VBSCF_SOURCES})
set_target_properties(xmvb_vbscf PROPERTIES OUTPUT_NAME xmvb-vbscf)
target_include_directories(xmvb_vbscf
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/LBFGSpp/include
    ${EIGEN3_INCLUDE_DIR}
    ${LAPACK_INCLUDE}
    ${LAPACKE_INCLUDE}
    ${CINT_INCLUDE})
if (XMVB_ENABLE_EIGEN_BLAS_BACKEND)
  target_compile_definitions(
    xmvb_vbscf
    PUBLIC
      EIGEN_USE_BLAS=1
      EIGEN_USE_LAPACKE=1)
endif()
target_link_libraries(xmvb_vbscf PUBLIC ${LAPACKE_LIB} xmvb_openmp)
if (LAPACK_TYPE STREQUAL "OpenBLAS")
  target_link_libraries(xmvb_vbscf PUBLIC ${LAPACK_LIB})
elseif (LAPACK_TYPE STREQUAL "refBLAS")
  target_link_libraries(xmvb_vbscf PUBLIC lapacke cblas lapack)
else()
  target_link_libraries(xmvb_vbscf PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

add_library(xmvb_runtime STATIC ${XMVB_RUNTIME_SOURCES})
target_include_directories(xmvb_runtime
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
target_link_libraries(xmvb_runtime PUBLIC xmvb_vbscf xmvb_openmp)

function(xmvb_link_standalone_runtime target_name)
  target_link_libraries(${target_name}
    PRIVATE
      ${CINT_LIB}
      xmvb_openmp
      xmvb_vbscf
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

add_executable(xmvb
  cli/main.cpp
  cli/options.cpp
  cli/report.cpp
  cli/run.cpp)
set_target_properties(xmvb PROPERTIES OUTPUT_NAME xmvb-cpp.exe)
target_link_libraries(xmvb
  PRIVATE
    xmvb_vbscf
    xmvb_runtime)
xmvb_link_standalone_runtime(xmvb)

file(
  GLOB_RECURSE XMVB_RUNTIME_BASIS_INPUTS
  "${CMAKE_SOURCE_DIR}/basis/*")
set(XMVB_RUNTIME_ASSET_STAMP
  "${CMAKE_BINARY_DIR}/CMakeFiles/xmvb_runtime_assets.stamp")
add_custom_command(
  OUTPUT ${XMVB_RUNTIME_ASSET_STAMP}
  COMMAND ${CMAKE_COMMAND}
          -DXMVB_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DXMVB_BINARY_DIR=${CMAKE_BINARY_DIR}
          -P ${CMAKE_SOURCE_DIR}/cmake/stage_runtime_assets.cmake
  COMMAND ${CMAKE_COMMAND} -E touch ${XMVB_RUNTIME_ASSET_STAMP}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/cmake/stage_runtime_assets.cmake
    ${XMVB_RUNTIME_BASIS_INPUTS}
  # `basis/` is copied into the build tree, so basis-file edits should restage
  # runtime assets. `data/` is linked instead of copied and contains volatile
  # training corpora, logs, and scratch outputs. Do not expand that directory
  # into the Ninja dependency graph, or every dataset churn will force a CMake
  # regenerate and invalidate the whole Ninja deps database.
  COMMENT "Staging standalone xmvb-cpp runtime assets into the build tree")
add_custom_target(
  xmvb_runtime_assets
  DEPENDS ${XMVB_RUNTIME_ASSET_STAMP})
add_dependencies(xmvb xmvb_runtime_assets)

if (XMVB_BUILD_DEV_TARGETS)
  include(tools/targets.cmake)
endif()
