cmake_minimum_required(VERSION 3.16)

project(xmvb_standalone LANGUAGES C CXX Fortran)

include(CTest)

file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" XMVB_CONDA_PREFIX)

option(
  XMVB_ENABLE_CCACHE
  "Use ccache as the compiler launcher when available."
  ON)

if (XMVB_ENABLE_CCACHE)
  find_program(XMVB_CCACHE_PROGRAM ccache)
  if (XMVB_CCACHE_PROGRAM)
    foreach(_xmvb_language C CXX Fortran)
      if (NOT DEFINED CMAKE_${_xmvb_language}_COMPILER_LAUNCHER OR
          CMAKE_${_xmvb_language}_COMPILER_LAUNCHER STREQUAL "")
        set(
          CMAKE_${_xmvb_language}_COMPILER_LAUNCHER
          "${XMVB_CCACHE_PROGRAM}"
          CACHE STRING
          "Compiler launcher for ${_xmvb_language}"
          FORCE)
      endif()
    endforeach()
    message(STATUS "XMVB-CPP: enabling ccache via ${XMVB_CCACHE_PROGRAM}")
  endif()
endif()

macro(_xmvb_default_root_from_conda cache_var description)
  if (NOT DEFINED ${cache_var} AND XMVB_CONDA_PREFIX)
    set(${cache_var} "${XMVB_CONDA_PREFIX}" CACHE PATH "${description}" FORCE)
  endif()
endmacro()

_xmvb_default_root_from_conda(LAPACK_ROOT_DIR "Root directory for OpenBLAS/LAPACK")
_xmvb_default_root_from_conda(CINT_ROOT_DIR "Root directory for Libcint")
_xmvb_default_root_from_conda(EIGEN3_ROOT_DIR "Root directory for Eigen3")

foreach(_xmvb_prefix
    ${XMVB_CONDA_PREFIX}
    ${LAPACK_ROOT_DIR}
    ${CINT_ROOT_DIR}
    ${EIGEN3_ROOT_DIR})
  if (_xmvb_prefix)
    list(PREPEND CMAKE_PREFIX_PATH "${_xmvb_prefix}")
  endif()
endforeach()
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

find_package(OpenMP REQUIRED)

if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json" FORCE)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-g -ggdb3 -fdiagnostics-color=always)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fpermissive>)
add_compile_options(-ffunction-sections -pipe)
add_link_options(-Wl,--gc-sections -Wl,--allow-shlib-undefined)

if (NOT DEFINED XMVB_BUILD_DEV_TARGETS)
  set(
    XMVB_BUILD_DEV_TARGETS
    OFF
    CACHE BOOL
    "Build C++ comparison, diagnostic, and test targets in addition to xmvb-cpp.exe."
    FORCE)
endif()

set(LAPACK_TYPE "OpenBLAS" CACHE STRING "Type of LAPACK library")

find_package(OpenBLAS CONFIG HINTS ${LAPACK_ROOT_DIR} ${XMVB_CONDA_PREFIX} REQUIRED)
find_library(
  LAPACK_LIB
  openblas
  HINTS
    ${LAPACK_ROOT_DIR}/lib
    ${LAPACK_ROOT_DIR}/lib64
    ${XMVB_CONDA_PREFIX}/lib
  REQUIRED)
find_path(
  LAPACK_INCLUDE
  NAMES lapack.h
  HINTS
    ${LAPACK_ROOT_DIR}/include
    ${LAPACK_ROOT_DIR}/include/openblas
    ${XMVB_CONDA_PREFIX}/include
    ${XMVB_CONDA_PREFIX}/include/openblas
  REQUIRED)
find_path(
  LAPACKE_INCLUDE
  NAMES lapacke.h
  HINTS
    ${LAPACK_ROOT_DIR}/include
    ${LAPACK_ROOT_DIR}/include/openblas
    ${XMVB_CONDA_PREFIX}/include
    ${XMVB_CONDA_PREFIX}/include/openblas
  REQUIRED)
set(LAPACKE_LIB "${LAPACK_LIB}" CACHE FILEPATH "LAPACKE library provided by OpenBLAS" FORCE)

find_library(
  CINT_LIB
  cint
  HINTS
    ${CINT_ROOT_DIR}/lib
    ${CINT_ROOT_DIR}/lib64
    ${XMVB_CONDA_PREFIX}/lib
  REQUIRED)
find_path(
  CINT_INCLUDE
  NAMES cint.h
  HINTS
    ${CINT_ROOT_DIR}/include
    ${XMVB_CONDA_PREFIX}/include
  REQUIRED)

add_subdirectory(src)
