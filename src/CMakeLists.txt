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

set(XMVB_INPUT_DECK_SOURCES
  input/deck/model.cpp
  input/deck/metadata.cpp
  input/deck/primary_basis.cpp
  input/deck/orbital_support.cpp
  input/deck/raw_structures.cpp
)

set(XMVB_LIBCINT_SOURCES
  libcint/auxiliary_basis.cpp
  libcint/direct_shell.cpp
  libcint/materialized_provider.cpp
  libcint/ri_provider.cpp
  libcint/materialized_input.cpp
)

set(XMVB_GUESS_SOURCES
  guess/block.cpp
  guess/closed_shell_fock.cpp
  guess/initial.cpp
  guess/restricted_hf.cpp
)

set(XMVB_INPUT_SOURCES
  input/loading/generated_structures.cpp
  input/loading/loader.cpp
)

set(XMVB_OUTPUT_SOURCES
  output/molden/writer.cpp
  output/text/report.cpp
  output/text/sections.cpp
  output/trace/accepted_iteration.cpp
  output/trace/tnhvp.cpp
)

# Keep component ownership as strict as the canonical VBSCF manifest. A new
# translation unit must be assigned to exactly one component before it builds.
file(
  GLOB_RECURSE _xmvb_discovered_component_sources
  CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/guess/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/input/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/libcint/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/*.cpp")
set(_xmvb_declared_component_sources
  ${XMVB_INPUT_DECK_SOURCES}
  ${XMVB_LIBCINT_SOURCES}
  ${XMVB_GUESS_SOURCES}
  ${XMVB_INPUT_SOURCES}
  ${XMVB_OUTPUT_SOURCES})
list(SORT _xmvb_declared_component_sources)
list(SORT _xmvb_discovered_component_sources)
if (NOT "${_xmvb_declared_component_sources}" STREQUAL
    "${_xmvb_discovered_component_sources}")
  message(FATAL_ERROR
    "src/CMakeLists.txt does not own every component .cpp exactly once.\n"
    "Declared: ${_xmvb_declared_component_sources}\n"
    "Discovered: ${_xmvb_discovered_component_sources}")
endif()
unset(_xmvb_declared_component_sources)
unset(_xmvb_discovered_component_sources)

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
target_link_libraries(xmvb_vbscf
  PUBLIC
    ${CINT_LIB}
    ${LAPACKE_LIB}
    xmvb_openmp
    m)
if (LAPACK_TYPE STREQUAL "OpenBLAS")
  target_link_libraries(xmvb_vbscf PUBLIC ${LAPACK_LIB})
elseif (LAPACK_TYPE STREQUAL "refBLAS")
  target_link_libraries(xmvb_vbscf PUBLIC lapacke cblas lapack)
else()
  target_link_libraries(xmvb_vbscf PUBLIC ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
endif()

function(xmvb_add_component target_name)
  add_library(${target_name} STATIC ${ARGN})
  target_include_directories(${target_name}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${EIGEN3_INCLUDE_DIR}
      ${LAPACK_INCLUDE}
      ${LAPACKE_INCLUDE}
      ${CINT_INCLUDE})
endfunction()

xmvb_add_component(xmvb_libcint ${XMVB_LIBCINT_SOURCES})
target_link_libraries(xmvb_libcint PUBLIC xmvb_vbscf)

xmvb_add_component(xmvb_input_deck ${XMVB_INPUT_DECK_SOURCES})
target_link_libraries(xmvb_input_deck PUBLIC xmvb_libcint xmvb_vbscf)

xmvb_add_component(xmvb_guess ${XMVB_GUESS_SOURCES})
target_link_libraries(
  xmvb_guess
  PUBLIC
    xmvb_input_deck
    xmvb_libcint
    xmvb_vbscf)

xmvb_add_component(xmvb_input ${XMVB_INPUT_SOURCES})
target_link_libraries(
  xmvb_input
  PUBLIC
    xmvb_guess
    xmvb_input_deck
    xmvb_libcint
    xmvb_vbscf)

xmvb_add_component(xmvb_output ${XMVB_OUTPUT_SOURCES})
target_compile_definitions(
  xmvb_output
  PRIVATE XMVB_VERSION="${PROJECT_VERSION}")
target_link_libraries(
  xmvb_output
  PUBLIC
    xmvb_input
    xmvb_libcint
    xmvb_vbscf)

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
  cli/run.cpp)
set_target_properties(xmvb PROPERTIES OUTPUT_NAME xmvb-cpp.exe)
target_link_libraries(xmvb
  PRIVATE
    xmvb_input
    xmvb_output
    xmvb_vbscf)

file(
  GLOB_RECURSE XMVB_BASIS_INPUTS
  "${CMAKE_SOURCE_DIR}/basis/*")
set(XMVB_ASSET_STAMP
  "${CMAKE_BINARY_DIR}/CMakeFiles/xmvb_assets.stamp")
add_custom_command(
  OUTPUT ${XMVB_ASSET_STAMP}
  COMMAND ${CMAKE_COMMAND}
          -DXMVB_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DXMVB_BINARY_DIR=${CMAKE_BINARY_DIR}
          -P ${CMAKE_SOURCE_DIR}/cmake/stage_assets.cmake
  COMMAND ${CMAKE_COMMAND} -E touch ${XMVB_ASSET_STAMP}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/cmake/stage_assets.cmake
    ${XMVB_BASIS_INPUTS}
  # `basis/` is copied into the build tree, so basis-file edits should restage
  # build assets. `data/` is linked instead of copied and contains volatile
  # training corpora, logs, and scratch outputs. Do not expand that directory
  # into the Ninja dependency graph, or every dataset churn will force a CMake
  # regenerate and invalidate the whole Ninja deps database.
  COMMENT "Staging xmvb-cpp basis and data assets into the build tree")
add_custom_target(
  xmvb_assets
  DEPENDS ${XMVB_ASSET_STAMP})
add_dependencies(xmvb xmvb_assets)

if (BUILD_TESTING)
  add_executable(
    test_input_guess_selection
    ${CMAKE_SOURCE_DIR}/tests/input/test_guess_selection.cpp)
  target_link_libraries(test_input_guess_selection PRIVATE xmvb_input_deck)
  add_test(NAME input_guess_selection COMMAND test_input_guess_selection)

  add_test(
    NAME text_report_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      -P ${CMAKE_SOURCE_DIR}/tests/output/check_f2_report.cmake)
  set_tests_properties(
    text_report_f2
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
  add_test(
    NAME tnhvp_trace_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      -DXMVB_TRACE_ROOT=${CMAKE_BINARY_DIR}/tests/tnhvp_trace
      -P ${CMAKE_SOURCE_DIR}/tests/output/check_tnhvp_trace.cmake)
  set_tests_properties(
    tnhvp_trace_f2
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
  add_test(
    NAME eigensolver_equivalence_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      -P ${CMAKE_SOURCE_DIR}/tests/output/check_eigensolver_equivalence.cmake)
  set_tests_properties(
    eigensolver_equivalence_f2
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
  add_test(
    NAME eigensolver_equivalence_f2_oeo
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_OEO.xmi
      -P ${CMAKE_SOURCE_DIR}/tests/output/check_eigensolver_equivalence.cmake)
  set_tests_properties(
    eigensolver_equivalence_f2_oeo
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")
endif()

if (XMVB_BUILD_DEV_TARGETS)
  include(tools/targets.cmake)
endif()
