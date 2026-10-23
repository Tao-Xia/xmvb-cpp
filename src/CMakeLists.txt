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

set(XMVB_INTEGRAL_SOURCES
  integrals/ao/graph.cpp
  integrals/ao/libcint.cpp
  integrals/ao/exact.cpp
  integrals/ao/exact_rows.cpp
  integrals/ao/ri_operator.cpp
  integrals/ao/ri_transform.cpp
  integrals/jk/contraction.cpp
  integrals/basis/values.cpp
  integrals/active/one_electron.cpp
  integrals/active/exact.cpp
  integrals/active/ri.cpp
  integrals/active/ri_transform.cpp
  integrals/active/matrix_adjoint.cpp
  integrals/active/two_electron.cpp
  integrals/active/pair_map.cpp
  integrals/active/transform.cpp
  integrals/active/raw.cpp
  integrals/response/adjoint.cpp
  integrals/response/directional.cpp
  integrals/response/ri.cpp
  integrals/response/ri_ao.cpp
  integrals/response/orbital_pair.cpp
  integrals/response/density_diagonal.cpp
  integrals/response/exchange.cpp
  integrals/response/pair_transforms.cpp
  integrals/response/ao_pair_operator.cpp
  integrals/response/generated_rows.cpp)

set(XMVB_LINALG_SOURCES
  core/memory.cpp
  core/pair_index.cpp
  linalg/exponential.cpp
  linalg/eigensolver.cpp
  linalg/eigen_response.cpp
  linalg/minres.cpp)

set(XMVB_MCSCF_SOURCES
  mcscf/ci/sigma.cpp
  mcscf/ci/sigma_tiles.cpp
  mcscf/ci/workspace.cpp
  mcscf/ci/sigma_plan.cpp
  mcscf/ci/generator_tiles.cpp
  mcscf/ci/spin_channels.cpp
  mcscf/ci/factors.cpp
  mcscf/ci/factor_generator.cpp
  mcscf/ci/factor_plan.cpp
  mcscf/ci/planner.cpp
  mcscf/ci/strings.cpp
  mcscf/ci/spin.cpp
  mcscf/states/weights.cpp)

# Model-independent optimizers are shared by HF/KS, VBSCF and CASSCF.
set(XMVB_OPTIMIZATION_SOURCES
  mcscf/optimization/ciah/solver.cpp
  mcscf/optimization/ciah/driver.cpp
  mcscf/optimization/neo/globalization.cpp
  mcscf/optimization/problem.cpp
  mcscf/optimization/neo/solver.cpp
  mcscf/optimization/preconditioners/block_inverse_bfgs.cpp
  mcscf/optimization/preconditioners/coarse.cpp
  mcscf/optimization/preconditioners/orbital_diagonal.cpp
  mcscf/optimization/preconditioners/shifted_metric.cpp)
set(XMVB_CASSCF_SOURCES
  casscf/integrals/active.cpp
  casscf/orbitals/rotation.cpp
  casscf/orbitals/frame.cpp
  casscf/derivatives/point.cpp
  casscf/derivatives/diagonal.cpp
  casscf/derivatives/coupled.cpp
  casscf/driver/casci.cpp
  casscf/driver/casscf.cpp)

set(XMVB_INPUT_DECK_SOURCES
  input/deck/model.cpp
  input/deck/metadata.cpp
  input/deck/primary_basis.cpp
  input/deck/orbital_support.cpp
  input/deck/raw_structures.cpp
)

set(XMVB_LIBCINT_SOURCES
  libcint/direct_shell.cpp
  libcint/materialized_provider.cpp
  libcint/ri_provider.cpp
  libcint/materialized_input.cpp
)

set(XMVB_GUESS_SOURCES
  guess/block.cpp
  guess/initial.cpp
)

set(XMVB_SCF_SOURCES
  scf/optimization/orbitals.cpp
  scf/optimization/ciah.cpp
  scf/driver/solver.cpp
  scf/mixing/diis.cpp
  scf/hf/potential.cpp)

set(XMVB_DFT_SOURCES
  dft/grid/atomic.cpp
  dft/grid/partition.cpp
  dft/xc/functional.cpp
  dft/integration/cache.cpp
  dft/integration/density.cpp
  dft/integration/potential.cpp
  dft/driver/ks.cpp
  dft/driver/potential.cpp)

set(XMVB_INPUT_SOURCES
  input/loading/generated_structures.cpp
  input/loading/loader.cpp
  input/loading/casscf.cpp
  input/loading/molecule.cpp
)

set(XMVB_OUTPUT_SOURCES
  output/molden/writer.cpp
  output/orbitals/writer.cpp
  output/text/report.cpp
  output/text/sections.cpp
  output/text/structure.cpp
  output/text/normalization.cpp
  output/trace/accepted_iteration.cpp
  output/trace/ciah.cpp
  output/trace/tnhvp.cpp
)

# Keep component ownership as strict as the canonical VBSCF manifest. A new
# translation unit must be assigned to exactly one component before it builds.
file(
  GLOB_RECURSE _xmvb_discovered_component_sources
  CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/guess/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/scf/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/dft/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/input/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/libcint/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/output/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/linalg/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/integrals/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mcscf/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/casscf/*.cpp")
set(_xmvb_declared_component_sources
  ${XMVB_INPUT_DECK_SOURCES}
  ${XMVB_LIBCINT_SOURCES}
  ${XMVB_GUESS_SOURCES}
  ${XMVB_SCF_SOURCES}
  ${XMVB_DFT_SOURCES}
  ${XMVB_INPUT_SOURCES}
  ${XMVB_OUTPUT_SOURCES}
  ${XMVB_LINALG_SOURCES}
  ${XMVB_INTEGRAL_SOURCES}
  ${XMVB_MCSCF_SOURCES}
  ${XMVB_OPTIMIZATION_SOURCES}
  ${XMVB_CASSCF_SOURCES})
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

xmvb_add_component(xmvb_linalg ${XMVB_LINALG_SOURCES})
target_compile_features(xmvb_linalg PUBLIC cxx_std_23)
if (XMVB_ENABLE_EIGEN_BLAS_BACKEND)
  target_compile_definitions(xmvb_linalg PUBLIC EIGEN_USE_BLAS=1 EIGEN_USE_LAPACKE=1)
endif()
target_link_libraries(xmvb_linalg PUBLIC ${LAPACKE_LIB} ${LAPACK_LIB} xmvb_openmp m)

xmvb_add_component(xmvb_integrals
  ${XMVB_INTEGRAL_SOURCES})
target_link_libraries(xmvb_integrals PUBLIC xmvb_linalg ${CINT_LIB})

xmvb_add_component(xmvb_optimization ${XMVB_OPTIMIZATION_SOURCES})
target_link_libraries(xmvb_optimization PUBLIC xmvb_linalg)

xmvb_add_component(xmvb_scf ${XMVB_SCF_SOURCES})
target_link_libraries(xmvb_scf PUBLIC xmvb_integrals xmvb_optimization)

xmvb_add_component(xmvb_dft ${XMVB_DFT_SOURCES})
target_include_directories(xmvb_dft PRIVATE ${LIBXC_INCLUDE_DIR})
target_link_libraries(xmvb_dft PUBLIC xmvb_scf PRIVATE ${LIBXC_LIBRARY})

xmvb_add_component(xmvb_mcscf ${XMVB_MCSCF_SOURCES})
target_link_libraries(xmvb_mcscf PUBLIC xmvb_linalg xmvb_optimization)
target_link_libraries(xmvb_mcscf PUBLIC xmvb_integrals)
target_link_libraries(xmvb_vbscf PUBLIC xmvb_mcscf xmvb_integrals)

xmvb_add_component(xmvb_casscf ${XMVB_CASSCF_SOURCES})
target_link_libraries(xmvb_casscf PUBLIC xmvb_mcscf)

# The common CI and orthogonal model must never depend on VB representations.
file(GLOB_RECURSE _xmvb_common_headers CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/linalg/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/linalg/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/parallel/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/scf/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/scf/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/dft/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/dft/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/integrals/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/integrals/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/libcint/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/libcint/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mcscf/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/mcscf/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/casscf/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/casscf/*.cpp")
foreach(_xmvb_common_file IN LISTS _xmvb_common_headers)
  file(STRINGS "${_xmvb_common_file}" _xmvb_vb_includes
    REGEX "^[ \t]*#[ \t]*include[ \t]*\"vbscf/")
  if (_xmvb_vb_includes)
    message(FATAL_ERROR "Common/CASSCF source depends on VBSCF: ${_xmvb_common_file}")
  endif()
  if (_xmvb_common_file MATCHES "/(linalg|parallel|scf|dft)/")
    file(STRINGS "${_xmvb_common_file}" _xmvb_model_includes
      REGEX "^[ \t]*#[ \t]*include[ \t]*\"(mcscf|casscf)/")
    foreach(_xmvb_model_include IN LISTS _xmvb_model_includes)
      if (NOT (_xmvb_common_file MATCHES "/(scf|dft)/" AND
          _xmvb_model_include MATCHES "\"mcscf/optimization/"))
        message(FATAL_ERROR "Numerical utility depends on an electronic model: ${_xmvb_common_file}")
      endif()
    endforeach()
  endif()
  if (_xmvb_common_file MATCHES "/integrals/" AND NOT
      _xmvb_common_file MATCHES "/casscf/integrals/")
    file(STRINGS "${_xmvb_common_file}" _xmvb_integral_model_includes
      REGEX "^[ \t]*#[ \t]*include[ \t]*\"(mcscf|casscf|input|guess|libcint|output)/")
    if (_xmvb_integral_model_includes)
      message(FATAL_ERROR "Integral kernel depends on a model or I/O: ${_xmvb_common_file}")
    endif()
  endif()
endforeach()
unset(_xmvb_common_headers)
unset(_xmvb_common_file)
unset(_xmvb_vb_includes)
unset(_xmvb_model_includes)
unset(_xmvb_integral_model_includes)

if (XMVB_BUILD_DEV_TARGETS)
  foreach(_xmvb_dft_test basis grid functional xc)
    add_executable(test_dft_${_xmvb_dft_test}
      ${CMAKE_SOURCE_DIR}/tests/dft/test_${_xmvb_dft_test}.cpp)
    target_link_libraries(test_dft_${_xmvb_dft_test} PRIVATE xmvb_dft)
    if (BUILD_TESTING)
      add_test(NAME dft_${_xmvb_dft_test} COMMAND test_dft_${_xmvb_dft_test})
      set_tests_properties(dft_${_xmvb_dft_test} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    endif()
  endforeach()
  unset(_xmvb_dft_test)
  add_executable(dft_molecule ${CMAKE_SOURCE_DIR}/tests/dft/molecule.cpp)
  target_link_libraries(dft_molecule PRIVATE xmvb_dft xmvb_input)
  foreach(_xmvb_scf_test jk solver orbitals)
    add_executable(test_scf_${_xmvb_scf_test}
      ${CMAKE_SOURCE_DIR}/tests/scf/test_${_xmvb_scf_test}.cpp)
    target_link_libraries(test_scf_${_xmvb_scf_test} PRIVATE xmvb_scf)
    if (BUILD_TESTING)
      add_test(NAME scf_${_xmvb_scf_test} COMMAND test_scf_${_xmvb_scf_test})
      set_tests_properties(scf_${_xmvb_scf_test} PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    endif()
  endforeach()
  unset(_xmvb_scf_test)
  include(${CMAKE_SOURCE_DIR}/tests/linalg/targets.cmake)
  include(${CMAKE_SOURCE_DIR}/tests/mcscf/targets.cmake)
  add_executable(test_workspace ${CMAKE_SOURCE_DIR}/tests/core/test_workspace.cpp)
  target_include_directories(test_workspace PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(test_workspace PRIVATE xmvb_linalg)
  add_executable(test_memory ${CMAKE_SOURCE_DIR}/tests/core/test_memory.cpp)
  target_link_libraries(test_memory PRIVATE xmvb_linalg)
  if (BUILD_TESTING)
    add_test(NAME memory COMMAND test_memory)
    add_test(NAME workspace COMMAND test_workspace)
  endif()
  add_executable(test_openmp ${CMAKE_SOURCE_DIR}/tests/parallel/test_openmp.cpp)
  target_include_directories(test_openmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(test_openmp PRIVATE xmvb_openmp)
  if (BUILD_TESTING)
    add_test(NAME openmp COMMAND test_openmp)
  endif()
  add_executable(test_ci_factors ${CMAKE_SOURCE_DIR}/tests/mcscf/test_factors.cpp)
  target_link_libraries(test_ci_factors PRIVATE xmvb_mcscf)
  add_executable(test_casci ${CMAKE_SOURCE_DIR}/tests/mcscf/test_casci.cpp)
  target_link_libraries(test_casci PRIVATE xmvb_casscf)
  add_executable(test_casscf_derivatives ${CMAKE_SOURCE_DIR}/tests/mcscf/test_casscf_derivatives.cpp)
  target_link_libraries(test_casscf_derivatives PRIVATE xmvb_casscf)
  add_executable(test_casscf_coupled ${CMAKE_SOURCE_DIR}/tests/mcscf/test_casscf_coupled.cpp)
  target_link_libraries(test_casscf_coupled PRIVATE xmvb_casscf)
  add_executable(casscf_fixture
    ${CMAKE_SOURCE_DIR}/tools/casscf/main.cpp
    ${CMAKE_SOURCE_DIR}/tools/casscf/fixture.cpp)
  target_link_libraries(casscf_fixture PRIVATE xmvb_casscf)
  add_executable(test_integrals ${CMAKE_SOURCE_DIR}/tests/mcscf/test_integrals.cpp)
  target_link_libraries(test_integrals PRIVATE xmvb_integrals)
  if (BUILD_TESTING)
    add_test(NAME ci_factors COMMAND test_ci_factors)
    add_test(NAME casscf_xmi
      COMMAND ${CMAKE_COMMAND}
        -DPROGRAM=$<TARGET_FILE:xmvb>
        -DINPUT=${CMAKE_SOURCE_DIR}/testdata/casscf/H2.xmi
        -DRUN_DIR=${CMAKE_CURRENT_BINARY_DIR}/casscf-xmi-test
        -P ${CMAKE_SOURCE_DIR}/tests/mcscf/check_casscf_xmi.cmake)
    set_tests_properties(casscf_xmi PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    set_tests_properties(ci_factors PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    add_test(NAME casci COMMAND test_casci)
    add_test(NAME casscf_derivatives COMMAND test_casscf_derivatives)
    add_test(NAME casscf_coupled COMMAND test_casscf_coupled)
    set_tests_properties(casscf_derivatives PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    set_tests_properties(casscf_coupled PROPERTIES
      ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    add_test(NAME integrals COMMAND test_integrals)
    set_tests_properties(integrals PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
    set_tests_properties(casci PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
  endif()
endif()

xmvb_add_component(xmvb_libcint ${XMVB_LIBCINT_SOURCES})
target_link_libraries(xmvb_libcint PUBLIC xmvb_integrals)

xmvb_add_component(xmvb_input_deck ${XMVB_INPUT_DECK_SOURCES})
target_link_libraries(xmvb_input_deck PUBLIC xmvb_libcint xmvb_vbscf)

xmvb_add_component(xmvb_guess ${XMVB_GUESS_SOURCES})
target_link_libraries(
  xmvb_guess
  PUBLIC
    xmvb_scf
    xmvb_input_deck
    xmvb_libcint
    xmvb_vbscf)

xmvb_add_component(xmvb_input ${XMVB_INPUT_SOURCES})
target_link_libraries(xmvb_input PUBLIC xmvb_casscf)
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
  cli/run.cpp
  cli/dft.cpp
  cli/casscf.cpp)
set_target_properties(xmvb PROPERTIES OUTPUT_NAME xmvb-cpp.exe)
target_link_libraries(xmvb
  PRIVATE
    xmvb_input
    xmvb_output
    xmvb_casscf
    xmvb_dft
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
  add_test(NAME dft_xmi
    COMMAND ${CMAKE_COMMAND}
      -DPROGRAM=$<TARGET_FILE:xmvb>
      -DINPUT=${CMAKE_SOURCE_DIR}/testdata/dft/He.xmi
      -DUKS_INPUT=${CMAKE_SOURCE_DIR}/testdata/dft/H_PBE.xmi
      -DBASIS=${CMAKE_SOURCE_DIR}/basis/cc-pvdz.gbs
      -DRUN_DIR=${CMAKE_CURRENT_BINARY_DIR}/dft-xmi-test
      -P ${CMAKE_SOURCE_DIR}/tests/dft/check_xmi.cmake)
  set_tests_properties(dft_xmi PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
  add_executable(test_output_orbitals ${CMAKE_SOURCE_DIR}/tests/output/test_orbitals.cpp)
  target_link_libraries(test_output_orbitals PRIVATE xmvb_output xmvb_guess)
  add_test(NAME output_orbitals COMMAND test_output_orbitals)

  add_executable(
    test_output_structure
    output/text/structure.cpp
    ${CMAKE_SOURCE_DIR}/tests/output/test_structure.cpp)
  target_include_directories(
    test_output_structure
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${EIGEN3_INCLUDE_DIR})
  add_test(NAME output_structure COMMAND test_output_structure)

  add_executable(test_output_normalization ${CMAKE_SOURCE_DIR}/tests/output/test_normalization.cpp)
  target_link_libraries(test_output_normalization PRIVATE xmvb_output)
  add_test(NAME output_normalization COMMAND test_output_normalization)

  add_executable(
    test_cli_options
    cli/options.cpp
    ${CMAKE_SOURCE_DIR}/tests/cli/test_options.cpp)
  target_include_directories(
    test_cli_options
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${EIGEN3_INCLUDE_DIR})
  add_test(NAME cli_options COMMAND test_cli_options)

  add_executable(test_product_frontier ${CMAKE_SOURCE_DIR}/tests/mcscf/test_product_frontier.cpp)
  target_link_libraries(test_product_frontier PRIVATE xmvb_mcscf)
  add_test(NAME product_frontier COMMAND test_product_frontier)

  add_executable(test_ri_ao_response ${CMAKE_SOURCE_DIR}/tests/mcscf/test_ri_ao_response.cpp)
  target_link_libraries(test_ri_ao_response PRIVATE xmvb_integrals)
  add_test(NAME ri_ao_response COMMAND test_ri_ao_response)
  set_tests_properties(ri_ao_response PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1")

  add_executable(
    test_input_auxiliary_basis
    ${CMAKE_SOURCE_DIR}/tests/input/test_auxiliary_basis.cpp)
  target_link_libraries(test_input_auxiliary_basis PRIVATE xmvb_input_deck)

  add_executable(test_libcint_ri ${CMAKE_SOURCE_DIR}/tests/vbscf/unit/test_libcint_ri.cpp)
  target_link_libraries(test_libcint_ri PRIVATE xmvb_libcint)
  add_test(NAME libcint_ri COMMAND test_libcint_ri)
  set_tests_properties(libcint_ri PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")

  add_executable(test_libcint_shell_screening
    ${CMAKE_SOURCE_DIR}/tests/vbscf/unit/test_libcint_shell_screening.cpp)
  target_link_libraries(test_libcint_shell_screening PRIVATE xmvb_libcint)
  add_test(NAME libcint_shell_screening COMMAND test_libcint_shell_screening)
  set_tests_properties(libcint_shell_screening PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1")
  add_test(
    NAME input_auxiliary_basis
    COMMAND
      test_input_auxiliary_basis
      ${CMAKE_SOURCE_DIR}/testdata/vbscf/MnF2.xmi)

  add_executable(
    test_input_guess_selection
    ${CMAKE_SOURCE_DIR}/tests/input/test_guess_selection.cpp)
  target_link_libraries(test_input_guess_selection PRIVATE xmvb_input_deck)
  add_test(NAME input_guess_selection COMMAND test_input_guess_selection)

  add_executable(
    test_generated_structures
    ${CMAKE_SOURCE_DIR}/tests/input/test_generated_structures.cpp)
  target_link_libraries(test_generated_structures PRIVATE xmvb_input)
  add_test(NAME generated_structures COMMAND test_generated_structures)

  add_executable(
    test_full_space
    ${CMAKE_SOURCE_DIR}/tests/input/test_full_space.cpp)
  target_link_libraries(test_full_space PRIVATE xmvb_input)
  add_test(NAME full_space COMMAND test_full_space)

  add_test(
    NAME state_average_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2_SA2.xmi
      -P ${CMAKE_SOURCE_DIR}/tests/output/check_f2_state_average.cmake)
  set_tests_properties(
    state_average_f2
    PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      ENVIRONMENT
        "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;GOTO_NUM_THREADS=1;MKL_NUM_THREADS=1")

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
    NAME ri_tnhvp_f2
    COMMAND
      ${CMAKE_COMMAND}
      -DXMVB_EXECUTABLE=$<TARGET_FILE:xmvb>
      -DXMVB_INPUT=${CMAKE_SOURCE_DIR}/testdata/vbscf/F2.xmi
      -DXMVB_SCRATCH=${CMAKE_BINARY_DIR}/tests/ri_tnhvp_f2
      -P ${CMAKE_SOURCE_DIR}/tests/output/check_f2_ri_tnhvp.cmake)
  set_tests_properties(
    ri_tnhvp_f2
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

if (XMVB_ENABLE_CUDA)
  add_subdirectory(cuda)
  if (XMVB_BUILD_DEV_TARGETS)
    include(${CMAKE_SOURCE_DIR}/tests/cuda/targets.cmake)
  endif()
endif()
