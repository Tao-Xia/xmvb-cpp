if (NOT DEFINED XMVB_SOURCE_DIR)
  message(FATAL_ERROR "XMVB_SOURCE_DIR is required")
endif()

if (NOT DEFINED XMVB_BINARY_DIR)
  message(FATAL_ERROR "XMVB_BINARY_DIR is required")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${XMVB_SOURCE_DIR}/basis"
          "${XMVB_BINARY_DIR}/basis"
  RESULT_VARIABLE copy_basis_result)
if (NOT copy_basis_result EQUAL 0)
  message(FATAL_ERROR "failed to stage basis directory")
endif()

if (EXISTS "${XMVB_BINARY_DIR}/data" OR IS_SYMLINK "${XMVB_BINARY_DIR}/data")
  file(REMOVE_RECURSE "${XMVB_BINARY_DIR}/data")
endif()

file(CREATE_LINK
  "${XMVB_SOURCE_DIR}/data"
  "${XMVB_BINARY_DIR}/data"
  SYMBOLIC)
