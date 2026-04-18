#pragma once

#include <stddef.h>

#include "runtime_c/cpp_runtime_extractor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XmvbCppRuntimeHandle XmvbCppRuntimeHandle;

int xmvb_cpp_runtime_create(
    XmvbCppRuntimeHandle** runtime_handle,
    char* error_message,
    size_t error_message_capacity);

void xmvb_cpp_runtime_destroy(XmvbCppRuntimeHandle* runtime_handle);

int xmvb_cpp_runtime_load_input(
    XmvbCppRuntimeHandle* runtime_handle,
    char* input_file_path,
    const char* executable_path,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_initialize_wavefunction(
    XmvbCppRuntimeHandle* runtime_handle,
    const char* runtime_output_stem,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_prepare_libcint_buffers(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_run_vbprep(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_run_one_electron_integrals(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_run_two_electron_integrals(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_finalize_integral_storage(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_setup_hf(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_run_vbguess(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_clear_orbital_guess(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_copy_snapshot(
    const XmvbCppRuntimeHandle* runtime_handle,
    CppRuntimeSnapshot* snapshot,
    char* error_message,
    size_t error_message_capacity);

int xmvb_cpp_runtime_copy_snapshot_with_options(
    const XmvbCppRuntimeHandle* runtime_handle,
    int copy_legacy_ao_integrals,
    CppRuntimeSnapshot* snapshot,
    char* error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif
