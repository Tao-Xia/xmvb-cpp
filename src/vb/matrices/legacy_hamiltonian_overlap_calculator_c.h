#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int legacy_calculate_hamiltonian_overlap(
    const char* input_file_path,
    int n_threads,
    int run_orbital_optimization,
    int* n_structures,
    double* total_energy,
    double* one_electron_energy,
    double** hamiltonian_matrix,
    double** overlap_matrix,
    char* error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif
