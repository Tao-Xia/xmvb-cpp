# Column-Major Migration

This note records the remaining row-major storage hotspots that must be retired so the C++ codebase converges on one dense-storage convention: column-major.

## Policy

- New C++ dense matrix storage must be column-major.
- New C++ matrix/vector semantics must use Eigen types, not flattened `std::vector<double>`.
- Existing row-major buffers are migration debt, not an accepted parallel style.
- Mechanical type replacement is not sufficient when a flattened `std::vector<double>` is part of the data contract. Those buffers must be migrated together with every producer and consumer.

## High-Risk Contract Chains

### Active-space two-electron C++ pipeline

These public buffers historically carried explicit row-major layout contracts and must stay migrated together as one coherent slice:

- `ActiveSpaceTwoElectronResult::ri_active_pair_factors`
- `ActiveSpaceTwoElectronResult::dense_active_coefficients`
- `ActiveSpaceTwoElectronResult::dense_ao_pair_products`
- `LibcintRiIntegralProviderResult::metric_whitened_ao_pair_factors`

The end state for this chain is not "row-major vector replaced by col-major vector". The end state is `Eigen::MatrixXd` at public and cross-module boundaries, with any unavoidable legacy flattening isolated to narrow compatibility shims.

Current status:

- `ActiveSpaceTwoElectronResult::*` dense matrix caches are already `Eigen::MatrixXd`.
- `LibcintRiIntegralProviderResult::auxiliary_metric_matrix` and `metric_whitened_ao_pair_factors` are already `Eigen::MatrixXd`.
- `OrbitalPreparationResult::inactive_density_low_rank_factors` and `AoEffectiveOneElectronRiLowRankFactors::scaled_factor_matrix` are already `Eigen::MatrixXd`.
- `OrbitalPreparationResult::auxiliary_orbital_matrix` and `inactive_density_matrix` are already `Eigen::MatrixXd`.
- `ActiveSpaceOneElectronBuilder`, `ActiveSpaceMatrixBackpropagator`, `AoEffectiveOneElectronBuilder`, and `AoEffectiveOneElectronBackpropagator` now accept those orbital-preparation matrices through `Eigen::Ref<const Eigen::MatrixXd>` compatibility boundaries.
- Do not reintroduce row-pointer arithmetic over RI packed-factor rows. Column-major Eigen matrices do not store logical rows contiguously.

Primary implementation files on this chain:

- `src/vb/orbital/active_space_two_electron_builder.cpp`
- `src/vb/orbital/active_space_two_electron_backpropagator.cpp`
- `src/vb/orbital/active_space_two_electron_utils.cpp`
- `src/vb/scf/cpp_orbital_gradient_evaluator.cpp`
- `src/vb/scf/exact_orbital_second_order_operator.cpp`
- `src/vb/orbital/ri_active_space_two_electron_builder.cpp`
- `src/runtime/libcint_ri_integral_provider.cpp`
- `src/vb/orbital/ao_effective_one_electron_ri_operator.cpp`

### Determinant / structure backward kernels

These files still map coefficient tables through row-major Eigen types and need a second migration wave after the active-space pipeline settles:

- `src/vb/scf/opposite_spin_matrix_backward.cpp`
- `src/vb/scf/same_spin_matrix_backward.cpp`
- `src/vb/biorthogonal_vbscf/biorthogonal_opposite_spin_matrix_backward.cpp`
- `src/vb/biorthogonal_vbscf/biorthogonal_same_spin_matrix_backward.cpp`

## Legacy C Runtime And BLAS

The legacy C runtime and older C kernels still call BLAS through `CblasRowMajor`. These files are separate from the modern C++ Eigen cleanup and should be migrated deliberately:

- `src/fock/*.c`
- `src/mol/xgrids.c`
- `src/runtime_c/**/*.c`
- `src/solver/diis.c`

## Recommended Order

1. Migrate the active-space two-electron C++ buffer contracts to column-major and validate exact-mode smoke tests.
2. Remove remaining internal row-major compatibility shims from the active-space and RI backward kernels.
3. Migrate same-spin / opposite-spin backward coefficient maps.
4. Tackle legacy C `CblasRowMajor` code only after the modern C++ path is stable.
