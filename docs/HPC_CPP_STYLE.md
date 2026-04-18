# HPC C++ Style

This repository contains performance-sensitive numerical chemistry code. New code should optimize for clarity, reuse, and stable conventions rather than for local shorthand.

## File Responsibilities

- Headers own reusable declarations. Put reusable classes, structs, helper types, and function declarations in `*.hpp`.
- Source files own implementations. Keep `*.cpp` focused on function bodies and truly file-local helpers inside anonymous namespaces.
- Do not duplicate reusable local machinery into another `*.cpp`. If a helper is needed in more than one place, move it into a shared header or module.

## Eigen Usage

- Default dense matrix type: `Eigen::MatrixXd`.
- Default dense vector type: `Eigen::VectorXd`.
- The repository-wide dense-storage convention is column-major. New flattened matrix buffers must follow the same column-major layout as the corresponding `Eigen::MatrixXd` view.
- Do not use `std::vector<double>` to represent a mathematical dense matrix or dense vector in modern C++ code.
- If the data has matrix semantics, store it as `Eigen::MatrixXd`.
- If the data has vector semantics, store it as `Eigen::VectorXd`.
- `std::vector<T>` is for true 1D data only: index lists, sparse/picked entries, packed-triangle storage, raw IO buffers, or other layouts that are genuinely not dense Eigen matrices.
- If a legacy kernel still needs one row-contiguous compatibility buffer, keep that buffer local, name it with a `_buffer` suffix, and convert at the boundary with explicit helpers such as `copy_matrix_to_legacy_row_buffer(...)` and `copy_legacy_row_buffer_to_matrix(...)`.
- Do not add aliases such as `using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;`.
- Do not introduce new row-major dense storage in C++ code.
- Existing row-major legacy or interoperability paths must be treated as migration debt. Isolate them, document the reason, and prefer converting at the boundary into column-major storage.
- Prefer direct `Eigen::Map<const Eigen::MatrixXd>` or `Eigen::Ref<const Eigen::MatrixXd>` forms over introducing project-local shorthand aliases.
- Do not do row-pointer arithmetic on `Eigen::MatrixXd`. In column-major storage, columns are contiguous and rows are generally strided. If a legacy kernel truly needs one row as a contiguous buffer, make that copy explicitly at the boundary.

## Debug And Monitoring Code

- Do not add new `std::cout`, `std::cerr`, or ad hoc diagnostic prints in hot paths.
- Do not add environment-gated probe toggles or wrapper functions unless the behavior is a maintained product feature.
- If temporary diagnostics are needed during development, keep them local and remove them before landing the change.

## Reuse And Duplication

- Reuse shared helpers for structure-basis, determinant-pair, and spin-pair logic instead of copying the same algorithm into another file.
- When removing deprecated paths, delete the dead code rather than keeping inactive wrappers or alternate implementations around.
- Prefer one canonical implementation for each mathematical transformation.

## Comments

- Nontrivial numerical code needs comments in both the header and the implementation file.
- Implementation comments should explain the mathematical role of the routine, the meaning of key intermediates, and the expected dimensions or index conventions.
- Avoid comments that only restate syntax.
