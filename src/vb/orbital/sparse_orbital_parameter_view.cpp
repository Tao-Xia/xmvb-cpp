#include "vb/orbital/sparse_orbital_parameter_view.hpp"

#include <stdexcept>

namespace xmvb::vb {

namespace {

int get_differentiable_coefficient_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  // Legacy `getvars` parameterizes HAO orbitals with `ma0` variables, not with
  // the fully expanded support length stored in `ma/nv`. Preserving that count
  // here keeps the C++ optimizer on the same orbital manifold as legacy VBSCF.
  const bool have_original_counts =
      orbital_preparation_input.original_orbital_basis_counts.size() ==
      orbital_preparation_input.n_orbitals;
  const int explicit_count = have_original_counts
      ? orbital_preparation_input.original_orbital_basis_counts[orbital_index]
      : orbital_preparation_input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }

  if (explicit_count == 1) {
    return 1;
  }

  int coefficient_count = 0;
  while (coefficient_count < orbital_preparation_input.n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [orbital_index *
                 orbital_preparation_input.n_basis_functions +
             coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

int get_physical_support_coefficient_count(
    const OrbitalPreparationInput& orbital_preparation_input,
    int orbital_index) {
  const int explicit_count =
      orbital_preparation_input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int coefficient_count = 0;
  while (coefficient_count < orbital_preparation_input.n_basis_functions) {
    const int basis_function_index =
        orbital_preparation_input.orbital_basis_index_table
            [orbital_index *
                 orbital_preparation_input.n_basis_functions +
             coefficient_count];
    if (basis_function_index == 0) {
      break;
    }
    ++coefficient_count;
  }
  return coefficient_count;
}

}  // namespace

void enforce_strict_sparse_orbital_support(
    OrbitalPreparationInput* orbital_preparation_input) {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument(
        "orbital_preparation_input must not be null while enforcing sparse support");
  }

  const int n_orbitals = orbital_preparation_input->n_orbitals;
  const int n_basis_functions = orbital_preparation_input->n_basis_functions;
  if (n_orbitals < 0 || n_basis_functions < 0) {
    throw std::invalid_argument(
        "orbital dimensions must be non-negative while enforcing sparse support");
  }

  const std::size_t expected_table_size =
      n_orbitals * n_basis_functions;
  if (orbital_preparation_input->orbital_value_table.size() != expected_table_size ||
      orbital_preparation_input->orbital_basis_index_table.size() != expected_table_size) {
    throw std::invalid_argument(
        "orbital tables do not match orbital dimensions while enforcing sparse support");
  }
  if (orbital_preparation_input->orbital_basis_counts.size() !=
      n_orbitals) {
    throw std::invalid_argument(
        "orbital_basis_counts size mismatch while enforcing sparse support");
  }

  // Only the leading explicit sparse slots define the orbital. Every padded
  // tail entry must remain blank so later orbital preparation and export code
  // cannot revive stale support-external coefficients.
  for (int orbital_index = 0; orbital_index < n_orbitals; ++orbital_index) {
    const int coefficient_count =
        get_physical_support_coefficient_count(
            *orbital_preparation_input,
            orbital_index);
    if (coefficient_count < 0 || coefficient_count > n_basis_functions) {
      throw std::runtime_error("invalid sparse orbital coefficient count");
    }

    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int basis_function_index =
          orbital_preparation_input->orbital_basis_index_table
              [orbital_index * n_basis_functions +
               coefficient_index];
      if (basis_function_index <= 0 ||
          basis_function_index > n_basis_functions) {
        throw std::runtime_error(
            "invalid sparse orbital basis index while enforcing sparse support");
      }
    }

    for (int coefficient_index = coefficient_count;
         coefficient_index < n_basis_functions;
         ++coefficient_index) {
      orbital_preparation_input->orbital_value_table
          [orbital_index * n_basis_functions +
           coefficient_index] = 0.0;
      orbital_preparation_input->orbital_basis_index_table
          [orbital_index * n_basis_functions +
           coefficient_index] = 0;
    }
  }
}

SparseOrbitalParameterView::SparseOrbitalParameterView(
    const OrbitalPreparationInput& orbital_preparation_input)
    : n_orbitals_(orbital_preparation_input.n_orbitals),
      n_basis_functions_(orbital_preparation_input.n_basis_functions),
      total_slot_count_(orbital_preparation_input.n_orbitals *
                        orbital_preparation_input.n_basis_functions),
      orbital_coefficient_counts_(orbital_preparation_input.n_orbitals, 0),
      flat_to_packed_index_(total_slot_count_, -1) {
  if (n_orbitals_ <= 0 || n_basis_functions_ <= 0) {
    throw std::invalid_argument(
        "SparseOrbitalParameterView requires positive orbital dimensions");
  }

  differentiable_parameter_indices_.reserve(
      total_slot_count_);
  // The legacy runtime stores each orbital in a fixed-width row of length
  // `n_basis_functions`, but only the leading explicit coefficients are
  // differentiable. Build a stable dense view once so all optimizers use the
  // same packing order.
  for (int orbital_index = 0; orbital_index < n_orbitals_; ++orbital_index) {
    const int coefficient_count =
        get_differentiable_coefficient_count(
            orbital_preparation_input,
            orbital_index);
    if (coefficient_count < 0 || coefficient_count > n_basis_functions_) {
      throw std::runtime_error("invalid sparse orbital coefficient count");
    }
    orbital_coefficient_counts_[orbital_index] = coefficient_count;
    for (int coefficient_index = 0;
         coefficient_index < coefficient_count;
         ++coefficient_index) {
      const int flat_index =
          orbital_index * n_basis_functions_ + coefficient_index;
      flat_to_packed_index_[flat_index] =
          static_cast<int>(differentiable_parameter_indices_.size());
      differentiable_parameter_indices_.push_back(flat_index);
    }
  }
}

int SparseOrbitalParameterView::orbital_coefficient_count(int orbital_index) const {
  if (orbital_index < 0 || orbital_index >= n_orbitals_) {
    throw std::out_of_range("orbital index is out of range");
  }
  return orbital_coefficient_counts_[orbital_index];
}

int SparseOrbitalParameterView::packed_index(
    int orbital_index,
    int coefficient_index) const {
  if (orbital_index < 0 || orbital_index >= n_orbitals_ ||
      coefficient_index < 0 ||
      coefficient_index >= orbital_coefficient_count(orbital_index)) {
    return -1;
  }
  const int flat_index = orbital_index * n_basis_functions_ + coefficient_index;
  return flat_to_packed_index_[flat_index];
}

Eigen::VectorXd SparseOrbitalParameterView::pack(
    const OrbitalPreparationInput& orbital_preparation_input) const {
  if (static_cast<int>(orbital_preparation_input.orbital_value_table.size()) !=
      total_slot_count_) {
    throw std::invalid_argument(
        "orbital_value_table size does not match packed sparse-orbital layout");
  }

  Eigen::VectorXd packed_parameters(
      static_cast<Eigen::Index>(differentiable_parameter_indices_.size()));
  // `packed_parameters` is the optimizer coordinate vector. Entry `k`
  // corresponds to one explicit sparse coefficient in `orbital_value_table`.
  for (Eigen::Index packed_offset = 0;
       packed_offset < packed_parameters.size();
       ++packed_offset) {
    const int flat_index =
        differentiable_parameter_indices_[packed_offset];
    packed_parameters[packed_offset] =
        orbital_preparation_input.orbital_value_table[flat_index];
  }
  return packed_parameters;
}

Eigen::VectorXd SparseOrbitalParameterView::gather_from_full(
    const std::vector<double>& full_vector) const {
  if (static_cast<int>(full_vector.size()) != total_slot_count_) {
    throw std::invalid_argument("full vector size does not match orbital_value_table");
  }

  Eigen::VectorXd packed_vector(
      static_cast<Eigen::Index>(differentiable_parameter_indices_.size()));
  // Gradients are often produced in the full padded layout; gather only the
  // explicit sparse coefficients into the same packed ordering used for
  // optimization variables.
  for (Eigen::Index packed_offset = 0;
       packed_offset < packed_vector.size();
       ++packed_offset) {
    const int flat_index =
        differentiable_parameter_indices_[packed_offset];
    packed_vector[packed_offset] = full_vector[flat_index];
  }
  return packed_vector;
}

void SparseOrbitalParameterView::unpack(
    const Eigen::VectorXd& packed_parameters,
    OrbitalPreparationInput* orbital_preparation_input) const {
  if (orbital_preparation_input == nullptr) {
    throw std::invalid_argument("orbital_preparation_input must not be null");
  }
  if (packed_parameters.size() !=
      static_cast<Eigen::Index>(differentiable_parameter_indices_.size())) {
    throw std::invalid_argument("packed parameter size does not match parameter view");
  }
  if (static_cast<int>(orbital_preparation_input->orbital_value_table.size()) !=
      total_slot_count_) {
    throw std::invalid_argument(
        "orbital_value_table size does not match packed sparse-orbital layout");
  }

  // Only overwrite the explicit sparse coefficients. Padding slots remain
  // untouched so the reconstructed legacy layout stays structurally identical
  // to the original input.
  for (Eigen::Index packed_offset = 0;
       packed_offset < packed_parameters.size();
       ++packed_offset) {
    const int flat_index =
        differentiable_parameter_indices_[packed_offset];
    orbital_preparation_input->orbital_value_table[flat_index] =
        packed_parameters[packed_offset];
  }
  enforce_strict_sparse_orbital_support(orbital_preparation_input);
}

}  // namespace xmvb::vb
