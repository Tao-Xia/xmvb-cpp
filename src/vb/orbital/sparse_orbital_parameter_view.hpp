#pragma once

#include <vector>

#include <Eigen/Core>

#include "vb/orbital/orbital_preparation_input.hpp"

namespace xmvb::vb {

/**
 * @brief Packed view of the differentiable sparse-orbital coefficient table.
 *
 * The runtime stores orbital coefficients in the full legacy
 * `orbital_value_table` layout grouped by orbital and padded to
 * `n_basis_functions` entries per orbital. Optimizers only need the explicit
 * differentiable coefficients, so this class centralizes the packing and
 * unpacking logic instead of reimplementing it in every backend.
 */
class SparseOrbitalParameterView {
public:
  explicit SparseOrbitalParameterView(
      const OrbitalPreparationInput& orbital_preparation_input);

  int size() const noexcept {
    return static_cast<int>(differentiable_parameter_indices_.size());
  }

  int orbital_coefficient_count(int orbital_index) const;

  int packed_index(int orbital_index, int coefficient_index) const;

  Eigen::VectorXd pack(const OrbitalPreparationInput& orbital_preparation_input) const;

  Eigen::VectorXd gather_from_full(
      const std::vector<double>& full_vector) const;

  void unpack(
      const Eigen::VectorXd& packed_parameters,
      OrbitalPreparationInput* orbital_preparation_input) const;

  const std::vector<int>& differentiable_parameter_indices() const noexcept {
    return differentiable_parameter_indices_;
  }

private:
  int n_orbitals_ = 0;
  int n_basis_functions_ = 0;
  int total_slot_count_ = 0;
  std::vector<int> orbital_coefficient_counts_;
  std::vector<int> differentiable_parameter_indices_;
  std::vector<int> flat_to_packed_index_;
};

/**
 * @brief Clears coefficient storage that lies outside each orbital's sparse support.
 *
 * The padded orbital tables reserve `n_basis_functions` slots per orbital even
 * though only the leading sparse entries are meaningful. This helper removes
 * stale tail data so later dense reconstructions cannot accidentally interpret
 * padding as physical orbital support.
 */
void enforce_strict_sparse_orbital_support(
    OrbitalPreparationInput* orbital_preparation_input);

}  // namespace xmvb::vb
