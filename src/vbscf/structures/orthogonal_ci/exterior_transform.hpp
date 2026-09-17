#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

/**
 * @brief Exterior-power action of an upper-triangular orbital transform.
 *
 * If nonorthogonal orbitals satisfy `Phi = Chi R`, this object applies
 * `wedge^n R` to fixed-spin determinant coefficients without forming its
 * dense compound matrix. Completeness of the determinant space is mandatory.
 */
class ExteriorOrbitalTransform {
public:
  ExteriorOrbitalTransform(
      const std::vector<std::vector<int>>& determinants,
      const Eigen::Ref<const Eigen::MatrixXd>& upper_orbital_transform);

  /** @brief Applies the exterior transform to matrix rows in place. */
  void apply_left(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the exterior transform to matrix columns in place. */
  void apply_right(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the transpose exterior transform to matrix rows. */
  void apply_adjoint_left(Eigen::MatrixXd* coefficients) const;
  /** @brief Applies the transpose exterior transform to matrix columns. */
  void apply_adjoint_right(Eigen::MatrixXd* coefficients) const;

  int dimension() const noexcept { return static_cast<int>(masks_.size()); }
  int n_orbitals() const noexcept { return n_orbitals_; }
  int n_electrons() const noexcept { return n_electrons_; }
  std::size_t shear_pair_count() const noexcept;
  std::size_t dynamic_bytes() const noexcept;

private:
  struct DeterminantPair {
    int source = 0;
    int target = 0;
    double sign = 1.0;
  };

  struct Shear {
    double coefficient = 0.0;
    std::vector<DeterminantPair> pairs;
  };

  void validate_left(const Eigen::MatrixXd& coefficients) const;
  void validate_right(const Eigen::MatrixXd& coefficients) const;

  int n_orbitals_ = 0;
  int n_electrons_ = 0;
  std::vector<std::uint64_t> masks_;
  Eigen::VectorXd determinant_scales_;
  std::vector<Shear> shears_;
};

}  // namespace xmvb::vb
