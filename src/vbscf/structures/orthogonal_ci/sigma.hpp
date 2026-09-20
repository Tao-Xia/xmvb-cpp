#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/structures/orthogonal_ci/integrals.hpp"

namespace xmvb::vb {

/** @brief Integral adjoint of one orthonormal direct-CI bilinear form. */
struct DirectCiIntegralAdjoint {
  Eigen::MatrixXd one_electron;
  Eigen::MatrixXd pair_kernel;
};

/**
 * @brief Exact Slater--Condon sigma action in an orthonormal active basis.
 */
class DirectCiSigmaAction {
public:
  DirectCiSigmaAction(
      const std::vector<std::vector<int>>& alpha_determinants,
      const std::vector<std::vector<int>>& beta_determinants,
      const OrthogonalActiveIntegrals& integrals);

  /**
   * @brief Applies the Hamiltonian to packed determinant-product blocks.
   *
   * The input shape is `(n_alpha, block_width * n_beta)`; beta columns for one
   * block vector are contiguous.
   */
  Eigen::MatrixXd apply(
      const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const;

  /**
   * @brief Applies a general one-body orbital generator.
   *
   * The generator follows `sum_pq kappa(p,q) a_p^+ a_q` and need not be
   * symmetric. The coefficient packing is identical to `apply()`.
   */
  Eigen::MatrixXd apply_one_body_generator(
      const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
      const Eigen::Ref<const Eigen::MatrixXd>& generator) const;

  /**
   * @brief Differentiates `left.dot(H * right)` with respect to the integrals.
   *
   * Both returned matrices use the symmetric full-matrix Frobenius convention:
   * their inner products with symmetric integral directions equal the exact
   * directional derivative of the bilinear form. The traversal uses the same
   * Slater--Condon connection graph as `apply()`.
   */
  DirectCiIntegralAdjoint integral_adjoint(
      const Eigen::Ref<const Eigen::MatrixXd>& left,
      const Eigen::Ref<const Eigen::MatrixXd>& right) const;

  /**
   * @brief Unsymmetrized one-body adjoint for an orbital-generator action.
   *
   * The returned matrix `D` satisfies
   * `left.dot(E(kappa) * right) = D.cwiseProduct(kappa).sum()` for a general,
   * not necessarily symmetric one-body generator `kappa`.
   */
  Eigen::MatrixXd one_body_generator_adjoint(
      const Eigen::Ref<const Eigen::MatrixXd>& left,
      const Eigen::Ref<const Eigen::MatrixXd>& right) const;

  std::size_t dynamic_bytes() const noexcept;

private:
  struct PairKernelTerm {
    int first_pair = 0;
    int second_pair = 0;
    double coefficient = 0.0;
  };

  struct HamiltonianConnection {
    int source = 0;
    double value = 0.0;
    int density_pair = -1;
    double density_sign = 0.0;
    int one_electron_row = -1;
    int one_electron_column = -1;
    double one_electron_sign = 0.0;
    std::array<PairKernelTerm, 2> pair_terms{};
    int n_pair_terms = 0;
  };

  struct DensityConnections {
    std::vector<int> sources;
    std::vector<int> pairs;
    std::vector<double> signs;
    std::vector<int> created_orbitals;
    std::vector<int> annihilated_orbitals;

    std::size_t size() const noexcept { return sources.size(); }
    void reserve(std::size_t capacity);
    void append(
        int source,
        int pair,
        double sign,
        int created_orbital,
        int annihilated_orbital);
    std::size_t dynamic_bytes() const noexcept;
  };

  struct SpinConnections {
    std::vector<double> diagonal;
    std::vector<std::vector<HamiltonianConnection>> off_diagonal;
    std::vector<DensityConnections> singles;
    std::vector<std::vector<int>> occupied;
  };

  SpinConnections build_spin_connections(
      const std::vector<std::vector<int>>& determinants,
      const OrthogonalActiveIntegrals& integrals) const;
  Eigen::MatrixXd build_coulomb_diagonal(
      const std::vector<std::vector<int>>& determinants) const;
  const SpinConnections& beta_connections() const noexcept;
  const Eigen::MatrixXd& beta_coulomb_diagonal() const noexcept;

  int n_orbitals_ = 0;
  int n_alpha_ = 0;
  int n_beta_ = 0;
  Eigen::MatrixXd pair_kernel_;
  SpinConnections alpha_;
  std::optional<SpinConnections> distinct_beta_;
  Eigen::MatrixXd alpha_coulomb_diagonal_;
  std::optional<Eigen::MatrixXd> distinct_beta_coulomb_diagonal_;
};

}  // namespace xmvb::vb
