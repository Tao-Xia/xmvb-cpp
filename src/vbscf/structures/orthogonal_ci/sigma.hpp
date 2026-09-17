#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "vbscf/structures/orthogonal_ci/integrals.hpp"

namespace xmvb::vb {

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

  std::size_t dynamic_bytes() const noexcept;

private:
  struct HamiltonianConnection {
    int source = 0;
    double value = 0.0;
    int density_pair = -1;
    double density_sign = 0.0;
  };

  struct DensityConnection {
    int source = 0;
    int pair = 0;
    double sign = 1.0;
  };

  struct SpinConnections {
    std::vector<double> diagonal;
    std::vector<std::vector<HamiltonianConnection>> off_diagonal;
    std::vector<std::vector<DensityConnection>> singles;
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
