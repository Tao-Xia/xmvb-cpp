#include "vb/pdft/becke_partition.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vb/pdft/bragg_slater_radii.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb::pdft {

namespace {

// Small epsilon to avoid division by zero
constexpr double kEpsilon = 1.0e-12;
constexpr double kMaxSizeAdjustment = 0.5;

}  // anonymous namespace

double BeckePartition::cell_function(double mu) {
  // Becke's polynomial cell function:
  //   s(mu) = (3/2) * mu - (1/2) * mu^3  for |mu| <= 1
  //         = sign(mu)                    for |mu| > 1

  if (mu >= 1.0) {
    return 1.0;
  }
  if (mu <= -1.0) {
    return -1.0;
  }

  const double mu2 = mu * mu;
  return 1.5 * mu - 0.5 * mu * mu2;
}

double BeckePartition::iterated_cell_function(double mu, int k) {
  // Apply cell function k times: s_k(mu) = s(s(...s(mu)...))
  double result = mu;
  for (int i = 0; i < k; ++i) {
    result = cell_function(result);
  }
  return result;
}

double BeckePartition::size_adjustment_factor(
    int atomic_number_A,
    int atomic_number_B) {
  // Becke's heteronuclear correction is expressed in terms of the radius
  // ratio chi = R_A / R_B.  The additive shift parameter
  //
  //   a_AB = (1 - chi^2) / (4 chi)
  //
  // is clipped to [-1/2, 1/2] before entering
  //
  //   mu'_AB = mu_AB + a_AB (1 - mu_AB^2).
  const double radius_A = bragg_slater_radius_bohr(atomic_number_A);
  const double radius_B = bragg_slater_radius_bohr(atomic_number_B);
  const double chi_AB = radius_A / (radius_B + kEpsilon);
  const double adjustment =
      (1.0 - chi_AB * chi_AB) / (4.0 * chi_AB + kEpsilon);
  return std::clamp(
      adjustment,
      -kMaxSizeAdjustment,
      kMaxSizeAdjustment);
}

Eigen::MatrixXd BeckePartition::compute_weights(
    const Eigen::MatrixXd& grid_points,
    const Eigen::MatrixXd& atomic_coords,
    const Eigen::VectorXi& atomic_numbers,
    int exponent) {
  const int n_points = static_cast<int>(grid_points.rows());
  const int n_atoms = static_cast<int>(atomic_coords.rows());

  if (atomic_numbers.size() != n_atoms) {
    throw std::invalid_argument(
        "atomic_numbers size must match number of atoms");
  }

  if (grid_points.cols() != 3 || atomic_coords.cols() != 3) {
    throw std::invalid_argument(
        "grid_points and atomic_coords must have 3 columns");
  }

  // Partition weights: w_A(r) = P_A(r) / sum_B P_B(r)
  Eigen::MatrixXd weights(n_points, n_atoms);

  // Parallel loop over grid points
  #pragma omp parallel for schedule(static) if(n_points > 1000)
  for (int g = 0; g < n_points; ++g) {
    const Eigen::Vector3d r = grid_points.row(g);

    // Compute P_A(r) for each atom A
    // P_A(r) = prod_{B != A} [(1 - s_k(mu_AB(r))) / 2]
    Eigen::VectorXd P(n_atoms);

    for (int A = 0; A < n_atoms; ++A) {
      const Eigen::Vector3d R_A = atomic_coords.row(A);
      const double r_A = (r - R_A).norm();

      double product = 1.0;

      for (int B = 0; B < n_atoms; ++B) {
        if (B == A) continue;

        const Eigen::Vector3d R_B = atomic_coords.row(B);
        const double r_B = (r - R_B).norm();
        const double R_AB = (R_A - R_B).norm();

        // Confocal elliptical coordinate: mu_AB = (r_A - r_B) / R_AB
        double mu_AB = (r_A - r_B) / (R_AB + kEpsilon);

        // Apply atomic size adjustment
        const double nu_AB = size_adjustment_factor(
            atomic_numbers(A), atomic_numbers(B));
        mu_AB = mu_AB + nu_AB * (1.0 - mu_AB * mu_AB);

        // Apply iterated cell function
        const double s_k = iterated_cell_function(mu_AB, exponent);

        // Accumulate product: (1 - s_k) / 2
        product *= 0.5 * (1.0 - s_k);
      }

      P(A) = product;
    }

    // Normalize: w_A(r) = P_A(r) / sum_B P_B(r)
    const double sum_P = P.sum();

    if (sum_P > kEpsilon) {
      weights.row(g) = P / sum_P;
    } else {
      // Degenerate case: assign equal weights
      weights.row(g).setConstant(1.0 / n_atoms);
    }
  }

  return weights;
}

}  // namespace xmvb::vb::pdft
