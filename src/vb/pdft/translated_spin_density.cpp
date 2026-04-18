#include "vb/pdft/translated_spin_density.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb::pdft {

TranslatedSpinDensity compute_translated_spin_density(
    const Eigen::VectorXd& rho,
    const Eigen::VectorXd& pi,
    double density_threshold) {
  const int n_points = static_cast<int>(rho.size());

  if (pi.size() != n_points) {
    throw std::invalid_argument("rho and pi must have the same size");
  }

  if (density_threshold <= 0.0) {
    throw std::invalid_argument("density_threshold must be positive");
  }

  TranslatedSpinDensity result;
  result.on_top_ratio.resize(n_points);
  result.zeta_translated.resize(n_points);
  result.rho_alpha.resize(n_points);
  result.rho_beta.resize(n_points);

  // Compute translated spin densities point-by-point
  #pragma omp parallel for schedule(static) if(n_points > 1000)
  for (int g = 0; g < n_points; ++g) {
    const double rho_g = rho(g);
    const double pi_g = pi(g);

    // Apply density threshold for numerical stability
    if (rho_g < density_threshold) {
      // Very low density: set everything to zero
      result.on_top_ratio(g) = 0.0;
      result.zeta_translated(g) = 0.0;
      result.rho_alpha(g) = 0.0;
      result.rho_beta(g) = 0.0;
      continue;
    }

    // Compute on-top ratio: R = 4 * Pi / rho^2
    const double rho2 = rho_g * rho_g;
    double R = 4.0 * pi_g / rho2;

    // Clamp R to [0, 1] for numerical stability
    // R > 1 is unphysical (would give imaginary zeta_t)
    // R < 0 can occur due to numerical noise
    R = std::max(0.0, std::min(1.0, R));

    result.on_top_ratio(g) = R;

    // Compute translated spin polarization: zeta_t = sqrt(1 - R)
    const double zeta_t = std::sqrt(1.0 - R);
    result.zeta_translated(g) = zeta_t;

    // Compute translated spin densities
    // rho_alpha = rho * (1 + zeta_t) / 2
    // rho_beta = rho * (1 - zeta_t) / 2
    result.rho_alpha(g) = 0.5 * rho_g * (1.0 + zeta_t);
    result.rho_beta(g) = 0.5 * rho_g * (1.0 - zeta_t);

    // Ensure non-negativity (should be guaranteed by construction, but check anyway)
    result.rho_alpha(g) = std::max(0.0, result.rho_alpha(g));
    result.rho_beta(g) = std::max(0.0, result.rho_beta(g));
  }

  return result;
}

}  // namespace xmvb::vb::pdft
