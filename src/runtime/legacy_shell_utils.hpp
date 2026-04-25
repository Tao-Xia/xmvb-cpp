#pragma once

#include <cmath>

namespace xmvb::vb {

/**
 * The standalone C++ path still needs the historical XMVB shell-level
 * normalization used when reading Gaussian basis coefficients and when writing
 * Molden files back out. These helpers intentionally mirror that shell-level
 * convention, not libcint's per-Cartesian-function normalization.
 */

inline int cartesian_ao_count(int angular_momentum) {
  return (angular_momentum + 1) * (angular_momentum + 2) / 2;
}

inline double legacy_factorial(int n) {
  double result = 1.0;
  for (int value = n; value > 0; --value) {
    result *= static_cast<double>(value);
  }
  return result;
}

inline double legacy_shell_normalization(
    int angular_momentum,
    double exponent) {
  constexpr double kPi = 3.141592653589793238462643383279;
  constexpr double kLegacyNormS = 0.28209479177387814;
  constexpr double kLegacyNormP = 0.4886025119029199;

  double normalization =
      std::pow(2.0, 2 * angular_momentum + 3) *
      std::pow(2.0 * exponent, angular_momentum + 1.5) /
      std::sqrt(kPi);
  normalization *= legacy_factorial(angular_momentum + 1);
  normalization /= legacy_factorial(2 * angular_momentum + 2);
  normalization = std::sqrt(normalization);
  if (angular_momentum == 0) {
    normalization *= kLegacyNormS;
  } else if (angular_momentum == 1) {
    normalization *= kLegacyNormP;
  }
  return normalization;
}

}  // namespace xmvb::vb
