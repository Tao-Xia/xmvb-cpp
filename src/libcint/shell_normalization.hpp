#pragma once

#include <cmath>

namespace xmvb::vb {

/**
 * Shell-level normalization shared by Gaussian basis input and Molden output.
 * This is distinct from libcint's per-Cartesian-function normalization.
 */

inline int cartesian_ao_count(int angular_momentum) {
  return (angular_momentum + 1) * (angular_momentum + 2) / 2;
}

inline double factorial_as_double(int n) {
  double result = 1.0;
  for (int value = n; value > 0; --value) {
    result *= static_cast<double>(value);
  }
  return result;
}

inline double shell_normalization(
    int angular_momentum,
    double exponent) {
  constexpr double kPi = 3.141592653589793238462643383279;
  constexpr double kSphericalNormalizationS = 0.28209479177387814;
  constexpr double kSphericalNormalizationP = 0.4886025119029199;

  double normalization =
      std::pow(2.0, 2 * angular_momentum + 3) *
      std::pow(2.0 * exponent, angular_momentum + 1.5) /
      std::sqrt(kPi);
  normalization *= factorial_as_double(angular_momentum + 1);
  normalization /= factorial_as_double(2 * angular_momentum + 2);
  normalization = std::sqrt(normalization);
  if (angular_momentum == 0) {
    normalization *= kSphericalNormalizationS;
  } else if (angular_momentum == 1) {
    normalization *= kSphericalNormalizationP;
  }
  return normalization;
}

}  // namespace xmvb::vb
