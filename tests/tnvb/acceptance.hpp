#pragma once

#include <cmath>
#include <stdexcept>

namespace tnvb_test {

struct ComparisonErrors {
  double energy, h, s, coefficient, weight, gradient;
};

struct AcceptanceProfile {
  const char* name;
  double energy, coefficient, weight, gradient;
  bool matrices_required;

  bool passed(const ComparisonErrors& e) const {
    return std::isfinite(e.energy+e.h+e.s+e.coefficient+e.weight+e.gradient) &&
        e.energy<=energy && e.coefficient<=coefficient && e.weight<=weight &&
        e.gradient<=gradient && (!matrices_required || strict_matrices_passed(e));
  }

  static bool strict_matrices_passed(const ComparisonErrors& e) {
    return std::isfinite(e.h+e.s) && e.h<=1e-8 && e.s<=1e-10;
  }
};

inline AcceptanceProfile acceptance_profile(int active_orbitals) {
  if (active_orbitals==6) return {"benzene",1e-9,1e-8,1e-8,1e-7,true};
  // User-confirmed C14 policy: energy/gradient/coefficients/weights are gates;
  // finite-D matrix discrepancies remain explicitly reported diagnostics.
  if (active_orbitals==14) return {"C14",1e-6,1e-4,1e-4,1e-7,false};
  throw std::invalid_argument("acceptance profiles are restricted to benzene and C14");
}

}  // namespace tnvb_test
