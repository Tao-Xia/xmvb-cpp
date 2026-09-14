#pragma once

#include <cmath>
#include <stdexcept>

namespace xmvb::vb {

/** @brief Structure-space generalized eigensolver used during optimization. */
enum class StructureEigensolver {
  Davidson,
  Dense,
};

/**
 * @brief Accuracy requested from the structure-space eigensolver.
 *
 * The energy threshold is converted to a relative eigen-equation backward
 * error with the current Ritz-value scale. The gradient threshold independently
 * limits that backward error. Keeping both requirements at the VBSCF boundary
 * prevents the core solver from inventing a machine-precision target.
 */
struct StructureSolveAccuracy {
  double energy_tolerance = 1.0e-7;
  double gradient_tolerance = 1.0e-3;

  /** @brief Validates both outer accuracy requirements. */
  void validate() const {
    if (!std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
        !std::isfinite(gradient_tolerance) || gradient_tolerance <= 0.0) {
      throw std::invalid_argument(
          "structure solve accuracy tolerances must be positive");
    }
  }
};

/** @brief Returns the stable input/output name of a structure eigensolver. */
inline const char* structure_eigensolver_name(StructureEigensolver solver) {
  switch (solver) {
    case StructureEigensolver::Davidson:
      return "davidson";
    case StructureEigensolver::Dense:
      return "dense";
  }
  return "unknown";
}

}  // namespace xmvb::vb
