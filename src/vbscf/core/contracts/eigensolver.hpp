#pragma once

#include <algorithm>
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

  /**
   * @brief Backward-error tolerance for a first-order state response.
   *
   * A first-order eigensystem response enters a second-order orbital model.
   * Its residual is therefore limited by both the requested gradient accuracy
   * and the square root of the relative Ritz-energy accuracy.  The latter is
   * the first-order accuracy associated with a second-order energy target.
   *
   * @param selected_energy_scale Maximum magnitude of the selected Ritz
   *        values. Values below one do not tighten the relative energy scale.
   */
  double response_backward_error_tolerance(
      double selected_energy_scale) const {
    validate();
    if (!std::isfinite(selected_energy_scale) ||
        selected_energy_scale < 0.0) {
      throw std::invalid_argument(
          "selected-state energy scale must be finite and nonnegative");
    }
    const double relative_energy_accuracy =
        energy_tolerance / std::max(1.0, selected_energy_scale);
    return std::min(
        gradient_tolerance,
        std::sqrt(relative_energy_accuracy));
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
