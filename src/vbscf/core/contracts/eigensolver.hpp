#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "linalg/eigensolver.hpp"

namespace xmvb::vb {

/** @brief Structure-space generalized eigensolver used during optimization. */
enum class StructureEigensolver {
    Davidson,
    Dense,
};

/**
 * @brief Accuracy requested from the structure-space eigensolver.
 *
 * CI adapters with an exact metric or a physical residual bound use consecutive
 * energy change and the physical residual norm. A residual tolerance alone does not certify
 * orbital-gradient accuracy: small gaps can amplify orbital coupling.
 */
struct StructureSolveAccuracy {
    double energy_tolerance = 1.0e-7;
    double gradient_tolerance = 1.0e-3;

    /** @brief Validates both outer accuracy requirements. */
    void validate() const {
        if (!std::isfinite(energy_tolerance) || energy_tolerance <= 0.0 ||
            !std::isfinite(gradient_tolerance) || gradient_tolerance <= 0.0) {
            throw std::invalid_argument("structure solve accuracy tolerances must be positive");
        }
    }

    /** @brief CI first-order stationarity budget, in hartree.
   *
   * Apply the supplied outer energy budget to the linear CI residual as well
   * as the quadratic energy error; do not infer state-vector accuracy from
   * energy convergence or normalize this budget by the total Ritz energy.
   * The gradient budget can tighten, but never relax, this requirement.
   */
    double stationarity_tolerance() const {
        validate();
        return std::min(energy_tolerance, gradient_tolerance);
    }

    /** Use the same CI energy/residual policy on CPU and GPU when the
     * adapter supplies isometric coordinates or a physical dual-norm callback.
     * A spin-only or diagonal preconditioner does not establish this property.
     */
    void configure_physical_stopping(linalg::DavidsonOptions& options) const {
        validate();
        options.stopping = linalg::DavidsonStoppingCriterion::EnergyAndResidual;
        options.energy_tolerance = std::min(energy_tolerance, 1.0e-8);
        options.stationarity_tolerance = std::sqrt(options.energy_tolerance);
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
    double response_backward_error_tolerance(double selected_energy_scale) const {
        validate();
        if (!std::isfinite(selected_energy_scale) || selected_energy_scale < 0.0) {
            throw std::invalid_argument(
                "selected-state energy scale must be finite and nonnegative");
        }
        const double relative_energy_accuracy =
            energy_tolerance / std::max(1.0, selected_energy_scale);
        return std::min(gradient_tolerance, std::sqrt(relative_energy_accuracy));
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

} // namespace xmvb::vb
