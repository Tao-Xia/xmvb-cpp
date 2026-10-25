#pragma once

#include <Eigen/Core>

#include <array>
#include <memory>
#include <span>

namespace xmvb::integrals {
struct LibcintInput;
class BasisEvaluator;
} // namespace xmvb::integrals

namespace xmvb::dft {
struct Grid;
class Functional;
class GridBasis;
class ResponseCache;

struct XcResult {
    double energy = 0.0;
    std::array<double, 2> electron_count{};
    std::array<Eigen::MatrixXd, 2> potential;
};

/** Integrates the restricted/spin-polarized semilocal XC part on an AO grid.
 * Global-hybrid exact exchange is added by KsPotential, not this integrator.
 * Input, grid, and functional must outlive the integrator and remain unmodified.
 * Owns one budgeted shared AO/reference workspace; calls are serialized by SCF.
 */
class XcIntegrator {
  public:
    XcIntegrator(const integrals::LibcintInput& input,
                 const Grid& grid,
                 const Functional& functional);
    ~XcIntegrator();
    XcIntegrator(const XcIntegrator&) = delete;
    XcIntegrator& operator=(const XcIntegrator&) = delete;

    XcResult evaluate(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density) const;

    /** Directional derivative of the AO XC potential at density. */
    std::array<Eigen::MatrixXd, 2>
    response(std::span<const Eigen::Ref<const Eigen::MatrixXd>> density,
             std::span<const Eigen::Ref<const Eigen::MatrixXd>> delta_density) const;

  private:
    const Grid& grid_;
    const Functional& functional_;
    std::unique_ptr<GridBasis> basis_;
    std::unique_ptr<ResponseCache> response_;
};

} // namespace xmvb::dft
