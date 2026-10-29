#pragma once

#include "core/index.hpp"

#include <functional>
#include <Eigen/Core>

namespace xmvb::mcscf {

/** @brief Sampled orbital spectra; diagnostic only, never a stopping rule. */
struct CoarseSpectrum {
    Eigen::Index dimension = 0;
    bool positive_curvature = false;
    double curvature_min = 0.0, curvature_max = 0.0;
    double preconditioned_min = 0.0, preconditioned_max = 0.0;
};

/** @brief Sampled Schur curvature; never used to change the optimizer policy. */
struct SchurSpectrum {
    Eigen::Index orbital_dimension = 0, structure_dimension = 0;
    bool structure_positive = false, orbital_positive = false;
    double orbital_min = 0.0, orbital_max = 0.0;
    double structure_min = 0.0, structure_max = 0.0;
    double relaxed_min = 0.0, relaxed_max = 0.0;
    double maximum_fraction_removed = 0.0;
};

/** @brief Symmetric balancing inverse on a certified sampled coupled space.
 * Images and metric images must belong to the same frozen accepted point.
 * No operator evaluation or cross-point history is retained by this class.
 */
class CoarsePreconditioner {
  public:
    CoarsePreconditioner(Eigen::Index dimension,
                         int capacity,
                         double lindep,
                         double action_accuracy = 0.0);
    /** @brief Admit a rank-resolved sample only if both image projections certify symmetry. */
    void observe(const Eigen::VectorXd& direction,
                 const Eigen::VectorXd& image,
                 const Eigen::VectorXd& metric_image);
    /** @brief Apply signed range balancing with one base inverse call, including at shift poles. */
    Eigen::VectorXd
    apply(const Eigen::VectorXd& residual,
          double eigenvalue,
          const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inverse) const;
    int size() const noexcept {
        return size_;
    }
    /** @brief Compare a base inverse against cached orbital-only curvature.
   * Uses no new operator images and only a small sampled generalized spectrum.
   * The inverse callback receives full-size vectors with zero structure tail.
   */
    CoarseSpectrum
    orbital_spectrum(Eigen::Index orbital_dimension,
                     const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& inverse) const;
    /** @brief Eliminate cached structure samples without new operator actions.
   * Spectra use the sampled physical metric; no complete-space certificate
   * or complete structure-response inverse is inferred from these samples.
   */
    SchurSpectrum schur_spectrum(Eigen::Index orbital_dimension) const;

  private:
    int size_ = 0;
    double lindep_, action_accuracy_;
    Eigen::MatrixXd basis_, images_, metrics_;
    /** Certified symmetric projections of the accepted physical images. */
    Eigen::MatrixXd hessian_projection_, metric_projection_;
    /** Normalized original samples equal basis_ times this small QR factor. */
    Eigen::MatrixXd sample_factor_;
};

} // namespace xmvb::mcscf
