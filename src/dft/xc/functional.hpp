#pragma once

#include <Eigen/Core>

#include <memory>
#include <string>

namespace xmvb::dft {

struct XcValues {
    Eigen::VectorXd exc;    // Energy per electron.
    Eigen::MatrixXd vrho;   // Derivative of rho * exc with respect to spin density.
    Eigen::MatrixXd vsigma; // Derivative with respect to gradient dot products.
};

/** @brief Pointwise XC Hessian, packed in Libxc spin order.
 * Reference-only data, reusable for all directions at the same density.
 */
struct XcKernel {
    int n_spin = 1;
    Eigen::MatrixXd rho2, rho_sigma, sigma2;

    /** @brief Apply to signed directions without clipping them.
   * LDA ignores delta_sigma; GGA uses spin order (aa,ab,bb). Output matrices
   * must not alias the inputs or the retained kernel.
   */
    void apply(const Eigen::Ref<const Eigen::MatrixXd>& delta_rho,
               const Eigen::Ref<const Eigen::MatrixXd>& delta_sigma,
               Eigen::MatrixXd* delta_vrho,
               Eigen::MatrixXd* delta_vsigma) const;
};

/** A sum of Libxc LDA/GGA components, including global hybrids.
 * evaluate/kernel return only the semilocal part; the SCF model must add the
 * exact exchange reported by exact_exchange(). Range-separated and nonlocal
 * functionals are rejected, never silently reduced to their semilocal part.
 * rho is N x n_spin: total density for n_spin=1, alpha/beta for n_spin=2.
 * For GGA, sigma is N x 1 or N x 3 (alpha-alpha, alpha-beta, beta-beta).
 * For LDA, sigma may have zero or the corresponding GGA number of columns;
 * its values are ignored. Libxc's own low-density thresholds remain in effect.
 *
 * evaluate allocates, in addition to the XcValues results, bulk Libxc buffers
 * of N * (1 + 2*n_spin) doubles for LDA or
 * N * (1 + 2*n_spin + 2*n_sigma) doubles for GGA. It allocates no storage
 * per point or per component. n_sigma is 1 or 3 according to n_spin.
 */
class Functional {
  public:
    explicit Functional(const std::string& name = "LDA_X,LDA_C_VWN", int n_spin = 1);
    ~Functional();
    Functional(const Functional&) = delete;
    Functional& operator=(const Functional&) = delete;
    Functional(Functional&&) noexcept;
    Functional& operator=(Functional&&) noexcept;

    static std::string library_version();
    /** @brief Resolve common aliases to explicit Libxc component names. */
    static std::string canonical_name(const std::string& name);
    int n_spin() const;
    bool is_gga() const;
    /** @brief Global HF exchange fraction, summed over the selected components. */
    double exact_exchange() const;

    void evaluate(const Eigen::Ref<const Eigen::MatrixXd>& rho,
                  const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                  XcValues* out) const;

    /** @brief Build the reference-only analytical LDA/GGA response kernel. */
    void kernel(const Eigen::Ref<const Eigen::MatrixXd>& rho,
                const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                XcKernel* out) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::dft
