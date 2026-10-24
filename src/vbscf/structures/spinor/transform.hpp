#pragma once

#include <complex>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "vbscf/structures/spinor/strings.hpp"

namespace xmvb::vb::spinor {

/** @brief Factorized wedge^N R action over the complete occupation carrier.
 *
 * R must be upper triangular with positive real diagonal, as supplied by the
 * computational Frame. Shears may cross A/B groups. No compound matrix or
 * per-shear carrier connection graph is retained.
 */
class Transform {
  public:
    Transform(std::shared_ptr<const Strings> strings,
              const Eigen::Ref<const Eigen::MatrixXcd>& upper);

    /** @brief In-place image transform of (carrier dimension, block width). */
    void apply(Eigen::Ref<Eigen::MatrixXcd> block) const;
    /** @brief In-place Hermitian adjoint, including conjugated shear factors. */
    void apply_adjoint(Eigen::Ref<Eigen::MatrixXcd> block) const;
    const std::shared_ptr<const Strings>& strings() const noexcept {
        return strings_;
    }

  private:
    struct Shear {
        int created;
        int removed;
        std::complex<double> coefficient;
    };
    void validate(const Eigen::Ref<const Eigen::MatrixXcd>& block) const;
    std::shared_ptr<const Strings> strings_;
    Eigen::VectorXcd scales_;
    std::vector<Shear> shears_;
};

} // namespace xmvb::vb::spinor
