#pragma once

#include <memory>
#include <vector>

#include "core/workspace.hpp"
#include "mcscf/ci/sigma.hpp"

namespace xmvb::mcscf {

/** An internal orthogonal CI gauge; physical orbitals are never changed.
 * Adjacent Givens rotations reuse small one-spin exchange neighborhoods.
 * Coefficients are transformed in place, without a compound matrix or a
 * second product array. This frame does not assert closure of selected VB.
 */
class CiOrbitalFrame {
  public:
    CiOrbitalFrame(const DirectCiSigmaAction& reference, const ActiveIntegrals& integrals);
    const ActiveIntegrals& integrals() const noexcept {
        return integrals_;
    }
    const DirectCiSigmaAction& action() const noexcept {
        return *action_;
    }
    const Eigen::MatrixXd& orbitals() const noexcept {
        return orbitals_;
    }

    /** C_old = wedge(U)_alpha C_new wedge(U)_beta^T.
     * inverse=true maps old coefficients to this frame; RHS blocks stay independent.
     */
    void transform(Eigen::Ref<Eigen::MatrixXd> coefficients, bool inverse) const;

  private:
    struct Pair {
        StringIndex first, second;
    };
    struct Rotation {
        int first;
        double cosine, sine;
    };
    using Neighborhoods = std::vector<std::vector<Pair>>;
    static Neighborhoods neighborhoods(const SpinConnections& graph);
    const SpinConnections* alpha_;
    const SpinConnections* beta_;
    Neighborhoods alpha_pairs_, beta_pairs_;
    std::vector<Rotation> rotations_;
    std::uint64_t negative_ = 0;
    Eigen::MatrixXd orbitals_;
    ActiveIntegrals integrals_;
    std::unique_ptr<DirectCiSigmaAction> action_;
    workspace::Temporary metadata_;
};

} // namespace xmvb::mcscf
