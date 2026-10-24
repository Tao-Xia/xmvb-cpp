#pragma once

#include <cstddef>

#include <Eigen/Core>

#include "vbscf/structures/spinor/basis.hpp"
#include "vbscf/structures/spinor/sigma.hpp"
#include "vbscf/structures/spinor/transform.hpp"

namespace xmvb::vb::spinor {

/** @brief H/S images of one selected-structure block. */
struct Images {
    Eigen::MatrixXcd hamiltonian;
    Eigen::MatrixXcd overlap;
};

/** @brief Exact selected-VB projection of a complete spinor carrier action.
 *
 * Owns B, the factorized wedge transform, and the orthogonal sigma. All three
 * must reference the same Strings object. The caller supplies a workspace
 * byte budget; an oversized block is rejected before carrier allocation.
 * No structure matrices or complete carrier pair matrices are assembled.
 */
class Action {
  public:
    Action(Basis basis, Transform transform, Sigma sigma, std::size_t workspace_bytes);
    int dimension() const noexcept {
        return basis_.size();
    }
    Images apply(const Eigen::Ref<const Eigen::MatrixXcd>& block) const;

  private:
    Basis basis_;
    Transform transform_;
    Sigma sigma_;
    std::size_t workspace_bytes_;
};

} // namespace xmvb::vb::spinor
