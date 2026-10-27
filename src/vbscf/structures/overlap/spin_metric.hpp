#pragma once

#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "core/workspace.hpp"
#include "mcscf/ci/string_table.hpp"

namespace xmvb::vb {

class StructureAction;

/** @brief Orbital-independent spin Gram G=W^T W, factored by occupation.
 * Valid for any independent selection of VB structures with a definite spatial
 * occupation per structure; neither CAS completeness nor orbital closure is
 * assumed. Stores sum of squared occupation-block sizes, not a global Gram.
 */
class SpinMetric {
  public:
    enum class Part { Gram, Lower, Upper };

    SpinMetric(const StructureAction& action,
               const mcscf::StringTable& alpha,
               const mcscf::StringTable& beta,
               std::size_t byte_limit);

    /** @brief Exact G, L or L^T solve for G=L L^T. No orbital transform. */
    Eigen::MatrixXd solve(const Eigen::Ref<const Eigen::MatrixXd>& rhs,
                          Part part = Part::Gram) const;

    /** @brief SPD block-scaled inverse, matching the supplied diagonal sums.
   * One positive scalar per occupation preserves the full spin Gram geometry.
   * The supplied diagonal may be the existing positive Jacobi estimate of M.
   * This is only a preconditioner of the physical nonorthogonal metric M.
   * Its use never replaces M, changes the selected space or the solve target.
   */
    Eigen::VectorXd precondition(const Eigen::VectorXd& rhs,
                                 const Eigen::VectorXd& physical_diagonal) const;

    /** @brief Inverse of K G K, matching a supplied positive model diagonal.
   * K_ii=sqrt(diagonal_i/G_ii). Retains spin recoupling, without averaging
   * energy denominators or changing any physical H/M action.
   */
    Eigen::MatrixXd solve_diagonal_model(const Eigen::Ref<const Eigen::MatrixXd>& rhs,
                                         const Eigen::Ref<const Eigen::VectorXd>& diagonal) const;

    std::size_t factor_bytes() const noexcept {
        return factor_bytes_;
    }
    std::size_t block_count() const noexcept {
        return blocks_.size();
    }
    Eigen::Index max_block_size() const noexcept {
        return max_block_size_;
    }
    const Eigen::VectorXd& diagonal() const noexcept {
        return diagonal_;
    }

    /** @brief Factor residency only, never admission to the physical CI path. */
    static bool fits(const StructureAction& action,
                     const mcscf::StringTable& alpha,
                     const mcscf::StringTable& beta,
                     std::size_t byte_limit);

    /** @brief Occupation topology only; performs no numerical Gram contraction. */
    static std::vector<std::vector<Eigen::Index>> occupation_groups(const StructureAction& action,
                                                                    const mcscf::StringTable& alpha,
                                                                    const mcscf::StringTable& beta);

  private:
    struct Block {
        std::vector<Eigen::Index> indices;
        Eigen::MatrixXd lower;
    };
    Eigen::Index dimension_;
    /** Factor owners outlive CI actions and share the process memory allowance. */
    workspace::Reservation residency_;
    Eigen::VectorXd diagonal_;
    std::vector<Block> blocks_;
    std::size_t factor_bytes_ = 0;
    Eigen::Index max_block_size_ = 0;
};

} // namespace xmvb::vb
