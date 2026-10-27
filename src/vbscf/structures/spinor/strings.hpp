#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb::spinor {

class Sigma;

/**
 * @brief Complete fixed-electron occupation carrier for ordered A then B labels.
 *
 * Bit p denotes spinor p. Masks are ordered by increasing A occupation count,
 * then increasing A mask, then increasing B mask. A/B are storage groups, not
 * physical spin projections. The full carrier has C(n_spinor,n_electrons)
 * rows; max_states is an explicit allocation bound, not a truncation rule.
 */
class Strings {
  public:
    Strings(int n_spinor, int n_electrons, int n_a, std::size_t max_states);

    Eigen::Index dimension() const noexcept {
        return static_cast<Eigen::Index>(masks_.size());
    }
    int n_spinor() const noexcept {
        return n_spinor_;
    }
    int n_electrons() const noexcept {
        return n_electrons_;
    }
    int n_a() const noexcept {
        return n_a_;
    }
    const std::vector<std::uint64_t>& masks() const noexcept {
        return masks_;
    }

    /** @throws std::invalid_argument if mask is outside this carrier. */
    Eigen::Index index(std::uint64_t mask) const;

    /**
   * @brief Target row for a_created^dagger a_removed on source row.
   *
   * Returns -1 when the operation vanishes. On a valid connection, *sign is
   * +1 or -1, including +1 for an occupied diagonal substitution. Labels and
   * source row must be in range; sign must be non-null.
   */
    Eigen::Index substitution(Eigen::Index source, int created, int removed, int* sign) const;

    /**
   * @brief Apply sum_pq block(p,q) a_p^dagger a_q to every RHS column.
   *
   * matrix has (dimension(), block_width) shape and follows masks() row order.
   * block may be non-Hermitian and may connect A and B groups.
   */
    Eigen::MatrixXcd apply_one_body(const Eigen::Ref<const Eigen::MatrixXcd>& block,
                                    const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const;

  private:
    friend class Sigma;
    Eigen::MatrixXcd
    apply_one_body_unchecked(const Eigen::Ref<const Eigen::MatrixXcd>& block,
                             const Eigen::Ref<const Eigen::MatrixXcd>& matrix) const;

    int n_spinor_ = 0;
    int n_electrons_ = 0;
    int n_a_ = 0;
    std::vector<std::uint64_t> masks_;
    std::unordered_map<std::uint64_t, Eigen::Index> index_by_mask_;
};

} // namespace xmvb::vb::spinor
