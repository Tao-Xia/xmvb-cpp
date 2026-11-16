#pragma once

#include <Eigen/Core>

#include "mcscf/ci/links.hpp"

namespace xmvb::mcscf::detail {

/** Existing contiguous alpha extent, shared by standalone and fused sigma. */
inline constexpr int sigma_alpha_tile = 160;

/** One alpha tile's generator addresses, shared by all beta spectators.
 * Zero is exact absent support; otherwise +/- (source+1) encodes the borrowed
 * spin-link address and sign. With <=63 orbitals, even the largest complete
 * one-spin sector and its sentinel fit in signed StringIndex.
 * No coefficients or product-space data are retained.
 */
class AlphaChannelSources {
  public:
    void reset(const SpinConnections& graph,
               StringIndex first,
               StringIndex count,
               int pairs,
               bool off_diagonal = false) {
        off_diagonal_ = off_diagonal;
        sources_.setZero(count, pairs);
        for (StringIndex row = 0; row < count; ++row) {
            const auto target = first + row;
            if (!off_diagonal_)
                for (const int orbital : graph.occupied(target))
                    sources_(row, orbital * (orbital + 1) / 2 + orbital) = target + 1;
            for (const auto& link : graph.singles(target)) {
                const int high = std::max(link.created, link.annihilated);
                const int pair = off_diagonal_ ? link.pair - high : link.pair;
                sources_(row, pair) = link.sign > 0 ? link.source + 1 : -(link.source + 1);
            }
        }
    }

    /** Fill each channel once from its two signed-permutation sources. */
    void scatter(const SpinConnections& beta_graph,
                 const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                 StringIndex alpha_begin,
                 StringIndex alpha_count,
                 StringIndex beta_target,
                 Eigen::Ref<Eigen::MatrixXd> intermediate) const {
        const auto beta_size = beta_graph.n_strings();
        const auto width = coefficients.cols() / beta_size;
        const auto orbitals = beta_graph.orbitals;
        const auto occupied = beta_graph.basis.masks()[beta_target];
        int pair = 0;
        for (int high = 0; high < orbitals; ++high)
            for (int low = 0; low < high + !off_diagonal_; ++low, ++pair) {
                StringIndex beta_source = 0;
                if (high == low) {
                    if ((occupied >> high) & 1)
                        beta_source = beta_target + 1;
                } else if (((occupied >> high) ^ (occupied >> low)) & 1) {
                    const auto& link = beta_graph.singles(beta_target)[beta_graph.link_prefix(
                        beta_target, high, (std::uint64_t{1} << low) - 1)];
                    beta_source = link.sign > 0 ? link.source + 1 : -(link.source + 1);
                }
                const auto* alpha = column(pair);
                const auto assign = [&]<bool Beta>() {
                    const auto source = beta_source < 0 ? -beta_source : beta_source;
                    const double sign = beta_source < 0 ? -1.0 : 1.0;
                    for (int block = 0; block < width; ++block) {
                        const auto* beta_column =
                            Beta ? coefficients.col(block * beta_size + source - 1).data() +
                                       alpha_begin
                                 : nullptr;
                        const auto* alpha_column =
                            coefficients.col(block * beta_size + beta_target).data();
                        auto* output = intermediate.col(pair).data() + block * alpha_count;
#pragma omp simd
                        for (StringIndex row = 0; row < alpha_count; ++row) {
                            double value = Beta ? sign * beta_column[row] : 0.0;
                            const auto encoded = alpha[row];
                            if (encoded > 0)
                                value += alpha_column[encoded - 1];
                            else if (encoded < 0)
                                value -= alpha_column[-encoded - 1];
                            output[row] = value;
                        }
                    }
                };
                if (beta_source)
                    assign.template operator()<true>();
                else
                    assign.template operator()<false>();
            }
    }

    /** Adjoint alpha scatter, consuming each contiguous channel once.
     * A packed generator is a signed partial permutation: within one channel
     * each source has a single target, so SIMD lanes never share an output.
     * Different beta columns remain owned by their original sweep worker.
     */
    void gather(StringIndex count,
                const Eigen::Ref<const Eigen::MatrixXd>& transformed,
                Eigen::Ref<Eigen::MatrixXd> sigma,
                StringIndex first_column,
                StringIndex beta_stride,
                int width) const {
        for (int rhs = 0; rhs < width; ++rhs) {
            auto* output = sigma.col(first_column + rhs * beta_stride).data();
            for (Eigen::Index pair = 0; pair < sources_.cols(); ++pair) {
                const auto* address = column(pair);
                const auto* values = transformed.col(pair).data() + rhs * count;
#pragma omp simd
                for (StringIndex row = 0; row < count; ++row) {
                    const auto encoded = address[row];
                    if (encoded > 0)
                        output[encoded - 1] += values[row];
                    else if (encoded < 0)
                        output[-encoded - 1] -= values[row];
                }
            }
        }
    }

  private:
    bool off_diagonal_ = false;
    const StringIndex* column(int pair) const noexcept {
        return sources_.col(pair).data();
    }

    Eigen::Matrix<StringIndex, Eigen::Dynamic, Eigen::Dynamic> sources_;
};

} // namespace xmvb::mcscf::detail
