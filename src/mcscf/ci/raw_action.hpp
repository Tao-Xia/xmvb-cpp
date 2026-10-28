#pragma once

#include <cstdint>

#include <cstddef>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "mcscf/ci/links.hpp"
#include "mcscf/ci/product_index.hpp"
#include "mcscf/ci/similarity.hpp"

namespace xmvb::mcscf {

/** @brief Scalar source work and vector channel work in the execution model. */
struct RawCiWork {
    long double source = 0, channel = 0;
    bool prefer_source() const noexcept {
        return source < channel;
    }
};

/** @brief Reverse seeds available before the linear Hamiltonian image. */
struct RawCiAdjoint {
    const Eigen::MatrixXd& seeds;
    CanonicalSeed value;
};

/** @brief Borrowed derivative lane; source columns can select a state of a block. */
struct RawCiDirection {
    const SimilarityHamiltonian* op = nullptr;
    const Eigen::MatrixXd* values = nullptr;
    std::int64_t first_column = 0;
    Eigen::MatrixXd& image;
    const Eigen::MatrixXd& seed;
    CanonicalSeed cotangent;
};

/** @brief Exact two-electron image of an arbitrary sparse spin-string source.
 * The immutable plan borrows complete single-spin connections. Source values
 * retain caller product order; beta-column CSR and target closure are built
 * once. Only requested source/response panels are numerical workspaces.
 * Connection owners and source topology must outlive this plan.
 */
class RawCiPlan {
  public:
    /** @brief Kernel operation model, independent of memory and input keywords.
     * Includes scalar canonical-element indexing and native BLAS packet work.
     */
    static RawCiWork
    work(const SpinConnections& alpha, const SpinConnections& beta, std::size_t source_products);
    RawCiPlan(const SpinConnections& alpha,
              const SpinConnections& beta,
              std::span<const ProductIndex> products);

    std::span<const StringIndex> beta_columns() const noexcept {
        return targets_;
    }
    std::span<const StringIndex> source_columns() const noexcept {
        return columns_;
    }
    std::size_t source_size() const noexcept {
        return products_.size();
    }
    bool complete_covalent_support() const noexcept {
        return covalent_;
    }
    bool has_source_column(StringIndex beta) const noexcept {
        return offsets_[beta] != offsets_[beta + 1];
    }
    std::size_t bytes() const noexcept;

    /** @brief Fill Hbar*X on requested beta columns, returning element evaluations.
     * Output column j*rhs+v contains beta[j], RHS v. All worker writes have
     * disjoint beta ownership; no atomics or per-thread large arrays are used.
     */
    std::size_t apply_columns(const SimilarityHamiltonian& hamiltonian,
                              const Eigen::Ref<const Eigen::MatrixXd>& values,
                              std::span<const StringIndex> beta,
                              Eigen::Ref<Eigen::MatrixXd> output,
                              const RawCiAdjoint* adjoint = nullptr,
                              std::span<const RawCiDirection> directions = {}) const;

    /** @brief Sparse original source times a single-spin panel, with exclusive row owners. */
    void multiply_source(const Eigen::Ref<const Eigen::VectorXd>& values,
                         const Eigen::Ref<const Eigen::MatrixXd>& input,
                         Eigen::Ref<Eigen::MatrixXd> output,
                         bool transpose = false) const;

    /** @brief Full oriented one-body moment <seed,E_pq source> on generated columns. */
    Eigen::MatrixXd one_body_moment(const Eigen::Ref<const Eigen::MatrixXd>& values,
                                    std::span<const StringIndex> beta,
                                    const Eigen::Ref<const Eigen::MatrixXd>& seeds) const;

    /** @brief Original sparse X columns, in the same layout as apply_columns. */
    void source_columns(const Eigen::Ref<const Eigen::MatrixXd>& values,
                        std::span<const StringIndex> beta,
                        Eigen::Ref<Eigen::MatrixXd> output) const;

  private:
    void validate(const Eigen::Ref<const Eigen::MatrixXd>& values,
                  std::span<const StringIndex> beta,
                  const Eigen::Ref<Eigen::MatrixXd>& output) const;
    template <class Function> std::size_t visit_column(StringIndex beta, Function&& consume) const;
    const SpinConnections& alpha_;
    const SpinConnections& beta_;
    std::span<const ProductIndex> products_;
    std::vector<std::size_t> offsets_, sources_, alpha_offsets_;
    std::vector<StringIndex> targets_, columns_;
    bool covalent_ = false;
};
} // namespace xmvb::mcscf
