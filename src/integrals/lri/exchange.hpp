#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <memory>
#include <vector>

namespace xmvb::integrals::lri {

struct AtomBasis {
    std::size_t orbitals;
    std::size_t auxiliaries;
};

// Canonical atom pair left <= right. Columns are i * n_right + j.
// Off-site products have one fit on each endpoint. On-site left_fit is the
// full symmetric product fit; right_fit is empty. No half-factor in this API.
struct PairFit {
    std::size_t left;
    std::size_t right;
    Eigen::MatrixXd left_fit;
    Eigen::MatrixXd right_fit;
};

// Canonical auxiliary Coulomb block; missing blocks represent exact zeros.
struct CoulombBlock {
    std::size_t left;
    std::size_t right;
    Eigen::MatrixXd values;
};

// Real, already folded Gamma/finite-system contraction K_ij = sum_kl (ik|jl) D_kl.
// This does not define a periodic Coulomb kernel or generate NAO integrals.
// Caller initializes MPI with MPI_THREAD_FUNNELED before construction and
// finalizes after destruction. MPI_COMM_SELF is used; calls are not concurrent.
// Signed symmetric D is allowed, including orbital-response densities.
class Exchange {
  public:
    Exchange(const std::vector<AtomBasis>& atoms,
             const std::vector<PairFit>& fits,
             const std::vector<CoulombBlock>& coulomb);
    ~Exchange();
    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;
    Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& density);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::integrals::lri
