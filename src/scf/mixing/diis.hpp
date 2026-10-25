#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::scf {

/** AO commutator F P S - S P F for symmetric F, P, and S. */
Eigen::MatrixXd commutator_error(const Eigen::Ref<const Eigen::MatrixXd>& fock,
                                 const Eigen::Ref<const Eigen::MatrixXd>& density,
                                 const Eigen::Ref<const Eigen::MatrixXd>& overlap);

/** Bounded DIIS history; Fock and residual storage may be rectangular. */
class Diis {
  public:
    explicit Diis(int max_history);

    void push(const Eigen::Ref<const Eigen::MatrixXd>& fock,
              const Eigen::Ref<const Eigen::MatrixXd>& error);

    bool extrapolate(Eigen::MatrixXd* fock, double* predicted_error_norm = nullptr) const;

    /** @brief Discard a density/Fock history after a nonlinear orbital step. */
    void clear() noexcept {
        history_.clear();
    }

  private:
    struct Entry {
        Eigen::MatrixXd fock;
        Eigen::MatrixXd error;
    };

    int max_history_;
    std::vector<Entry> history_;
};

} // namespace xmvb::scf
