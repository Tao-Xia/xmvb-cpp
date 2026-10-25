#pragma once

#include <Eigen/Core>

#include <vector>

#include "integrals/ao/input.hpp"
#include "vbscf/integrals/ao/one_electron/backpropagation_result.hpp"
#include "integrals/ao/ri.hpp"
#include "integrals/response/ri.hpp"
#include "vbscf/integrals/ao/one_electron/result.hpp"

namespace xmvb::vb {

/**
 * @brief Reverse-mode backpropagator for `AoEffectiveOneElectronBuilder`.
 *
 * This class propagates the gradient of the AO effective one-electron matrix
 * `F11` back to the inactive density matrix `P11`.
 */
class AoEffectiveOneElectronBackpropagator {
  public:
    /**
   * @brief Backpropagates through the AO-side RI inactive-density contraction.
   */
    AoEffectiveOneElectronBackpropagationResult
    backpropagate(const std::vector<double>& ao_effective_one_electron_gradient,
                  const xmvb::integrals::RiAoFactorization& ri_factorization,
                  int n_basis_functions) const;

    AoEffectiveOneElectronBackpropagationResult
    backpropagate(const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
                  const xmvb::integrals::RiAoFactorization& ri_factorization,
                  int n_basis_functions) const;

    /**
   * @brief Pull back an active-H1E adjoint using accepted one-index factors.
   * With a reference, reuse its forward G(P_i) for the reference-energy term
   * 2 G(P_i). Without it, return only G(C (g+g^T) C^T), as required by
   * orbital--structure coupling. No occupied transform is repeated.
   */
    AoEffectiveOneElectronBackpropagationResult
    backpropagate(const std::vector<double>& active_one_electron_gradient,
                  const integrals::RiActiveTwoElectronResponseCache& cache,
                  const AoEffectiveOneElectronResult* reference = nullptr) const;

    /**
   * @brief Backpropagates through the canonical exact AO operator.
   */
    AoEffectiveOneElectronBackpropagationResult
    backpropagate(const std::vector<double>& ao_effective_one_electron_gradient,
                  const xmvb::integrals::AoIntegralInput& ao_integral_input) const;

    AoEffectiveOneElectronBackpropagationResult
    backpropagate(const Eigen::Ref<const Eigen::MatrixXd>& ao_effective_one_electron_gradient,
                  const xmvb::integrals::AoIntegralInput& ao_integral_input) const;
};

} // namespace xmvb::vb
