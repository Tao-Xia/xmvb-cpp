#pragma once

#include <vector>

namespace xmvb::vb {

/**
 * @brief Non-owning active-space integral direction for one HVP application.
 *
 * The three buffers form one mathematical perturbation and must always be
 * propagated together. This view makes that coupling explicit without copying
 * the matrix-free workspaces owned by the HVP operator.
 */
struct ActiveSpaceIntegralDirectionView {
  const std::vector<double>& overlap;
  const std::vector<double>& one_electron;
  const std::vector<double>& packed_two_electron;
};

}  // namespace xmvb::vb
