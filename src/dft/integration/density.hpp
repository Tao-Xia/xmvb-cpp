#pragma once

#include <Eigen/Core>

#include <array>

namespace xmvb::dft {
struct BasisTile;

enum class DensityTileMode { Physical, Directional };

// Evaluate one spin channel on a shared AO tile. `projected` is reusable
// scratch and may be overwritten once all channels have been evaluated.
void evaluate_density_tile(const BasisTile& basis,
                           const Eigen::Ref<const Eigen::MatrixXd>& density_matrix,
                           bool with_gradient,
                           Eigen::Index spin,
                           Eigen::MatrixXd* projected,
                           Eigen::MatrixXd* rho,
                           std::array<Eigen::MatrixXd, 3>* density_gradient,
                           DensityTileMode mode = DensityTileMode::Physical);

} // namespace xmvb::dft
