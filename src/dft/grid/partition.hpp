#pragma once

#include <Eigen/Core>

#include <vector>

namespace xmvb::dft {

// Original, equal-size Becke partition with three polynomial smoothings.
// Scratch arrays are caller-owned so each OpenMP worker needs O(n_atoms) space.
class BeckePartition {
  public:
    explicit BeckePartition(const Eigen::MatrixXd& centers);

    int n_atoms() const {
        return static_cast<int>(centers_.rows());
    }
    void weights(const Eigen::Vector3d& point,
                 std::vector<double>& distances,
                 std::vector<double>& factors) const;

  private:
    Eigen::MatrixXd centers_;
    Eigen::MatrixXd inverse_separations_;
};

} // namespace xmvb::dft
