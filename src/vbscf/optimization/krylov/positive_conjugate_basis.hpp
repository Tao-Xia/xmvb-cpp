#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

// A-conjugate search directions for a fixed symmetric HVP operator. Store
// only directions with positive curvature. Their images have already been
// evaluated by the caller: restoring conjugacy consumes no additional HVP.
class PositiveConjugateBasis {
public:
  void append(const Eigen::VectorXd& direction, const Eigen::VectorXd& image) {
    const double norm = direction.stableNorm();
    if (direction.size() != image.size() || !direction.allFinite() ||
        !image.allFinite() || !(norm > 0.0) || !std::isfinite(norm) ||
        !(direction.dot(image) > 0.0)) {
      throw std::invalid_argument("invalid positive conjugate direction");
    }
    if (!directions_.empty() && direction.size() != directions_.front().size()) {
      throw std::invalid_argument("conjugate direction dimension mismatch");
    }
    directions_.push_back(direction / norm);
    images_.push_back(image / norm);
  }

  Eigen::VectorXd orthogonalize(Eigen::VectorXd direction) const {
    if (!direction.allFinite() ||
        (!directions_.empty() && direction.size() != directions_.front().size())) {
      throw std::invalid_argument("invalid conjugate-basis input");
    }
    // Full two-pass A-orthogonalization replaces the fragile three-term
    // recurrence. The normalized Euclidean HVP basis is still built separately.
    for (int pass = 0; pass < 2; ++pass) {
      for (std::size_t j = 0; j < directions_.size(); ++j) {
        const double denominator = directions_[j].dot(images_[j]);
        const double coefficient = direction.dot(images_[j]) / denominator;
        direction.noalias() -= coefficient * directions_[j];
      }
    }
    return direction;
  }

private:
  std::vector<Eigen::VectorXd> directions_;
  std::vector<Eigen::VectorXd> images_;
};

}  // namespace xmvb::vb
