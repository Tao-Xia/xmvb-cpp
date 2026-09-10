#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

// Accepted-point Euclidean coordinates only. Orthogonalize the direction
// BEFORE applying H: subtracting nearly equal HVPs and dividing by a small
// orthogonalization residual can destroy the cached second-order model.
// On success, cache (q, Hq) and reconstruct H*candidate by linearity.
// A dependent candidate is not admitted and does not consume an HVP.
template <typename HvpAction>
bool append_orthonormal_hvp_direction(
    const Eigen::VectorXd& candidate,
    HvpAction&& apply_hvp,
    std::vector<Eigen::VectorXd>* basis,
    std::vector<Eigen::VectorXd>* images,
    Eigen::VectorXd* candidate_image) {
  if (basis == nullptr || images == nullptr || candidate_image == nullptr ||
      basis->size() != images->size()) {
    throw std::invalid_argument("inconsistent orthonormal HVP storage");
  }
  if (candidate.size() == 0 || !candidate.allFinite()) return false;
  const double candidate_norm = candidate.stableNorm();
  if (!(candidate_norm > 0.0) || !std::isfinite(candidate_norm)) return false;

  // Scale first so large search amplitudes do not enter the projection sums.
  Eigen::VectorXd q = candidate / candidate_norm;
  Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(basis->size());
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t j = 0; j < basis->size(); ++j) {
      if ((*basis)[j].size() != candidate.size() ||
          (*images)[j].size() != candidate.size()) {
        throw std::invalid_argument("orthonormal HVP dimension mismatch");
      }
      const double coefficient = (*basis)[j].dot(q);
      coefficients[j] += coefficient;
      q.noalias() -= coefficient * (*basis)[j];
    }
  }
  const double orthogonal_norm = q.stableNorm();
  if (!(orthogonal_norm > std::sqrt(std::numeric_limits<double>::epsilon())) ||
      !std::isfinite(orthogonal_norm)) {
    return false;
  }
  q /= orthogonal_norm;
  Eigen::VectorXd hq = apply_hvp(q);
  if (hq.size() != candidate.size() || !hq.allFinite()) {
    throw std::runtime_error("invalid normalized-direction HVP");
  }

  Eigen::VectorXd reconstructed_image = orthogonal_norm * hq;
  for (std::size_t j = 0; j < basis->size(); ++j) {
    reconstructed_image.noalias() += coefficients[j] * (*images)[j];
  }
  reconstructed_image *= candidate_norm;
  if (!reconstructed_image.allFinite()) {
    throw std::runtime_error("non-finite reconstructed search-direction HVP");
  }
  basis->push_back(std::move(q));
  images->push_back(std::move(hq));
  *candidate_image = std::move(reconstructed_image);
  return true;
}

// Orthogonalize a set of candidates first, then evaluate all admitted
// normalized directions with one block HVP.  This is deliberately not a loop
// over append_orthonormal_hvp_direction: the accepted-point operator can share
// AO/integral response work only when the direction dimension is visible at
// the call boundary.
template <typename BlockHvpAction>
int append_orthonormal_hvp_block(
    const Eigen::Ref<const Eigen::MatrixXd>& candidates,
    BlockHvpAction&& apply_hvp_block,
    std::vector<Eigen::VectorXd>* basis,
    std::vector<Eigen::VectorXd>* images) {
  if (basis == nullptr || images == nullptr || basis->size() != images->size()) {
    throw std::invalid_argument("inconsistent orthonormal block-HVP storage");
  }
  if (candidates.rows() == 0 || candidates.cols() == 0) return 0;

  std::vector<Eigen::VectorXd> admitted;
  admitted.reserve(static_cast<std::size_t>(candidates.cols()));
  for (Eigen::Index column = 0; column < candidates.cols(); ++column) {
    const Eigen::VectorXd candidate = candidates.col(column);
    if (!candidate.allFinite()) continue;
    const double candidate_norm = candidate.stableNorm();
    if (!(candidate_norm > 0.0) || !std::isfinite(candidate_norm)) continue;

    Eigen::VectorXd q = candidate / candidate_norm;
    for (int pass = 0; pass < 2; ++pass) {
      for (const Eigen::VectorXd& existing : *basis) {
        if (existing.size() != candidates.rows()) {
          throw std::invalid_argument("orthonormal block-HVP dimension mismatch");
        }
        q.noalias() -= existing.dot(q) * existing;
      }
      for (const Eigen::VectorXd& pending : admitted) {
        q.noalias() -= pending.dot(q) * pending;
      }
    }
    const double orthogonal_norm = q.stableNorm();
    if (!(orthogonal_norm > std::sqrt(std::numeric_limits<double>::epsilon())) ||
        !std::isfinite(orthogonal_norm)) {
      continue;
    }
    admitted.push_back(q / orthogonal_norm);
  }
  if (admitted.empty()) return 0;

  Eigen::MatrixXd block(candidates.rows(), admitted.size());
  for (std::size_t column = 0; column < admitted.size(); ++column) {
    block.col(static_cast<Eigen::Index>(column)) = admitted[column];
  }
  Eigen::MatrixXd block_images = apply_hvp_block(block);
  if (block_images.rows() != block.rows() ||
      block_images.cols() != block.cols() ||
      !block_images.allFinite()) {
    throw std::runtime_error("invalid normalized-direction block HVP");
  }
  for (Eigen::Index column = 0; column < block.cols(); ++column) {
    basis->push_back(block.col(column));
    images->push_back(block_images.col(column));
  }
  return static_cast<int>(admitted.size());
}

}  // namespace xmvb::vb
