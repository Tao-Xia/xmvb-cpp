#include "vbscf/structures/orthogonal_ci/metric.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {

bool StructureMetricInverse::fits(
    const StructureAction& action,
    const xmvb::mcscf::StringTable& alpha_strings,
    const xmvb::mcscf::StringTable& beta_strings,
    std::size_t factor_byte_limit,
    std::size_t workspace_byte_limit) {
  if (!action.storage().orthogonal_direct_ci) return false;
  const std::size_t column_bytes =
      (alpha_strings.size() * beta_strings.size() +
       4 * static_cast<std::size_t>(action.n_structures_)) * sizeof(double);
  if (column_bytes == 0 || column_bytes > workspace_byte_limit) return false;
  return SpinMetric::fits(action, alpha_strings, beta_strings, factor_byte_limit);
}

StructureMetricInverse::StructureMetricInverse(
    const StructureAction& action,
    const xmvb::mcscf::StringTable& alpha_strings,
    const xmvb::mcscf::StringTable& beta_strings,
    bool complete_space,
    std::size_t factor_byte_limit,
    std::size_t workspace_byte_limit)
    : action_(&action), spin_(action, alpha_strings, beta_strings, factor_byte_limit) {
  if (!complete_space || !action.storage().orthogonal_direct_ci)
    throw std::invalid_argument("factorized metric inverse requires a complete spin sector");
  const std::size_t column_bytes =
      (alpha_strings.size() * beta_strings.size() +
       4 * static_cast<std::size_t>(action.n_structures_)) * sizeof(double);
  if (column_bytes == 0 || workspace_byte_limit < column_bytes)
    throw std::length_error("structure metric workspace cannot hold one carrier column");
  chunk_width_ = static_cast<int>(std::min<std::size_t>(
      workspace_byte_limit / column_bytes,
      std::numeric_limits<int>::max() / action.n_unique_beta_));
}

Eigen::MatrixXd StructureMetricInverse::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& rhs) const {
  if (rhs.rows() != action_->n_structures_ || rhs.cols() <= 0 || !rhs.allFinite())
    throw std::invalid_argument("structure metric inverse RHS is invalid");
  Eigen::MatrixXd result(rhs.rows(), rhs.cols());
  for (Eigen::Index first = 0; first < rhs.cols(); first += chunk_width_) {
    const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, rhs.cols() - first);
    Eigen::MatrixXd carrier = action_->expand_structure_block(spin_.solve(rhs.middleCols(first, count)));
    action_->inverse_adjoint_orbital_transform(&carrier);
    action_->inverse_orbital_transform(&carrier);
    result.middleCols(first, count) = spin_.solve(action_->contract_spin_product_block(carrier));
  }
  if (!result.allFinite())
    throw std::runtime_error("structure metric inverse produced nonfinite coefficients");
  return result;
}

Eigen::MatrixXd StructureMetricInverse::apply_carrier_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& rhs,
    const Eigen::Ref<const Eigen::VectorXd>& energies,double eigenvalue) const {
  if (rhs.rows()!=action_->n_structures_ || rhs.cols()<=0 ||
      rhs.cols()!=energies.size() || !rhs.allFinite())
    throw std::invalid_argument("invalid complete-sector carrier inverse RHS");
  Eigen::MatrixXd result(rhs.rows(),rhs.cols());
  for (Eigen::Index first=0;first<rhs.cols();first+=chunk_width_) {
    const Eigen::Index count=std::min<Eigen::Index>(chunk_width_,rhs.cols()-first);
    Eigen::MatrixXd carrier=action_->expand_structure_block(spin_.solve(rhs.middleCols(first,count)));
    action_->inverse_adjoint_orbital_transform(&carrier);
    action_->divide_carrier_diagonal(carrier,energies.segment(first,count),eigenvalue);
    action_->inverse_orbital_transform(&carrier);
    result.middleCols(first,count)=spin_.solve(action_->contract_spin_product_block(carrier));
  }
  return result;
}

Eigen::MatrixXd StructureMetricInverse::to_ci_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
  Eigen::MatrixXd result(coefficients.rows(),coefficients.cols());
  for (Eigen::Index first=0; first<coefficients.cols(); first+=chunk_width_) {
    const Eigen::Index count=std::min<Eigen::Index>(chunk_width_,coefficients.cols()-first);
    const auto carrier=action_->orthogonalize_structure_block(coefficients.middleCols(first,count));
    result.middleCols(first,count)=spin_.solve(
        action_->contract_spin_product_block(carrier),SpinMetric::Part::Lower);
  }
  return result;
}

Eigen::MatrixXd StructureMetricInverse::from_ci_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& coordinates) const {
  Eigen::MatrixXd result(coordinates.rows(),coordinates.cols());
  for (Eigen::Index first=0; first<coordinates.cols(); first+=chunk_width_) {
    const Eigen::Index count=std::min<Eigen::Index>(chunk_width_,coordinates.cols()-first);
    auto carrier=action_->expand_structure_block(
        spin_.solve(coordinates.middleCols(first,count),SpinMetric::Part::Upper));
    action_->inverse_orbital_transform(&carrier);
    result.middleCols(first,count)=spin_.solve(action_->contract_spin_product_block(carrier));
  }
  return result;
}

Eigen::MatrixXd StructureMetricInverse::ci_covector_to_structure(
    const Eigen::Ref<const Eigen::MatrixXd>& covector) const {
  Eigen::MatrixXd result(covector.rows(),covector.cols());
  for (Eigen::Index first=0; first<covector.cols(); first+=chunk_width_) {
    const Eigen::Index count=std::min<Eigen::Index>(chunk_width_,covector.cols()-first);
    auto carrier=action_->expand_structure_block(
        spin_.solve(covector.middleCols(first,count),SpinMetric::Part::Upper));
    action_->adjoint_orbital_transform(&carrier);
    result.middleCols(first,count)=action_->contract_spin_product_block(carrier);
  }
  return result;
}

Eigen::MatrixXd StructureMetricInverse::apply_ci_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& coordinates) const {
  Eigen::MatrixXd result(coordinates.rows(),coordinates.cols());
  // Sigma retains both its input and output carrier. Keep their combined
  // payload within the existing carrier-column allowance even for tiny G.
  const int width=std::max(1,chunk_width_/2);
  for (Eigen::Index first=0; first<coordinates.cols(); first+=width) {
    const Eigen::Index count=std::min<Eigen::Index>(width,coordinates.cols()-first);
    const auto carrier=action_->expand_structure_block(
        spin_.solve(coordinates.middleCols(first,count),SpinMetric::Part::Upper));
    const auto sigma=action_->apply_carrier_hamiltonian(carrier);
    result.middleCols(first,count)=spin_.solve(
        action_->contract_spin_product_block(sigma),SpinMetric::Part::Lower);
  }
  return result;
}

Eigen::MatrixXd StructureMetricInverse::apply_ci_preconditioner(
    const Eigen::Ref<const Eigen::MatrixXd>& covector,
    const Eigen::Ref<const Eigen::VectorXd>& energies,double eigenvalue) const {
  Eigen::MatrixXd result(covector.rows(),covector.cols());
  for (Eigen::Index first=0; first<covector.cols(); first+=chunk_width_) {
    const Eigen::Index count=std::min<Eigen::Index>(chunk_width_,covector.cols()-first);
    auto carrier=action_->expand_structure_block(
        spin_.solve(covector.middleCols(first,count),SpinMetric::Part::Upper));
    action_->divide_carrier_diagonal(carrier,energies.segment(first,count),eigenvalue);
    result.middleCols(first,count)=spin_.solve(
        action_->contract_spin_product_block(carrier),SpinMetric::Part::Lower);
  }
  return result;
}

Eigen::MatrixXd StructureMetricInverse::apply_at_transform(
    const Eigen::Ref<const Eigen::MatrixXd>& rhs,
    const ExteriorOrbitalTransform& alpha,
    const ExteriorOrbitalTransform& beta) const {
  if (rhs.rows() != action_->n_structures_ || rhs.cols() <= 0 || !rhs.allFinite())
    throw std::invalid_argument("trial metric inverse RHS is invalid");
  Eigen::MatrixXd result(rhs.rows(), rhs.cols());
  for (Eigen::Index first = 0; first < rhs.cols(); first += chunk_width_) {
    const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, rhs.cols() - first);
    Eigen::MatrixXd carrier = action_->expand_structure_block(spin_.solve(rhs.middleCols(first, count)));
    alpha.apply_inverse_adjoint_left(&carrier);
    for (Eigen::Index block = 0; block < count; ++block)
      beta.apply_inverse_adjoint_right_block(carrier.middleCols(block * beta.dimension(), beta.dimension()));
    alpha.apply_inverse_left(&carrier);
    for (Eigen::Index block = 0; block < count; ++block)
      beta.apply_inverse_right_block(carrier.middleCols(block * beta.dimension(), beta.dimension()));
    result.middleCols(first, count) = spin_.solve(action_->contract_spin_product_block(carrier));
  }
  return result;
}

void StructureMetricInverse::rebind(const StructureAction& action) {
  if (action.n_structures_ != action_->n_structures_ ||
      action.n_unique_alpha_ != action_->n_unique_alpha_ ||
      action.n_unique_beta_ != action_->n_unique_beta_)
    throw std::invalid_argument("moved structure metric action has different dimensions");
  action_ = &action;
}

}  // namespace xmvb::vb
