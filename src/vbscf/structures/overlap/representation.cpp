#include "vbscf/structures/overlap/representation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include <Eigen/LU>

#include "core/timing.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {
namespace {

using Strings = mcscf::StringTable;
constexpr std::size_t kRowWorkspaceBytes = 8 * 1024 * 1024;

void validate_strings(const Strings& strings, int orbitals) {
  if (strings.empty()) throw std::invalid_argument("empty spin-string carrier");
  (void)strings.basis(orbitals); // validates occupations and uniqueness without decoding

}

void validate_chart(const Eigen::Ref<const Eigen::MatrixXd>& r, int n) {
  if (r.rows() != n || r.cols() != n || !r.allFinite() ||
      (r.diagonal().array() <= 0.0).any())
    throw std::invalid_argument("invalid orbital chart");
  if (n > 1) {
    const double scale = std::max(1.0, r.cwiseAbs().maxCoeff());
    if (r.template triangularView<Eigen::StrictlyLower>()
            .toDenseMatrix().cwiseAbs().maxCoeff() >
        64.0 * std::numeric_limits<double>::epsilon() * scale)
      throw std::invalid_argument("orbital chart must be upper triangular");
  }
}

void validate_direction(const Eigen::Ref<const Eigen::MatrixXd>& d, int n) {
  if (d.rows() != n || d.cols() != n || !d.allFinite())
    throw std::invalid_argument("invalid orbital chart direction");
  if (n > 1) {
    const double scale = std::max(1.0, d.cwiseAbs().maxCoeff());
    if (d.template triangularView<Eigen::StrictlyLower>()
            .toDenseMatrix().cwiseAbs().maxCoeff() >
        64.0 * std::numeric_limits<double>::epsilon() * scale)
      throw std::invalid_argument("chart direction must be upper triangular");
  }
}

Eigen::MatrixXd upper_part(const Eigen::MatrixXd& matrix) {
  Eigen::MatrixXd result = matrix;
  result.template triangularView<Eigen::StrictlyLower>().setZero();
  return result;
}

// The determinant of a deleted minor is defined also at singular K. This is
// essential: neither the cofactor nor its derivative uses K^{-1}.
double deleted_minor(const Eigen::MatrixXd& k, const std::vector<int>& rows,
                     const std::vector<int>& cols, int omit_row_1 = -1,
                     int omit_col_1 = -1, int omit_row_2 = -1,
                     int omit_col_2 = -1) {
  const int size = static_cast<int>(rows.size()) - (omit_row_1 >= 0) -
                   (omit_row_2 >= 0);
  if (size == 0) return 1.0;
  Eigen::MatrixXd minor(size, size);
  for (int i = 0, ii = 0; i < static_cast<int>(rows.size()); ++i) {
    if (i == omit_row_1 || i == omit_row_2) continue;
    for (int j = 0, jj = 0; j < static_cast<int>(cols.size()); ++j) {
      if (j == omit_col_1 || j == omit_col_2) continue;
      minor(ii, jj++) = k(rows[i], cols[j]);
    }
    ++ii;
  }
  return minor.partialPivLu().determinant();
}

double pair_value(const Eigen::MatrixXd& k, const std::vector<int>& left,
                  const std::vector<int>& right) {
  return deleted_minor(k, left, right);
}

double pair_direction(const Eigen::MatrixXd& k, const Eigen::MatrixXd& dk,
                      const std::vector<int>& left,
                      const std::vector<int>& right) {
  double result = 0.0;
  const int electrons = static_cast<int>(left.size());
  for (int i = 0; i < electrons; ++i)
    for (int j = 0; j < electrons; ++j) {
      const double sign = (i + j) % 2 ? -1.0 : 1.0;
      result += sign * deleted_minor(k, left, right, i, j) *
                dk(left[i], right[j]);
    }
  return result;
}

void add_pair_gradient(const Eigen::MatrixXd& k, const Eigen::MatrixXd* dk,
                       const std::vector<int>& left,
                       const std::vector<int>& right, double weight,
                       double dweight, Eigen::MatrixXd* gradient,
                       Eigen::MatrixXd* dgradient) {
  const int electrons = static_cast<int>(left.size());
  for (int i = 0; i < electrons; ++i)
    for (int j = 0; j < electrons; ++j) {
      const double sign = (i + j) % 2 ? -1.0 : 1.0;
      const double cofactor = sign * deleted_minor(k, left, right, i, j);
      (*gradient)(left[i], right[j]) += weight * cofactor;
      if (dgradient == nullptr) continue;
      double dcofactor = 0.0;
      for (int p = 0; p < electrons; ++p) {
        if (p == i) continue;
        for (int q = 0; q < electrons; ++q) {
          if (q == j) continue;
          const int reduced_p = p - (p > i);
          const int reduced_q = q - (q > j);
          const double second_sign = (reduced_p + reduced_q) % 2 ? -sign : sign;
          dcofactor += second_sign *
              deleted_minor(k, left, right, i, j, p, q) *
              (*dk)(left[p], right[q]);
        }
      }
      (*dgradient)(left[i], right[j]) +=
          dweight * cofactor + weight * dcofactor;
    }
}

Eigen::MatrixXd row_of(const StructureAction& action,
                       const Eigen::MatrixXd& coeff, int alpha, int beta_size) {
  return action.expand_structure_tile(coeff, alpha, 1, 0, beta_size);
}

Eigen::MatrixXd column_of(const StructureAction& action,
                          const Eigen::MatrixXd& coeff, int beta, int alpha_size) {
  return action.expand_structure_tile(coeff, 0, alpha_size, beta, 1);
}

Eigen::MatrixXd apply_pair_kernel(
    const StructureAction& action, const Strings& alpha, const Strings& beta,
    const Eigen::MatrixXd& kernel,
    const Eigen::Ref<const Eigen::MatrixXd>& coeff) {
  const int na = static_cast<int>(alpha.size());
  const int nb = static_cast<int>(beta.size());
  Eigen::MatrixXd result(action.n_structures(), coeff.cols());
  const std::size_t denominator = sizeof(double) *
      static_cast<std::size_t>(std::max(1, 2 * nb + 1));
  const int rows_per_tile = std::max(1, std::min(
      na, static_cast<int>(kRowWorkspaceBytes / denominator)));
  for (int col = 0; col < coeff.cols(); ++col) {
    Eigen::MatrixXd one_coeff = coeff.col(col);
    Eigen::MatrixXd one_result = Eigen::MatrixXd::Zero(action.n_structures(), 1);
    for (int begin = 0; begin < na; begin += rows_per_tile) {
      const int rows = std::min(rows_per_tile, na - begin);
      Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(rows, nb);
      for (int ar = 0; ar < na; ++ar) {
        const Eigen::MatrixXd source = row_of(action, one_coeff, ar, nb);
        for (int ai = 0; ai < rows; ++ai)
          projected.row(ai).noalias() +=
              pair_value(kernel, alpha[begin + ai], alpha[ar]) * source.row(0);
      }
      Eigen::MatrixXd tile(rows, nb);
      for (int bl = 0; bl < nb; ++bl) {
        Eigen::VectorXd overlap(nb);
        for (int br = 0; br < nb; ++br)
          overlap[br] = pair_value(kernel, beta[bl], beta[br]);
        tile.col(bl).noalias() = projected * overlap;
      }
      action.add_spin_product_tile(tile, begin, 0, &one_result);
    }
    result.col(col) = one_result;
  }
  return result;
}

}  // namespace

RepresentationOverlap::RepresentationOverlap(
    const StructureAction& action, const Strings& alpha_strings,
    const Strings& beta_strings, int n_active)
    : action_(&action), alpha_(alpha_strings), beta_(beta_strings),
      n_active_(n_active), complete_carrier_(action.supports_integral_direction()) {
  if (n_active <= 0 || (complete_carrier_ && n_active > 63))
    throw std::invalid_argument("invalid orbital count for representation overlap");
  validate_strings(alpha_, n_active);
  validate_strings(beta_, n_active);
  if (static_cast<int>(alpha_.size()) != action.n_alpha_strings() ||
      static_cast<int>(beta_.size()) != action.n_beta_strings())
    throw std::invalid_argument("spin strings do not match action carrier");
}

bool RepresentationOverlap::PreparedMetric::fits_selected(int dimension) noexcept {
  const std::size_t n=dimension>0 ? static_cast<std::size_t>(dimension):0;
  return n && n<=kRowWorkspaceBytes/(4*sizeof(double))/n;
}

RepresentationOverlap::PreparedMetric::PreparedMetric(
    const StructureAction& action, const Strings& alpha, const Strings& beta,
    bool complete_carrier, const Eigen::Ref<const Eigen::MatrixXd>& upper)
    : action_(&action), alpha_(alpha), beta_(beta) {
  if (complete_carrier) {
    alpha_transform_ =
        std::make_shared<ExteriorOrbitalTransform>(alpha, upper);
    if (&alpha == &beta || alpha == beta) {
      beta_transform_ = alpha_transform_;
    } else {
      beta_transform_ =
          std::make_shared<ExteriorOrbitalTransform>(beta, upper);
    }
    // M = W^T (T_alpha^T T_alpha tensor T_beta^T T_beta) W is
    // immutable at this chart. Reapplying its exterior maps in every CG
    // iteration is unnecessary for a small selected space. Reserve half the
    // existing budget for construction and quarters for identity/result.
    const std::size_t n=action.n_structures();
    if (fits_selected(action.n_structures())) {
      selected_metric_ = std::make_shared<SelectedMetric>(
          action.cross_overlap(Eigen::MatrixXd::Identity(n,n),
          *alpha_transform_,*beta_transform_,*alpha_transform_,*beta_transform_,
          nullptr,nullptr,nullptr,nullptr,kRowWorkspaceBytes/2));
      alpha_transform_.reset();
      beta_transform_.reset();
    }
  } else {
    kernel_ = upper.transpose() * upper;
  }
}

Eigen::MatrixXd RepresentationOverlap::PreparedMetric::operator()(
    const Eigen::Ref<const Eigen::MatrixXd>& coeff) const {
  if (coeff.rows() != action_->n_structures() || coeff.cols() <= 0 ||
      !coeff.allFinite()) throw std::invalid_argument("invalid structure block");
  if (selected_metric_) return selected_metric_->overlap * coeff;
  if (!alpha_transform_)
    return apply_pair_kernel(*action_, alpha_, beta_, kernel_, coeff);

  return action_->cross_overlap(coeff, *alpha_transform_, *beta_transform_,
                                *alpha_transform_, *beta_transform_);
}

const Eigen::MatrixXd& RepresentationOverlap::PreparedMetric::selected_matrix() const noexcept {
  static const Eigen::MatrixXd empty;
  return selected_metric_ ? selected_metric_->overlap : empty;
}

RepresentationOverlap::PreparedMetric RepresentationOverlap::prepare_metric(
    const Eigen::Ref<const Eigen::MatrixXd>& upper) const {
  validate_chart(upper, n_active_);
  return PreparedMetric(*action_, alpha_, beta_, complete_carrier_, upper);
}

Eigen::MatrixXd RepresentationOverlap::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& left_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& right_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& coeff) const {
  profiling::ScopedPhase phase(profiling::Phase::CiOverlapApply);
  validate_chart(left_upper, n_active_);
  validate_chart(right_upper, n_active_);
  if (coeff.rows() != action_->n_structures() || coeff.cols() <= 0 ||
      !coeff.allFinite()) throw std::invalid_argument("invalid structure block");
  Eigen::MatrixXd result(action_->n_structures(), coeff.cols());
  if (complete_carrier_) {
    ExteriorOrbitalTransform la(alpha_, left_upper), lb(beta_, left_upper);
    ExteriorOrbitalTransform ra(alpha_, right_upper), rb(beta_, right_upper);
    return action_->cross_overlap(coeff, la, lb, ra, rb);
  }
  return apply_pair_kernel(*action_, alpha_, beta_,
                           left_upper.transpose() * right_upper, coeff);
}

Eigen::MatrixXd RepresentationOverlap::apply_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& left_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& right_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_left_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_right_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& coeff) const {
  profiling::ScopedPhase phase(profiling::Phase::CiOverlapDirection);
  validate_chart(left_upper, n_active_);
  validate_chart(right_upper, n_active_);
  validate_direction(delta_left_upper, n_active_);
  validate_direction(delta_right_upper, n_active_);
  if (coeff.rows() != action_->n_structures() || coeff.cols() <= 0 ||
      !coeff.allFinite()) throw std::invalid_argument("invalid structure block");
  const int na = static_cast<int>(alpha_.size());
  const int nb = static_cast<int>(beta_.size());
  Eigen::MatrixXd result(action_->n_structures(), coeff.cols());
  if (complete_carrier_) {
    ExteriorOrbitalTransform la(alpha_, left_upper), lb(beta_, left_upper);
    ExteriorOrbitalTransform ra(alpha_, right_upper), rb(beta_, right_upper);
    const auto dla = la.direction(delta_left_upper);
    const auto dlb = lb.direction(delta_left_upper);
    const auto dra = ra.direction(delta_right_upper);
    const auto drb = rb.direction(delta_right_upper);
    return action_->cross_overlap(coeff, la, lb, ra, rb, &dla, &dlb, &dra, &drb);
  }
  const Eigen::MatrixXd k = left_upper.transpose() * right_upper;
  const Eigen::MatrixXd dk = delta_left_upper.transpose() * right_upper +
                             left_upper.transpose() * delta_right_upper;
  const std::size_t denominator = sizeof(double) *
      static_cast<std::size_t>(std::max(1, 4 * nb + 1));
  const int rows_per_tile = std::max(1, std::min(
      na, static_cast<int>(kRowWorkspaceBytes / denominator)));
  for (int col = 0; col < coeff.cols(); ++col) {
    Eigen::MatrixXd one_coeff = coeff.col(col);
    Eigen::MatrixXd one_result = Eigen::MatrixXd::Zero(action_->n_structures(), 1);
    for (int begin = 0; begin < na; begin += rows_per_tile) {
      const int rows = std::min(rows_per_tile, na - begin);
      Eigen::MatrixXd projected = Eigen::MatrixXd::Zero(rows, nb);
      Eigen::MatrixXd dprojected = Eigen::MatrixXd::Zero(rows, nb);
      for (int ar = 0; ar < na; ++ar) {
        const Eigen::MatrixXd source = row_of(*action_, one_coeff, ar, nb);
        for (int ai = 0; ai < rows; ++ai) {
          const auto& left = alpha_[begin + ai];
          const auto& right = alpha_[ar];
          projected.row(ai).noalias() += pair_value(k, left, right) * source.row(0);
          dprojected.row(ai).noalias() +=
              pair_direction(k, dk, left, right) * source.row(0);
        }
      }
      Eigen::MatrixXd tile(rows, nb);
      for (int bl = 0; bl < nb; ++bl) {
        Eigen::VectorXd overlap(nb), doverlap(nb);
        for (int br = 0; br < nb; ++br) {
          overlap[br] = pair_value(k, beta_[bl], beta_[br]);
          doverlap[br] = pair_direction(k, dk, beta_[bl], beta_[br]);
        }
        tile.col(bl).noalias() = dprojected * overlap + projected * doverlap;
      }
      action_->add_spin_product_tile(tile, begin, 0, &one_result);
    }
    result.col(col) = one_result;
  }
  return result;
}

UpperPair RepresentationOverlap::adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& left_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& right_upper,
    const Eigen::Ref<const Eigen::VectorXd>& left_coeff,
    const Eigen::Ref<const Eigen::VectorXd>& right_coeff, OverlapSide side) const {
  profiling::ScopedPhase phase(profiling::Phase::CiOverlapAdjoint);
  validate_chart(left_upper, n_active_);
  validate_chart(right_upper, n_active_);
  if (left_coeff.size() != action_->n_structures() ||
      right_coeff.size() != action_->n_structures() ||
      !left_coeff.allFinite() || !right_coeff.allFinite())
    throw std::invalid_argument("invalid structure coefficient vectors");
  const Eigen::MatrixXd left = left_coeff;
  const Eigen::MatrixXd right = right_coeff;
  const int na = static_cast<int>(alpha_.size());
  const int nb = static_cast<int>(beta_.size());
  if (complete_carrier_) {
    ExteriorOrbitalTransform la(alpha_, left_upper), lb(beta_, left_upper);
    ExteriorOrbitalTransform ra(alpha_, right_upper), rb(beta_, right_upper);
    return action_->cross_overlap_adjoint(left_coeff, right_coeff, la, lb, ra, rb,
        nullptr,nullptr,64ULL*1024ULL*1024ULL,side);
  }
  const Eigen::MatrixXd k = left_upper.transpose() * right_upper;
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(n_active_, n_active_);
  // Project one right-alpha row through the beta overlap, then contract all
  // left-alpha rows. This has no U_alpha^2 U_beta^2 traversal or pair cache.
  for (int ar = 0; ar < na; ++ar) {
    const Eigen::MatrixXd xr = row_of(*action_, right, ar, nb);
    Eigen::VectorXd projected = Eigen::VectorXd::Zero(nb);
    for (int bl = 0; bl < nb; ++bl)
      for (int br = 0; br < nb; ++br)
        projected[bl] += pair_value(k, beta_[bl], beta_[br]) * xr(0, br);
    for (int al = 0; al < na; ++al) {
      const Eigen::MatrixXd xl = row_of(*action_, left, al, nb);
      const double weight = xl.row(0).dot(projected);
      add_pair_gradient(k, nullptr, alpha_[al], alpha_[ar],
                        weight, 0.0, &g, nullptr);
    }
  }
  // The symmetric projection by alpha yields all beta-pair weights.
  for (int br = 0; br < nb; ++br) {
    const Eigen::MatrixXd xr = column_of(*action_, right, br, na);
    Eigen::VectorXd projected = Eigen::VectorXd::Zero(na);
    for (int al = 0; al < na; ++al)
      for (int ar = 0; ar < na; ++ar)
        projected[al] += pair_value(k, alpha_[al], alpha_[ar]) * xr(ar, 0);
    for (int bl = 0; bl < nb; ++bl) {
      const Eigen::MatrixXd xl = column_of(*action_, left, bl, na);
      const double weight = xl.col(0).dot(projected);
      add_pair_gradient(k, nullptr, beta_[bl], beta_[br],
                        weight, 0.0, &g, nullptr);
    }
  }
  return {side != OverlapSide::Right ? upper_part(right_upper * g.transpose()) : Eigen::MatrixXd{},
          side != OverlapSide::Left ? upper_part(left_upper * g) : Eigen::MatrixXd{}};
}

UpperPair RepresentationOverlap::adjoint_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& left_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& right_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_left_upper,
    const Eigen::Ref<const Eigen::MatrixXd>& delta_right_upper,
    const Eigen::Ref<const Eigen::VectorXd>& left_coeff,
    const Eigen::Ref<const Eigen::VectorXd>& right_coeff, OverlapSide side) const {
  profiling::ScopedPhase phase(profiling::Phase::CiOverlapAdjointDirection);
  validate_chart(left_upper, n_active_);
  validate_chart(right_upper, n_active_);
  validate_direction(delta_left_upper, n_active_);
  validate_direction(delta_right_upper, n_active_);
  if (left_coeff.size() != action_->n_structures() ||
      right_coeff.size() != action_->n_structures() ||
      !left_coeff.allFinite() || !right_coeff.allFinite())
    throw std::invalid_argument("invalid structure coefficient vectors");
  const Eigen::MatrixXd left = left_coeff;
  const Eigen::MatrixXd right = right_coeff;
  const int na = static_cast<int>(alpha_.size());
  const int nb = static_cast<int>(beta_.size());
  if (complete_carrier_) {
    ExteriorOrbitalTransform la(alpha_, left_upper), lb(beta_, left_upper);
    ExteriorOrbitalTransform ra(alpha_, right_upper), rb(beta_, right_upper);
    const Eigen::MatrixXd dl = delta_left_upper, dr = delta_right_upper;
    return action_->cross_overlap_adjoint(left_coeff, right_coeff, la, lb, ra, rb,
        &dl,&dr,64ULL*1024ULL*1024ULL,side);
  }
  const Eigen::MatrixXd k = left_upper.transpose() * right_upper;
  const Eigen::MatrixXd dk = delta_left_upper.transpose() * right_upper +
                             left_upper.transpose() * delta_right_upper;
  Eigen::MatrixXd g = Eigen::MatrixXd::Zero(n_active_, n_active_);
  Eigen::MatrixXd dg = Eigen::MatrixXd::Zero(n_active_, n_active_);
  for (int ar = 0; ar < na; ++ar) {
    const Eigen::MatrixXd xr = row_of(*action_, right, ar, nb);
    Eigen::VectorXd projected = Eigen::VectorXd::Zero(nb);
    Eigen::VectorXd dprojected = Eigen::VectorXd::Zero(nb);
    for (int bl = 0; bl < nb; ++bl)
      for (int br = 0; br < nb; ++br) {
        projected[bl] += pair_value(k, beta_[bl], beta_[br]) * xr(0, br);
        dprojected[bl] += pair_direction(k, dk, beta_[bl], beta_[br]) * xr(0, br);
      }
    for (int al = 0; al < na; ++al) {
      const Eigen::MatrixXd xl = row_of(*action_, left, al, nb);
      add_pair_gradient(k, &dk, alpha_[al], alpha_[ar],
                        xl.row(0).dot(projected),
                        xl.row(0).dot(dprojected), &g, &dg);
    }
  }
  for (int br = 0; br < nb; ++br) {
    const Eigen::MatrixXd xr = column_of(*action_, right, br, na);
    Eigen::VectorXd projected = Eigen::VectorXd::Zero(na);
    Eigen::VectorXd dprojected = Eigen::VectorXd::Zero(na);
    for (int al = 0; al < na; ++al)
      for (int ar = 0; ar < na; ++ar) {
        projected[al] += pair_value(k, alpha_[al], alpha_[ar]) * xr(ar, 0);
        dprojected[al] += pair_direction(k, dk, alpha_[al], alpha_[ar]) * xr(ar, 0);
      }
    for (int bl = 0; bl < nb; ++bl) {
      const Eigen::MatrixXd xl = column_of(*action_, left, bl, na);
      add_pair_gradient(k, &dk, beta_[bl], beta_[br],
                        xl.col(0).dot(projected),
                        xl.col(0).dot(dprojected), &g, &dg);
    }
  }
  return {side != OverlapSide::Right ? upper_part(delta_right_upper * g.transpose() +
                     right_upper * dg.transpose()) : Eigen::MatrixXd{},
          side != OverlapSide::Left ? upper_part(delta_left_upper * g + left_upper * dg) : Eigen::MatrixXd{}};
}

OverlapTransportGradient RepresentationOverlap::transport_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& current,
    const Eigen::Ref<const Eigen::MatrixXd>& origin,
    const Eigen::Ref<const Eigen::MatrixXd>& direction,
    const Eigen::Ref<const Eigen::VectorXd>& lambda,
    const Eigen::Ref<const Eigen::VectorXd>& base,
    const Eigen::Ref<const Eigen::VectorXd>& reference,
    const Eigen::Ref<const Eigen::VectorXd>& transported) const {
  validate_chart(current,n_active_); validate_chart(origin,n_active_);
  validate_direction(direction,n_active_);
  if (lambda.size()!=action_->n_structures() || base.size()!=lambda.size() ||
      reference.size()!=lambda.size() || transported.size()!=lambda.size() ||
      !lambda.allFinite() || !base.allFinite() || !reference.allFinite() || !transported.allFinite())
    throw std::invalid_argument("invalid finite transport coefficients");
  if (complete_carrier_) {
    ExteriorOrbitalTransform ca(alpha_,current),oa(alpha_,origin);
    std::optional<ExteriorOrbitalTransform> cb,ob;
    if (alpha_!=beta_) { cb.emplace(beta_,current); ob.emplace(beta_,origin); }
    return action_->cross_overlap_transport_gradient(lambda,base,reference,transported,
        ca,cb ? *cb:ca,oa,ob ? *ob:oa,direction);
  }
  const Eigen::MatrixXd zero=Eigen::MatrixXd::Zero(n_active_,n_active_);
  const auto a=adjoint(current,origin,lambda,base,OverlapSide::Left);
  const auto b=adjoint_direction(current,origin,zero,direction,lambda,reference,OverlapSide::Left);
  const auto m=adjoint(current,current,lambda,transported);
  return {{a.left+b.left-m.left-m.right,
      adjoint(current,origin,lambda,reference,OverlapSide::Right).right},
      apply(origin,current,lambda).col(0)};
}

}  // namespace xmvb::vb
