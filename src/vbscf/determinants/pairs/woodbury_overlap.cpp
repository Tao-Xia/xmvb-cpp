#include "vbscf/determinants/pairs/woodbury_overlap.hpp"

#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/LU>

#include "vbscf/determinants/pairs/contractions.hpp"

namespace xmvb::vb {
namespace {

double infinity_norm(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return matrix.size() == 0
      ? 0.0
      : matrix.cwiseAbs().rowwise().sum().maxCoeff();
}

std::vector<int> changed_rows(
    const std::vector<int>& old_string,
    const std::vector<int>& new_string) {
  std::vector<int> rows;
  rows.reserve(old_string.size());
  for (int row = 0; row < static_cast<int>(old_string.size()); ++row) {
    if (old_string[row] != new_string[row]) {
      rows.push_back(row);
    }
  }
  return rows;
}

std::vector<int> changed_columns(
    const std::vector<int>& old_string,
    const std::vector<int>& new_string) {
  return changed_rows(old_string, new_string);
}

std::optional<DeterminantOverlapResult> make_updated_result(
    Eigen::MatrixXd new_overlap,
    Eigen::MatrixXd inverse_new,
    double middle_determinant,
    const DeterminantOverlapResult& old_result) {
  if (!is_certified_regular_overlap(new_overlap, inverse_new)) {
    return std::nullopt;
  }
  const double determinant =
      old_result.overlap_determinant * middle_determinant;
  if (middle_determinant == 0.0 || !std::isfinite(middle_determinant) ||
      determinant == 0.0 || !std::isfinite(determinant)) {
    return std::nullopt;
  }

  DeterminantOverlapResult result;
  result.n_electrons = old_result.n_electrons;
  result.overlap_submatrix = std::move(new_overlap);
  result.overlap_determinant = determinant;
  result.determinant_sign = std::signbit(determinant) ? -1.0 : 1.0;
  result.log_abs_determinant =
      old_result.log_abs_determinant + std::log(std::abs(middle_determinant));
  result.nullity = 0;
  result.inverse_overlap_submatrix = std::move(inverse_new);
  return result;
}

}  // namespace

double regular_overlap_condition_limit() noexcept {
  constexpr double highest_inverse_power = 4.0;
  return std::pow(
      std::numeric_limits<double>::epsilon(),
      -1.0 / (2.0 * highest_inverse_power));
}

bool is_certified_regular_overlap(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse) {
  if (!matrix.allFinite() || !inverse.allFinite()) {
    return false;
  }
  const double matrix_norm = infinity_norm(matrix);
  const double inverse_norm = infinity_norm(inverse);
  const double condition_estimate = matrix_norm * inverse_norm;

  // Accepted pairs feed derivatives containing as many as four inverse
  // factors.  This bound limits their roundoff amplification to sqrt(u), the
  // same precision-derived certificate used by CofactorDifferential.
  const double condition_limit = regular_overlap_condition_limit();
  if (!std::isfinite(condition_estimate) ||
      condition_estimate > condition_limit) {
    return false;
  }

  const Eigen::MatrixXd residual =
      Eigen::MatrixXd::Identity(matrix.rows(), matrix.cols()) -
      matrix * inverse;
  const double backward_error =
      infinity_norm(residual) / (1.0 + condition_estimate);
  return std::isfinite(backward_error) &&
      backward_error <= std::sqrt(std::numeric_limits<double>::epsilon());
}

std::optional<DeterminantOverlapResult> try_woodbury_right_overlap_update(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right_old,
    const std::vector<int>& occupied_right_new,
    const Eigen::Ref<const Eigen::MatrixXd>& active_overlap,
    const DeterminantOverlapResult& old_result) {
  const int n_electrons = static_cast<int>(occupied_left.size());
  if (static_cast<int>(occupied_right_old.size()) != n_electrons ||
      static_cast<int>(occupied_right_new.size()) != n_electrons ||
      old_result.nullity != 0 ||
      old_result.overlap_submatrix.rows() != n_electrons ||
      old_result.overlap_submatrix.cols() != n_electrons ||
      old_result.inverse_overlap_submatrix.rows() != n_electrons ||
      old_result.inverse_overlap_submatrix.cols() != n_electrons) {
    return std::nullopt;
  }

  const std::vector<int> rows = changed_rows(
      occupied_right_old, occupied_right_new);
  const int rank = static_cast<int>(rows.size());
  if (rank == 0) {
    return old_result;
  }
  // At full rank, Woodbury has the same cubic order as refactorization and
  // offers no algebraic advantage.
  if (rank >= n_electrons) {
    return std::nullopt;
  }

  Eigen::MatrixXd new_overlap = build_overlap_submatrix(
      occupied_left, occupied_right_new, active_overlap);
  Eigen::MatrixXd delta_rows(rank, n_electrons);
  Eigen::MatrixXd inverse_columns(n_electrons, rank);
  for (int local = 0; local < rank; ++local) {
    delta_rows.row(local) =
        new_overlap.row(rows[local]) -
        old_result.overlap_submatrix.row(rows[local]);
    inverse_columns.col(local) =
        old_result.inverse_overlap_submatrix.col(rows[local]);
  }

  const Eigen::MatrixXd middle =
      Eigen::MatrixXd::Identity(rank, rank) +
      delta_rows * inverse_columns;
  Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
  if (!middle_lu.isInvertible()) {
    return std::nullopt;
  }

  const Eigen::MatrixXd right_factor =
      delta_rows * old_result.inverse_overlap_submatrix;
  const Eigen::MatrixXd inverse_new =
      old_result.inverse_overlap_submatrix -
      inverse_columns * middle_lu.solve(right_factor);
  const double middle_determinant = middle_lu.determinant();
  return make_updated_result(
      std::move(new_overlap),
      std::move(inverse_new),
      middle_determinant,
      old_result);
}

std::optional<DeterminantOverlapResult> try_woodbury_left_overlap_update(
    const std::vector<int>& occupied_left_old,
    const std::vector<int>& occupied_left_new,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& active_overlap,
    const DeterminantOverlapResult& old_result) {
  const int n_electrons = static_cast<int>(occupied_right.size());
  if (static_cast<int>(occupied_left_old.size()) != n_electrons ||
      static_cast<int>(occupied_left_new.size()) != n_electrons ||
      old_result.nullity != 0 ||
      old_result.overlap_submatrix.rows() != n_electrons ||
      old_result.overlap_submatrix.cols() != n_electrons ||
      old_result.inverse_overlap_submatrix.rows() != n_electrons ||
      old_result.inverse_overlap_submatrix.cols() != n_electrons) {
    return std::nullopt;
  }

  const std::vector<int> columns = changed_columns(
      occupied_left_old, occupied_left_new);
  const int rank = static_cast<int>(columns.size());
  if (rank == 0) {
    return old_result;
  }
  if (rank >= n_electrons) {
    return std::nullopt;
  }

  Eigen::MatrixXd new_overlap = build_overlap_submatrix(
      occupied_left_new, occupied_right, active_overlap);
  Eigen::MatrixXd delta_columns(n_electrons, rank);
  Eigen::MatrixXd inverse_rows(rank, n_electrons);
  for (int local = 0; local < rank; ++local) {
    delta_columns.col(local) =
        new_overlap.col(columns[local]) -
        old_result.overlap_submatrix.col(columns[local]);
    inverse_rows.row(local) =
        old_result.inverse_overlap_submatrix.row(columns[local]);
  }

  const Eigen::MatrixXd inverse_delta =
      old_result.inverse_overlap_submatrix * delta_columns;
  Eigen::MatrixXd selected_inverse_delta(rank, rank);
  for (int local = 0; local < rank; ++local) {
    selected_inverse_delta.row(local) =
        inverse_delta.row(columns[local]);
  }
  const Eigen::MatrixXd middle =
      Eigen::MatrixXd::Identity(rank, rank) + selected_inverse_delta;
  Eigen::FullPivLU<Eigen::MatrixXd> middle_lu(middle);
  if (!middle_lu.isInvertible()) {
    return std::nullopt;
  }

  const Eigen::MatrixXd inverse_new =
      old_result.inverse_overlap_submatrix -
      inverse_delta * middle_lu.solve(inverse_rows);
  return make_updated_result(
      std::move(new_overlap),
      std::move(inverse_new),
      middle_lu.determinant(),
      old_result);
}

}  // namespace xmvb::vb
