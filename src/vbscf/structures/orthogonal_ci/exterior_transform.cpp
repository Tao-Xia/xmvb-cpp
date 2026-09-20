#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace xmvb::vb {
namespace {

std::size_t binomial(int n, int k) {
  if (k < 0 || k > n) {
    return 0;
  }
  k = std::min(k, n - k);
  std::size_t value = 1;
  for (int index = 1; index <= k; ++index) {
    value = value * static_cast<std::size_t>(n - k + index) /
        static_cast<std::size_t>(index);
  }
  return value;
}

double replacement_sign(std::uint64_t source, int removed, int inserted) {
  const int low = std::min(removed, inserted);
  const int high = std::max(removed, inserted);
  std::uint64_t between = 0;
  if (high - low > 1) {
    between = source &
        (((std::uint64_t{1} << high) - 1) ^
         ((std::uint64_t{1} << (low + 1)) - 1));
  }
  return __builtin_popcountll(between) % 2 == 0 ? 1.0 : -1.0;
}

}  // namespace

ExteriorOrbitalTransform::ExteriorOrbitalTransform(
    const std::vector<std::vector<int>>& determinants,
    const Eigen::Ref<const Eigen::MatrixXd>& upper_orbital_transform)
    : n_orbitals_(static_cast<int>(upper_orbital_transform.rows())),
      orbital_transform_(upper_orbital_transform) {
  if (n_orbitals_ <= 0 || n_orbitals_ > 63 ||
      upper_orbital_transform.cols() != n_orbitals_ ||
      determinants.empty() || !upper_orbital_transform.allFinite()) {
    throw std::invalid_argument("invalid exterior orbital transform inputs");
  }
  const double lower_error =
      upper_orbital_transform.template triangularView<Eigen::StrictlyLower>()
          .toDenseMatrix()
          .cwiseAbs()
          .maxCoeff();
  const double scale = std::max(
      1.0,
      upper_orbital_transform.cwiseAbs().maxCoeff());
  if (lower_error > 64.0 * std::numeric_limits<double>::epsilon() * scale ||
      (upper_orbital_transform.diagonal().array() <= 0.0).any()) {
    throw std::invalid_argument(
        "exterior orbital transform must be nonsingular upper triangular");
  }

  n_electrons_ = static_cast<int>(determinants.front().size());
  if (determinants.size() != binomial(n_orbitals_, n_electrons_)) {
    throw std::invalid_argument(
        "exterior transform requires a complete fixed-spin space");
  }
  masks_.reserve(determinants.size());
  std::unordered_map<std::uint64_t, int> index_by_mask;
  index_by_mask.reserve(determinants.size());
  for (int index = 0; index < static_cast<int>(determinants.size()); ++index) {
    const auto& occupied = determinants[index];
    if (static_cast<int>(occupied.size()) != n_electrons_) {
      throw std::invalid_argument(
          "fixed-spin determinants have inconsistent electron counts");
    }
    std::uint64_t mask = 0;
    int previous = -1;
    for (const int orbital : occupied) {
      if (orbital <= previous || orbital < 0 || orbital >= n_orbitals_) {
        throw std::invalid_argument(
            "fixed-spin determinant occupation is invalid");
      }
      mask |= std::uint64_t{1} << orbital;
      previous = orbital;
    }
    if (!index_by_mask.emplace(mask, index).second) {
      throw std::invalid_argument("fixed-spin determinant is duplicated");
    }
    masks_.push_back(mask);
  }

  determinant_scales_.resize(static_cast<int>(masks_.size()));
  for (int determinant = 0;
       determinant < static_cast<int>(masks_.size());
       ++determinant) {
    double scale_factor = 1.0;
    for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
      if ((masks_[determinant] & (std::uint64_t{1} << orbital)) != 0) {
        scale_factor *= upper_orbital_transform(orbital, orbital);
      }
    }
    determinant_scales_[determinant] = scale_factor;
  }

  // R = U D, where D contains the column diagonal and U is unit upper
  // triangular. Reduce U to identity by right-looking column eliminations.
  // Applying the inverses in elimination order evaluates U on a vector
  // because matrix factors act from right to left.
  Eigen::MatrixXd unit_upper = upper_orbital_transform;
  for (int column = 0; column < n_orbitals_; ++column) {
    unit_upper.col(column) /= upper_orbital_transform(column, column);
  }
  struct OrbitalShear {
    int inserted = 0;
    int removed = 0;
    double coefficient = 0.0;
  };
  std::vector<OrbitalShear> eliminations;
  for (int removed = 1; removed < n_orbitals_; ++removed) {
    for (int inserted = removed - 1; inserted >= 0; --inserted) {
      const double coefficient = unit_upper(inserted, removed);
      eliminations.push_back({inserted, removed, coefficient});
      unit_upper.col(removed).noalias() -=
          coefficient * unit_upper.col(inserted);
    }
  }

  shears_.reserve(eliminations.size());
  for (const OrbitalShear& operation : eliminations) {
    Shear shear;
    shear.coefficient = operation.coefficient;
    const std::uint64_t removed_bit =
        std::uint64_t{1} << operation.removed;
    const std::uint64_t inserted_bit =
        std::uint64_t{1} << operation.inserted;
    for (int source = 0; source < static_cast<int>(masks_.size()); ++source) {
      const std::uint64_t source_mask = masks_[source];
      if ((source_mask & removed_bit) == 0 ||
          (source_mask & inserted_bit) != 0) {
        continue;
      }
      const std::uint64_t target_mask =
          (source_mask ^ removed_bit) | inserted_bit;
      const auto target = index_by_mask.find(target_mask);
      if (target == index_by_mask.end()) {
        throw std::invalid_argument(
            "exterior transform requires a complete fixed-spin space");
      }
      shear.pairs.push_back(DeterminantPair{
          source,
          target->second,
          replacement_sign(
              source_mask,
              operation.removed,
              operation.inserted)});
    }
    shears_.push_back(std::move(shear));
  }
}

void ExteriorOrbitalTransform::validate_left(
    const Eigen::MatrixXd& coefficients) const {
  if (coefficients.rows() != dimension()) {
    throw std::invalid_argument(
        "left exterior transform determinant dimension mismatch");
  }
}

void ExteriorOrbitalTransform::validate_right(
    const Eigen::MatrixXd& coefficients) const {
  if (coefficients.cols() != dimension()) {
    throw std::invalid_argument(
        "right exterior transform determinant dimension mismatch");
  }
}

void ExteriorOrbitalTransform::apply_left(
    Eigen::MatrixXd* coefficients) const {
  if (coefficients == nullptr) {
    throw std::invalid_argument("left exterior transform output is null");
  }
  validate_left(*coefficients);
  coefficients->array().colwise() *= determinant_scales_.array();
  for (const Shear& shear : shears_) {
    for (const DeterminantPair& pair : shear.pairs) {
      coefficients->row(pair.target) +=
          (shear.coefficient * pair.sign) * coefficients->row(pair.source);
    }
  }
}

void ExteriorOrbitalTransform::apply_right(
    Eigen::MatrixXd* coefficients) const {
  if (coefficients == nullptr) {
    throw std::invalid_argument("right exterior transform output is null");
  }
  apply_right_block(*coefficients);
}

void ExteriorOrbitalTransform::apply_right_block(
    Eigen::Ref<Eigen::MatrixXd> coefficients) const {
  if (coefficients.cols() != dimension()) {
    throw std::invalid_argument(
        "right exterior transform determinant dimension mismatch");
  }
  coefficients.array().rowwise() *= determinant_scales_.transpose().array();
  for (const Shear& shear : shears_) {
    for (const DeterminantPair& pair : shear.pairs) {
      coefficients.col(pair.target) +=
          (shear.coefficient * pair.sign) * coefficients.col(pair.source);
    }
  }
}

void ExteriorOrbitalTransform::apply_adjoint_left(
    Eigen::MatrixXd* coefficients) const {
  if (coefficients == nullptr) {
    throw std::invalid_argument("left exterior adjoint output is null");
  }
  validate_left(*coefficients);
  for (auto shear = shears_.rbegin(); shear != shears_.rend(); ++shear) {
    for (const DeterminantPair& pair : shear->pairs) {
      coefficients->row(pair.source) +=
          (shear->coefficient * pair.sign) * coefficients->row(pair.target);
    }
  }
  coefficients->array().colwise() *= determinant_scales_.array();
}

void ExteriorOrbitalTransform::apply_adjoint_right(
    Eigen::MatrixXd* coefficients) const {
  if (coefficients == nullptr) {
    throw std::invalid_argument("right exterior adjoint output is null");
  }
  apply_adjoint_right_block(*coefficients);
}

void ExteriorOrbitalTransform::apply_adjoint_right_block(
    Eigen::Ref<Eigen::MatrixXd> coefficients) const {
  if (coefficients.cols() != dimension()) {
    throw std::invalid_argument(
        "right exterior transform determinant dimension mismatch");
  }
  for (auto shear = shears_.rbegin(); shear != shears_.rend(); ++shear) {
    for (const DeterminantPair& pair : shear->pairs) {
      coefficients.col(pair.source) +=
          (shear->coefficient * pair.sign) * coefficients.col(pair.target);
    }
  }
  coefficients.array().rowwise() *= determinant_scales_.transpose().array();
}

ExteriorTransformDirection ExteriorOrbitalTransform::direction(
    const Eigen::Ref<const Eigen::MatrixXd>& delta_orbital_transform) const {
  if (delta_orbital_transform.rows() != n_orbitals_ ||
      delta_orbital_transform.cols() != n_orbitals_ ||
      !delta_orbital_transform.allFinite()) {
    throw std::invalid_argument(
        "exterior transform direction has incompatible dimensions");
  }
  const double lower_error =
      delta_orbital_transform
          .template triangularView<Eigen::StrictlyLower>()
          .toDenseMatrix()
          .cwiseAbs()
          .maxCoeff();
  const double scale = std::max(
      1.0,
      delta_orbital_transform.cwiseAbs().maxCoeff());
  if (lower_error > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
    throw std::invalid_argument(
        "exterior transform direction must be upper triangular");
  }

  ExteriorTransformDirection result;
  result.determinant_scales.resize(dimension());
  for (int determinant = 0; determinant < dimension(); ++determinant) {
    double logarithmic_direction = 0.0;
    for (int orbital = 0; orbital < n_orbitals_; ++orbital) {
      if ((masks_[determinant] & (std::uint64_t{1} << orbital)) != 0) {
        logarithmic_direction +=
            delta_orbital_transform(orbital, orbital) /
            orbital_transform_(orbital, orbital);
      }
    }
    result.determinant_scales[determinant] =
        determinant_scales_[determinant] * logarithmic_direction;
  }

  Eigen::MatrixXd unit_upper = orbital_transform_;
  Eigen::MatrixXd delta_unit_upper = delta_orbital_transform;
  for (int column = 0; column < n_orbitals_; ++column) {
    const double diagonal = orbital_transform_(column, column);
    const double delta_diagonal = delta_orbital_transform(column, column);
    delta_unit_upper.col(column) =
        delta_unit_upper.col(column) / diagonal -
        unit_upper.col(column) * (delta_diagonal / (diagonal * diagonal));
    unit_upper.col(column) /= diagonal;
  }

  result.shear_coefficients.reserve(shears_.size());
  for (int removed = 1; removed < n_orbitals_; ++removed) {
    for (int inserted = removed - 1; inserted >= 0; --inserted) {
      const double coefficient = unit_upper(inserted, removed);
      const double delta_coefficient =
          delta_unit_upper(inserted, removed);
      result.shear_coefficients.push_back(delta_coefficient);
      delta_unit_upper.col(removed).noalias() -=
          delta_coefficient * unit_upper.col(inserted) +
          coefficient * delta_unit_upper.col(inserted);
      unit_upper.col(removed).noalias() -=
          coefficient * unit_upper.col(inserted);
    }
  }
  if (result.shear_coefficients.size() != shears_.size()) {
    throw std::logic_error(
        "exterior transform direction does not match its shear schedule");
  }
  return result;
}

void ExteriorOrbitalTransform::apply_directional_left(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
  if (coefficients == nullptr || directional_coefficients == nullptr ||
      coefficients->rows() != directional_coefficients->rows() ||
      coefficients->cols() != directional_coefficients->cols() ||
      direction.determinant_scales.size() != dimension() ||
      direction.shear_coefficients.size() != shears_.size()) {
    throw std::invalid_argument("invalid directional left exterior action");
  }
  validate_left(*coefficients);
  for (int row = 0; row < dimension(); ++row) {
    directional_coefficients->row(row) =
        determinant_scales_[row] * directional_coefficients->row(row) +
        direction.determinant_scales[row] * coefficients->row(row);
    coefficients->row(row) *= determinant_scales_[row];
  }
  for (std::size_t shear_index = 0;
       shear_index < shears_.size();
       ++shear_index) {
    const Shear& shear = shears_[shear_index];
    const double delta_coefficient =
        direction.shear_coefficients[shear_index];
    for (const DeterminantPair& pair : shear.pairs) {
      directional_coefficients->row(pair.target) += pair.sign *
          (delta_coefficient * coefficients->row(pair.source) +
           shear.coefficient * directional_coefficients->row(pair.source));
      coefficients->row(pair.target) +=
          (shear.coefficient * pair.sign) * coefficients->row(pair.source);
    }
  }
}

void ExteriorOrbitalTransform::apply_directional_right(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
  if (coefficients == nullptr || directional_coefficients == nullptr ||
      coefficients->rows() != directional_coefficients->rows() ||
      coefficients->cols() != directional_coefficients->cols() ||
      direction.determinant_scales.size() != dimension() ||
      direction.shear_coefficients.size() != shears_.size()) {
    throw std::invalid_argument("invalid directional right exterior action");
  }
  validate_right(*coefficients);
  for (int column = 0; column < dimension(); ++column) {
    directional_coefficients->col(column) =
        determinant_scales_[column] * directional_coefficients->col(column) +
        direction.determinant_scales[column] * coefficients->col(column);
    coefficients->col(column) *= determinant_scales_[column];
  }
  for (std::size_t shear_index = 0;
       shear_index < shears_.size();
       ++shear_index) {
    const Shear& shear = shears_[shear_index];
    const double delta_coefficient =
        direction.shear_coefficients[shear_index];
    for (const DeterminantPair& pair : shear.pairs) {
      directional_coefficients->col(pair.target) += pair.sign *
          (delta_coefficient * coefficients->col(pair.source) +
           shear.coefficient * directional_coefficients->col(pair.source));
      coefficients->col(pair.target) +=
          (shear.coefficient * pair.sign) * coefficients->col(pair.source);
    }
  }
}

void ExteriorOrbitalTransform::apply_directional_adjoint_left(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
  if (coefficients == nullptr || directional_coefficients == nullptr ||
      coefficients->rows() != directional_coefficients->rows() ||
      coefficients->cols() != directional_coefficients->cols() ||
      direction.determinant_scales.size() != dimension() ||
      direction.shear_coefficients.size() != shears_.size()) {
    throw std::invalid_argument("invalid directional adjoint-left exterior action");
  }
  validate_left(*coefficients);
  for (std::size_t reverse = shears_.size(); reverse-- > 0;) {
    const Shear& shear = shears_[reverse];
    const double delta_coefficient = direction.shear_coefficients[reverse];
    for (const DeterminantPair& pair : shear.pairs) {
      directional_coefficients->row(pair.source) += pair.sign *
          (delta_coefficient * coefficients->row(pair.target) +
           shear.coefficient * directional_coefficients->row(pair.target));
      coefficients->row(pair.source) +=
          (shear.coefficient * pair.sign) * coefficients->row(pair.target);
    }
  }
  for (int row = 0; row < dimension(); ++row) {
    directional_coefficients->row(row) =
        determinant_scales_[row] * directional_coefficients->row(row) +
        direction.determinant_scales[row] * coefficients->row(row);
    coefficients->row(row) *= determinant_scales_[row];
  }
}

void ExteriorOrbitalTransform::apply_directional_adjoint_right(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
  if (coefficients == nullptr || directional_coefficients == nullptr ||
      coefficients->rows() != directional_coefficients->rows() ||
      coefficients->cols() != directional_coefficients->cols() ||
      direction.determinant_scales.size() != dimension() ||
      direction.shear_coefficients.size() != shears_.size()) {
    throw std::invalid_argument("invalid directional adjoint-right exterior action");
  }
  validate_right(*coefficients);
  for (std::size_t reverse = shears_.size(); reverse-- > 0;) {
    const Shear& shear = shears_[reverse];
    const double delta_coefficient = direction.shear_coefficients[reverse];
    for (const DeterminantPair& pair : shear.pairs) {
      directional_coefficients->col(pair.source) += pair.sign *
          (delta_coefficient * coefficients->col(pair.target) +
           shear.coefficient * directional_coefficients->col(pair.target));
      coefficients->col(pair.source) +=
          (shear.coefficient * pair.sign) * coefficients->col(pair.target);
    }
  }
  for (int column = 0; column < dimension(); ++column) {
    directional_coefficients->col(column) =
        determinant_scales_[column] * directional_coefficients->col(column) +
        direction.determinant_scales[column] * coefficients->col(column);
    coefficients->col(column) *= determinant_scales_[column];
  }
}

std::size_t ExteriorOrbitalTransform::shear_pair_count() const noexcept {
  std::size_t count = 0;
  for (const Shear& shear : shears_) {
    count += shear.pairs.size();
  }
  return count;
}

std::size_t ExteriorOrbitalTransform::dynamic_bytes() const noexcept {
  std::size_t bytes =
      static_cast<std::size_t>(orbital_transform_.size()) * sizeof(double) +
      masks_.capacity() * sizeof(std::uint64_t) +
      static_cast<std::size_t>(determinant_scales_.size()) * sizeof(double) +
      shears_.capacity() * sizeof(Shear);
  for (const Shear& shear : shears_) {
    bytes += shear.pairs.capacity() * sizeof(DeterminantPair);
  }
  return bytes;
}

}  // namespace xmvb::vb
