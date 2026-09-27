#include "vbscf/determinants/algebra/compound.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace xmvb::vb {
namespace {

using IndexMatrix =
    Eigen::Matrix<Eigen::Index, Eigen::Dynamic, Eigen::Dynamic>;

void enumerate_subsets(
    int size, int order, int next, std::vector<int>& current,
    std::vector<std::vector<int>>& result) {
  if (static_cast<int>(current.size()) == order) {
    result.push_back(current);
    return;
  }

  const int missing = order - static_cast<int>(current.size());
  for (int value = next; value <= size - missing; ++value) {
    current.push_back(value);
    enumerate_subsets(size, order, value + 1, current, result);
    current.pop_back();
  }
}

double parity(int row_position, int column_position) {
  return (row_position + column_position) % 2 == 0 ? 1.0 : -1.0;
}

void add_rank_one_compound(
    const CompoundBasis& basis, int order,
    const Eigen::Ref<const Eigen::MatrixXd>& lower,
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right,
    Eigen::MatrixXd& target) {
  const Eigen::Index count = basis.level_size(order);
  for (Eigen::Index row = 0; row < count; ++row) {
    const auto& row_subset = basis.subset(order, row);
    for (Eigen::Index column = 0; column < count; ++column) {
      const auto& column_subset = basis.subset(order, column);
      double correction = 0.0;
      for (int row_position = 0; row_position < order; ++row_position) {
        const Eigen::Index lower_row =
            basis.removed_index(order, row, row_position);
        const double row_factor = left(row_subset[row_position]);
        for (int column_position = 0; column_position < order;
             ++column_position) {
          const Eigen::Index lower_column =
              basis.removed_index(order, column, column_position);
          correction += parity(row_position, column_position) * row_factor *
              right(column_subset[column_position]) *
              lower(lower_row, lower_column);
        }
      }
      target(row, column) += correction;
    }
  }
}

}  // namespace

CompoundBasis::CompoundBasis(int size, int max_order)
    : size_(size), max_order_(max_order) {
  if (size < 0 || max_order < 0 || max_order > size) {
    throw std::invalid_argument("invalid compound basis dimensions");
  }

  subsets_.resize(max_order + 1);
  removed_.resize(max_order + 1);
  subsets_[0].push_back({});

  for (int order = 1; order <= max_order; ++order) {
    std::vector<int> current;
    enumerate_subsets(size, order, 0, current, subsets_[order]);
    if (subsets_[order].size() >
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max())) {
      throw std::overflow_error("compound level exceeds Eigen index range");
    }

    std::map<std::vector<int>, Eigen::Index> lower_indices;
    for (Eigen::Index index = 0;
         index < static_cast<Eigen::Index>(subsets_[order - 1].size());
         ++index) {
      lower_indices.emplace(subsets_[order - 1][index], index);
    }

    IndexMatrix removal(subsets_[order].size(), order);
    for (Eigen::Index index = 0;
         index < static_cast<Eigen::Index>(subsets_[order].size()); ++index) {
      for (int position = 0; position < order; ++position) {
        std::vector<int> reduced = subsets_[order][index];
        reduced.erase(reduced.begin() + position);
        removal(index, position) = lower_indices.at(reduced);
      }
    }
    removed_[order] = std::move(removal);
  }
}

Eigen::Index CompoundBasis::level_size(int order) const {
  if (order < 0 || order > max_order_) {
    throw std::out_of_range("compound order is outside the basis");
  }
  return static_cast<Eigen::Index>(subsets_[order].size());
}

const std::vector<int>& CompoundBasis::subset(
    int order, Eigen::Index index) const {
  if (order < 0 || order > max_order_ || index < 0 ||
      index >= level_size(order)) {
    throw std::out_of_range("compound subset index is outside the basis");
  }
  return subsets_[order][index];
}

Eigen::Index CompoundBasis::removed_index(
    int order, Eigen::Index index, int position) const {
  if (order <= 0 || order > max_order_ || index < 0 ||
      index >= level_size(order) || position < 0 || position >= order) {
    throw std::out_of_range("invalid compound removal index");
  }
  return removed_[order](index, position);
}

CompoundHierarchy::CompoundHierarchy(int size, int max_order)
    : CompoundHierarchy(std::make_shared<CompoundBasis>(size, max_order)) {}

CompoundHierarchy::CompoundHierarchy(
    std::shared_ptr<const CompoundBasis> basis)
    : basis_(std::move(basis)), levels_(basis_->max_order() + 1) {
  for (int order = 0; order <= basis_->max_order(); ++order) {
    const Eigen::Index count = basis_->level_size(order);
    if (count != 0 &&
        count > std::numeric_limits<Eigen::Index>::max() / count) {
      throw std::overflow_error("compound matrix dimension overflow");
    }
    const auto entries = static_cast<std::uintmax_t>(count) *
        static_cast<std::uintmax_t>(count);
    if (entries > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
      throw std::overflow_error("compound matrix byte size overflow");
    }
    levels_[order] = Eigen::MatrixXd::Zero(count, count);
  }
  levels_[0](0, 0) = 1.0;
}

void CompoundHierarchy::assign(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != basis_->size() || matrix.cols() != basis_->size()) {
    throw std::invalid_argument("compound anchor matrix has the wrong size");
  }

  levels_[0](0, 0) = 1.0;
  for (int order = 1; order <= basis_->max_order(); ++order) {
    Eigen::MatrixXd& current = levels_[order];
    const Eigen::MatrixXd& lower = levels_[order - 1];
    const Eigen::Index count = basis_->level_size(order);
    for (Eigen::Index row = 0; row < count; ++row) {
      const auto& row_subset = basis_->subset(order, row);
      const int row_position = order - 1;
      const int matrix_row = row_subset.back();
      const Eigen::Index lower_row =
          basis_->removed_index(order, row, row_position);
      for (Eigen::Index column = 0; column < count; ++column) {
        const auto& column_subset = basis_->subset(order, column);
        double value = 0.0;
        for (int column_position = 0; column_position < order;
             ++column_position) {
          const Eigen::Index lower_column =
              basis_->removed_index(order, column, column_position);
          value += parity(row_position, column_position) *
              matrix(matrix_row, column_subset[column_position]) *
              lower(lower_row, lower_column);
        }
        current(row, column) = value;
      }
    }
  }
}

CompoundHierarchy CompoundHierarchy::directional(
    const Eigen::Ref<const Eigen::MatrixXd>& direction) const {
  if (direction.rows() != basis_->size() ||
      direction.cols() != basis_->size()) {
    throw std::invalid_argument("compound direction has the wrong size");
  }

  CompoundHierarchy result(basis_);
  result.levels_[0].setZero();
  for (int order = 1; order <= basis_->max_order(); ++order) {
    Eigen::MatrixXd& target = result.levels_[order];
    const Eigen::MatrixXd& lower = levels_[order - 1];
    const Eigen::Index count = basis_->level_size(order);
    for (Eigen::Index row = 0; row < count; ++row) {
      const auto& row_subset = basis_->subset(order, row);
      for (Eigen::Index column = 0; column < count; ++column) {
        const auto& column_subset = basis_->subset(order, column);
        double value = 0.0;
        for (int row_position = 0; row_position < order; ++row_position) {
          const Eigen::Index lower_row =
              basis_->removed_index(order, row, row_position);
          for (int column_position = 0; column_position < order;
               ++column_position) {
            const Eigen::Index lower_column =
                basis_->removed_index(order, column, column_position);
            value += parity(row_position, column_position) *
                direction(row_subset[row_position],
                          column_subset[column_position]) *
                lower(lower_row, lower_column);
          }
        }
        target(row, column) = value;
      }
    }
  }
  return result;
}

void CompoundHierarchy::validate_vector(
    const Eigen::Ref<const Eigen::VectorXd>& vector) const {
  if (vector.size() != basis_->size()) {
    throw std::invalid_argument("compound update vector has the wrong size");
  }
}

void CompoundHierarchy::validate_compatible(
    const CompoundHierarchy& other) const {
  if (basis_->size() != other.basis_->size() ||
      basis_->max_order() != other.basis_->max_order()) {
    throw std::invalid_argument("compound hierarchies are incompatible");
  }
}

void CompoundHierarchy::apply_rank_one(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right) {
  validate_vector(left);
  validate_vector(right);
  for (int order = basis_->max_order(); order >= 1; --order) {
    add_rank_one_compound(
        *basis_, order, levels_[order - 1], left, right, levels_[order]);
  }
}

void CompoundHierarchy::apply_rank_one(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right,
    const Eigen::Ref<const Eigen::VectorXd>& dleft,
    const Eigen::Ref<const Eigen::VectorXd>& dright,
    CompoundHierarchy& tangent) {
  validate_vector(left);
  validate_vector(right);
  validate_vector(dleft);
  validate_vector(dright);
  validate_compatible(tangent);
  if (this == &tangent) {
    throw std::invalid_argument("accepted and tangent hierarchies must differ");
  }

  for (int order = basis_->max_order(); order >= 1; --order) {
    add_rank_one_compound(
        *basis_, order, tangent.levels_[order - 1], left, right,
        tangent.levels_[order]);
    add_rank_one_compound(
        *basis_, order, levels_[order - 1], dleft, right,
        tangent.levels_[order]);
    add_rank_one_compound(
        *basis_, order, levels_[order - 1], left, dright,
        tangent.levels_[order]);
    add_rank_one_compound(
        *basis_, order, levels_[order - 1], left, right, levels_[order]);
  }
}

Eigen::MatrixXd CompoundHierarchy::pullback(
    int order, const Eigen::Ref<const Eigen::MatrixXd>& weights) const {
  if (order < 0 || order > basis_->max_order()) {
    throw std::out_of_range("compound pullback order is outside the hierarchy");
  }
  const Eigen::Index count = basis_->level_size(order);
  if (weights.rows() != count || weights.cols() != count) {
    throw std::invalid_argument("compound pullback weights have the wrong size");
  }

  Eigen::MatrixXd gradient =
      Eigen::MatrixXd::Zero(basis_->size(), basis_->size());
  if (order == 0) {
    return gradient;
  }

  const Eigen::MatrixXd& lower = levels_[order - 1];
  for (Eigen::Index row = 0; row < count; ++row) {
    const auto& row_subset = basis_->subset(order, row);
    for (Eigen::Index column = 0; column < count; ++column) {
      const auto& column_subset = basis_->subset(order, column);
      const double weight = weights(row, column);
      for (int row_position = 0; row_position < order; ++row_position) {
        const Eigen::Index lower_row =
            basis_->removed_index(order, row, row_position);
        for (int column_position = 0; column_position < order;
             ++column_position) {
          const Eigen::Index lower_column =
              basis_->removed_index(order, column, column_position);
          gradient(row_subset[row_position],
                   column_subset[column_position]) +=
              parity(row_position, column_position) * weight *
              lower(lower_row, lower_column);
        }
      }
    }
  }
  return gradient;
}

const Eigen::MatrixXd& CompoundHierarchy::level(int order) const {
  if (order < 0 || order > basis_->max_order()) {
    throw std::out_of_range("compound level is outside the hierarchy");
  }
  return levels_[order];
}

}  // namespace xmvb::vb
