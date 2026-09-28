#include "vbscf/determinants/algebra/contracted_exterior_jet.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

namespace xmvb::vb {
namespace {

using MatrixPolynomial = std::vector<Eigen::MatrixXd>;

std::vector<double> convolve(
    const std::vector<double>& left,
    const std::vector<double>& right,
    int maximum_order) {
  std::vector<double> result(maximum_order + 1, 0.0);
  for (int left_order = 0;
       left_order <= maximum_order &&
       left_order < static_cast<int>(left.size());
       ++left_order) {
    for (int right_order = 0;
         left_order + right_order <= maximum_order &&
         right_order < static_cast<int>(right.size());
         ++right_order) {
      result[left_order + right_order] +=
          left[left_order] * right[right_order];
    }
  }
  return result;
}

MatrixPolynomial multiply(
    const MatrixPolynomial& left,
    const MatrixPolynomial& right,
    int maximum_order) {
  if (left.empty() || right.empty() ||
      left.front().cols() != right.front().rows()) {
    throw std::invalid_argument(
        "contracted-exterior polynomial dimensions differ");
  }
  MatrixPolynomial result;
  result.reserve(maximum_order + 1);
  for (int order = 0; order <= maximum_order; ++order) {
    result.push_back(Eigen::MatrixXd::Zero(
        left.front().rows(), right.front().cols()));
    for (int left_order = 0;
         left_order <= order &&
         left_order < static_cast<int>(left.size());
         ++left_order) {
      const int right_order = order - left_order;
      if (right_order < static_cast<int>(right.size())) {
        result.back().noalias() +=
            left[left_order] * right[right_order];
      }
    }
  }
  return result;
}

std::vector<double> determinant_coefficients(
    const MatrixPolynomial& matrix,
    int maximum_order) {
  if (matrix.empty() || matrix.front().rows() != matrix.front().cols()) {
    throw std::invalid_argument(
        "contracted-exterior determinant requires square coefficients");
  }
  const int dimension = static_cast<int>(matrix.front().rows());
  if (dimension == 0) {
    std::vector<double> unit(maximum_order + 1, 0.0);
    unit[0] = 1.0;
    return unit;
  }
  for (const Eigen::MatrixXd& coefficient : matrix) {
    if (coefficient.rows() != dimension || coefficient.cols() != dimension) {
      throw std::invalid_argument(
          "contracted-exterior determinant coefficient shapes differ");
    }
  }

  // Faddeev--LeVerrier over the truncated polynomial ring is inverse-free,
  // including when a nonconstant coefficient is singular.
  MatrixPolynomial auxiliary;
  auxiliary.reserve(maximum_order + 1);
  auxiliary.push_back(Eigen::MatrixXd::Identity(dimension, dimension));
  for (int order = 1; order <= maximum_order; ++order) {
    auxiliary.push_back(Eigen::MatrixXd::Zero(dimension, dimension));
  }

  std::vector<double> characteristic(maximum_order + 1, 0.0);
  for (int step = 1; step <= dimension; ++step) {
    MatrixPolynomial product = multiply(matrix, auxiliary, maximum_order);
    for (int order = 0; order <= maximum_order; ++order) {
      characteristic[order] =
          -product[order].trace() / static_cast<double>(step);
      product[order].diagonal().array() += characteristic[order];
    }
    auxiliary = std::move(product);
  }
  if (dimension % 2 != 0) {
    for (double& coefficient : characteristic) {
      coefficient = -coefficient;
    }
  }
  return characteristic;
}

void validate_order(int dimension, int maximum_order) {
  if (maximum_order < 0 || maximum_order > dimension) {
    throw std::invalid_argument(
        "contracted-exterior order is outside the channel dimension");
  }
}

}  // namespace

ContractedExteriorJet build_contracted_exterior_jet(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    int maximum_order) {
  if (channel.rows() != channel.cols()) {
    throw std::invalid_argument(
        "contracted-exterior channel must be square");
  }
  validate_order(static_cast<int>(channel.rows()), maximum_order);

  ContractedExteriorJet result;
  result.coefficients.assign(maximum_order + 1, 0.0);
  result.coefficients[0] = 1.0;
  if (maximum_order == 0) {
    return result;
  }

  std::vector<double> traces(maximum_order + 1, 0.0);
  Eigen::MatrixXd power = channel;
  for (int order = 1; order <= maximum_order; ++order) {
    traces[order] = power.trace();
    if (order < maximum_order) {
      power = (power * channel).eval();
    }
  }
  for (int order = 1; order <= maximum_order; ++order) {
    double coefficient = 0.0;
    for (int trace_order = 1;
         trace_order <= order;
         ++trace_order) {
      const double sign = trace_order % 2 == 0 ? -1.0 : 1.0;
      coefficient += sign *
          result.coefficients[order - trace_order] * traces[trace_order];
    }
    result.coefficients[order] =
        coefficient / static_cast<double>(order);
  }
  return result;
}

void update_contracted_exterior_jet(
    const Eigen::Ref<const Eigen::MatrixXd>& channel,
    const Eigen::Ref<const Eigen::MatrixXd>& update_left,
    const Eigen::Ref<const Eigen::MatrixXd>& update_right,
    ContractedExteriorJet* jet) {
  if (jet == nullptr || jet->coefficients.empty()) {
    throw std::invalid_argument(
        "contracted-exterior update requires an initialized jet");
  }
  if (channel.rows() != channel.cols() ||
      update_left.rows() != channel.rows() ||
      update_right.rows() != channel.rows() ||
      update_left.cols() != update_right.cols()) {
    throw std::invalid_argument(
        "contracted-exterior low-rank update dimensions differ");
  }
  const int maximum_order =
      static_cast<int>(jet->coefficients.size()) - 1;
  validate_order(static_cast<int>(channel.rows()), maximum_order);
  const int rank = static_cast<int>(update_left.cols());
  if (rank == 0 || maximum_order == 0) {
    return;
  }

  MatrixPolynomial correction;
  correction.reserve(maximum_order + 1);
  correction.push_back(Eigen::MatrixXd::Identity(rank, rank));
  Eigen::MatrixXd krylov = update_left;
  for (int order = 1; order <= maximum_order; ++order) {
    const double sign = order % 2 == 0 ? -1.0 : 1.0;
    correction.push_back(
        sign * update_right.transpose() * krylov);
    if (order < maximum_order) {
      krylov = (channel * krylov).eval();
    }
  }
  const std::vector<double> determinant =
      determinant_coefficients(correction, maximum_order);
  jet->coefficients =
      convolve(jet->coefficients, determinant, maximum_order);
}

}  // namespace xmvb::vb
