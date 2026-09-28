#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "tools/pair_conditioning_census.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<double> flatten(const Eigen::MatrixXd& matrix) {
  return std::vector<double>(matrix.data(), matrix.data() + matrix.size());
}

void test_regular_pair() {
  const Eigen::Matrix4d overlap = Eigen::Matrix4d::Identity();
  const auto result = xmvb::tools::diagnose_pair_conditioning(
      {0, 1}, {0, 1}, flatten(overlap), 4);
  require(
      result.classification == xmvb::tools::PairConditioningClass::Regular,
      "an orthonormal self pair was not classified as regular");
  require(result.numerical_nullity == 0, "regular pair acquired nullity");
  require(
      std::abs(result.smallest_principal_singular_value - 1.0) < 1.0e-14,
      "regular pair principal singular value is not one");
}

void test_self_rank_deficiency() {
  Eigen::Matrix4d overlap = Eigen::Matrix4d::Identity();
  overlap(0, 1) = 1.0;
  overlap(1, 0) = 1.0;
  const auto result = xmvb::tools::diagnose_pair_conditioning(
      {0, 1}, {0, 1}, flatten(overlap), 4);
  require(
      result.classification ==
          xmvb::tools::PairConditioningClass::SelfRankDeficient,
      "a singular determinant self Gram was not identified");
}

void test_self_condition_limit() {
  Eigen::Matrix4d overlap = Eigen::Matrix4d::Identity();
  overlap(0, 1) = 0.99;
  overlap(1, 0) = 0.99;
  const auto result = xmvb::tools::diagnose_pair_conditioning(
      {0, 1}, {0, 1}, flatten(overlap), 4);
  require(
      result.classification ==
          xmvb::tools::PairConditioningClass::SelfConditionLimited,
      "a condition-limited determinant self Gram was not identified");
  require(
      std::abs(result.smallest_principal_singular_value - 1.0) < 1.0e-12,
      "self whitening did not remove representation conditioning");
}

void test_intrinsic_pair_conditioning() {
  Eigen::Matrix4d overlap = Eigen::Matrix4d::Identity();
  overlap(0, 2) = 0.5;
  overlap(2, 0) = 0.5;
  overlap(1, 3) = 1.0e-4;
  overlap(3, 1) = 1.0e-4;
  const auto result = xmvb::tools::diagnose_pair_conditioning(
      {0, 1}, {2, 3}, flatten(overlap), 4);
  require(
      result.classification ==
          xmvb::tools::PairConditioningClass::IntrinsicConditionLimited,
      "a principal-angle-limited pair was not identified");
  require(result.numerical_nullity == 0, "near-orthogonal pair became singular");
  require(result.dangerous_dimension == 1, "dangerous dimension is incorrect");

  overlap(1, 3) = 0.0;
  overlap(3, 1) = 0.0;
  const auto singular = xmvb::tools::diagnose_pair_conditioning(
      {0, 1}, {2, 3}, flatten(overlap), 4);
  require(
      singular.classification ==
          xmvb::tools::PairConditioningClass::IntrinsicRankDeficient,
      "an intrinsically singular pair was not identified");
  require(singular.numerical_nullity == 1, "intrinsic nullity is incorrect");
}

}  // namespace

int main() {
  try {
    test_regular_pair();
    test_self_rank_deficiency();
    test_self_condition_limit();
    test_intrinsic_pair_conditioning();
    std::cout << "Pair conditioning census tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
