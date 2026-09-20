#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/response/ri.hpp"
#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool is_close(
    const Eigen::Ref<const Eigen::MatrixXd>& actual,
    const Eigen::Ref<const Eigen::MatrixXd>& expected,
    double tolerance) {
  return (actual - expected).norm() <=
      tolerance * std::max(1.0, expected.norm());
}

Eigen::MatrixXd build_pair_map(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients) {
  const int n_bf = static_cast<int>(coefficients.rows());
  const int n_active = static_cast<int>(coefficients.cols());
  const int n_ao_pairs = n_bf * (n_bf + 1) / 2;
  const int n_active_pairs = n_active * (n_active + 1) / 2;
  Eigen::MatrixXd pair_map(n_ao_pairs, n_active_pairs);
  int ao_pair = 0;
  for (int first_basis = 0; first_basis < n_bf; ++first_basis) {
    for (int second_basis = 0;
         second_basis <= first_basis;
         ++second_basis, ++ao_pair) {
      int active_pair = 0;
      for (int first_active = 0;
           first_active < n_active;
           ++first_active) {
        for (int second_active = 0;
             second_active <= first_active;
             ++second_active, ++active_pair) {
          double value =
              coefficients(first_basis, first_active) *
              coefficients(second_basis, second_active);
          if (first_basis != second_basis) {
            value +=
                coefficients(second_basis, first_active) *
                coefficients(first_basis, second_active);
          }
          pair_map(ao_pair, active_pair) = value;
        }
      }
    }
  }
  return pair_map;
}

xmvb::vb::ActiveSpaceTwoElectronResult make_result(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    const xmvb::vb::RiAoFactorization& factorization) {
  xmvb::vb::ActiveSpaceTwoElectronResult result;
  result.representation =
      xmvb::vb::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
  result.n_auxiliary_functions = factorization.n_auxiliary_functions;
  result.dense_active_coefficients = coefficients;
  result.ri_active_pair_factors =
      factorization.metric_whitened_ao_pair_factors *
      build_pair_map(coefficients);
  return result;
}

std::vector<double> pack_gram(
    const Eigen::Ref<const Eigen::MatrixXd>& factors) {
  const Eigen::MatrixXd gram = factors.transpose() * factors;
  const int n_pairs = static_cast<int>(gram.rows());
  std::vector<double> packed(
      static_cast<std::size_t>(n_pairs) * (n_pairs + 1) / 2,
      0.0);
  for (int first = 0; first < n_pairs; ++first) {
    for (int second = 0; second <= first; ++second) {
      packed[xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(
          first, second)] = gram(first, second);
    }
  }
  return packed;
}

double packed_dot(
    const std::vector<double>& left,
    const std::vector<double>& right) {
  require(left.size() == right.size(), "packed dot size mismatch");
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    result += left[index] * right[index];
  }
  return result;
}

void check_direct_factor_direction() {
  constexpr int n_bf = 50;
  constexpr int n_active = 24;
  constexpr int n_auxiliary = 7;
  constexpr int n_ao_pairs = n_bf * (n_bf + 1) / 2;

  xmvb::vb::RiAoFactorization factorization;
  factorization.n_basis_functions = n_bf;
  factorization.n_auxiliary_functions = n_auxiliary;
  factorization.n_packed_ao_pairs = n_ao_pairs;
  factorization.metric_whitened_ao_pair_factors.resize(
      n_auxiliary, n_ao_pairs);
  for (Eigen::Index index = 0;
       index < factorization.metric_whitened_ao_pair_factors.size();
       ++index) {
    factorization.metric_whitened_ao_pair_factors.data()[index] =
        std::sin(0.013 * static_cast<double>(index + 1));
  }

  Eigen::MatrixXd coefficients(n_bf, n_active);
  Eigen::MatrixXd direction(n_bf, n_active);
  for (Eigen::Index index = 0; index < coefficients.size(); ++index) {
    coefficients.data()[index] =
        std::cos(0.021 * static_cast<double>(index + 2));
    direction.data()[index] =
        std::sin(0.017 * static_cast<double>(index + 3));
  }

  const auto accepted_result = make_result(coefficients, factorization);
  const auto cache =
      xmvb::vb::build_ri_active_two_electron_response_cache(
          factorization, accepted_result, n_active);
  const Eigen::MatrixXd actual =
      xmvb::vb::compute_ri_active_pair_factor_directional_derivative(
          cache, direction);
  xmvb::vb::PackedOrbitalPairMapMatrix pair_direction;
  xmvb::vb::build_packed_orbital_pair_map_directional_derivative(
      coefficients, direction, &pair_direction);
  const Eigen::MatrixXd expected =
      factorization.metric_whitened_ao_pair_factors * pair_direction;
  require(
      is_close(actual, expected, 2.0e-13),
      "direct RI active-pair factor direction changed the packed result");
  const Eigen::MatrixXd second_direction =
      -0.37 * direction + 0.11 * coefficients;
  const auto batch =
      xmvb::vb::compute_ri_active_pair_factor_directional_derivative_batch(
          cache, {direction, second_direction});
  require(batch.size() == 2 && is_close(batch[0], actual, 2.0e-13),
          "direct RI active-pair batch changed its first direction");
  require(
      is_close(
          batch[1],
          xmvb::vb::compute_ri_active_pair_factor_directional_derivative(
              cache, second_direction),
          2.0e-13),
      "direct RI active-pair batch changed its second direction");
}

}  // namespace

int main() {
  check_direct_factor_direction();
  constexpr int n_bf = 3;
  constexpr int n_active = 2;
  constexpr int n_auxiliary = 4;
  constexpr int n_ao_pairs = n_bf * (n_bf + 1) / 2;

  xmvb::vb::RiAoFactorization factorization;
  factorization.n_basis_functions = n_bf;
  factorization.n_auxiliary_functions = n_auxiliary;
  factorization.n_packed_ao_pairs = n_ao_pairs;
  factorization.metric_whitened_ao_pair_factors.resize(
      n_auxiliary,
      n_ao_pairs);
  factorization.metric_whitened_ao_pair_factors <<
      0.7, -0.2, 0.4, 0.1, -0.3, 0.8,
      -0.1, 0.9, 0.2, -0.5, 0.6, 0.3,
      0.4, 0.1, -0.7, 0.8, 0.2, -0.6,
      -0.3, 0.5, 0.6, -0.2, 0.9, 0.1;

  Eigen::MatrixXd coefficients(n_bf, n_active);
  coefficients <<
      0.8, -0.3,
      0.2, 0.9,
      -0.5, 0.4;
  Eigen::MatrixXd direction(n_bf, n_active);
  direction <<
      -0.2, 0.5,
      0.7, -0.1,
      0.3, 0.6;

  xmvb::vb::PackedOrbitalPairMapMatrix common_pair_map;
  xmvb::vb::build_packed_orbital_pair_map(
      coefficients,
      &common_pair_map);
  require(
      (common_pair_map - build_pair_map(coefficients)).norm() <= 1.0e-14,
      "common packed orbital pair map changed the canonical convention");

  const auto accepted_result = make_result(coefficients, factorization);
  const auto cache =
      xmvb::vb::build_ri_active_two_electron_response_cache(
          factorization,
          accepted_result,
          n_active);
  const Eigen::MatrixXd factor_direction =
      xmvb::vb::compute_ri_active_pair_factor_directional_derivative(
          cache,
          direction);
  const Eigen::MatrixXd second_direction =
      0.43 * direction - 0.08 * coefficients;
  const auto factor_direction_batch =
      xmvb::vb::compute_ri_active_pair_factor_directional_derivative_batch(
          cache, {direction, second_direction});
  require(
      factor_direction_batch.size() == 2 &&
          is_close(factor_direction_batch[0], factor_direction, 1.0e-13) &&
          is_close(
              factor_direction_batch[1],
              xmvb::vb::compute_ri_active_pair_factor_directional_derivative(
                  cache, second_direction),
              1.0e-13),
      "packed RI active-pair factor block disagrees with scalar actions");

  constexpr double step = 1.0e-6;
  const auto plus_result =
      make_result(coefficients + step * direction, factorization);
  const auto minus_result =
      make_result(coefficients - step * direction, factorization);
  const Eigen::MatrixXd finite_difference_factor_direction =
      (plus_result.ri_active_pair_factors -
       minus_result.ri_active_pair_factors) /
      (2.0 * step);
  require(
      is_close(factor_direction, finite_difference_factor_direction, 1.0e-9),
      "RI active-pair factor direction failed finite difference");

  const std::vector<double> packed_integral_direction =
      xmvb::vb::
          compute_ri_packed_active_two_electron_integral_directional_derivative(
              cache,
              factor_direction);
  const Eigen::MatrixXd packed_integral_direction_batch =
      xmvb::vb::
          compute_ri_packed_active_two_electron_integral_directional_derivative_batch(
              cache, factor_direction_batch);
  const std::vector<double> second_packed_integral_direction =
      xmvb::vb::
          compute_ri_packed_active_two_electron_integral_directional_derivative(
              cache, factor_direction_batch[1]);
  require(
      packed_integral_direction_batch.rows() ==
              static_cast<Eigen::Index>(packed_integral_direction.size()) &&
          packed_integral_direction_batch.cols() == 2,
      "RI packed active-2e direction block has the wrong shape");
  for (Eigen::Index index = 0;
       index < packed_integral_direction_batch.rows(); ++index) {
    require(
        std::abs(packed_integral_direction_batch(index, 0) -
                 packed_integral_direction[static_cast<std::size_t>(index)]) <=
                1.0e-13 &&
            std::abs(
                packed_integral_direction_batch(index, 1) -
                second_packed_integral_direction[
                    static_cast<std::size_t>(index)]) <= 1.0e-13,
        "RI packed active-2e direction block disagrees with scalar actions");
  }
  const std::vector<double> plus_integrals =
      pack_gram(plus_result.ri_active_pair_factors);
  const std::vector<double> minus_integrals =
      pack_gram(minus_result.ri_active_pair_factors);
  std::vector<double> finite_difference_integral_direction(
      plus_integrals.size(),
      0.0);
  for (std::size_t index = 0;
       index < plus_integrals.size();
       ++index) {
    finite_difference_integral_direction[index] =
        (plus_integrals[index] - minus_integrals[index]) /
        (2.0 * step);
    require(
        std::abs(packed_integral_direction[index] -
                 finite_difference_integral_direction[index]) <= 1.0e-8,
        "RI packed active-2e direction failed finite difference");
  }

  std::vector<double> packed_adjoint(
      packed_integral_direction.size(),
      0.0);
  for (std::size_t index = 0; index < packed_adjoint.size(); ++index) {
    packed_adjoint[index] =
        0.15 + 0.11 * static_cast<double>(index) -
        0.03 * static_cast<double>(index * index);
  }
  const Eigen::MatrixXd pullback =
      xmvb::vb::backpropagate_ri_packed_active_two_electron_gradient(
          packed_adjoint,
          cache);
  const double directional_duality =
      packed_dot(packed_adjoint, packed_integral_direction);
  require(
      std::abs(directional_duality -
               (pullback.array() * direction.array()).sum()) <= 1.0e-10,
      "RI packed-adjoint pullback violates directional duality");

  // Isolate every packed entry so diagonal entries exercise the doubled
  // factor adjoint while off-diagonal entries exercise both symmetric sides.
  for (std::size_t packed_index = 0;
       packed_index < packed_adjoint.size();
       ++packed_index) {
    std::vector<double> coordinate_adjoint(packed_adjoint.size(), 0.0);
    coordinate_adjoint[packed_index] = 1.0;
    const Eigen::MatrixXd coordinate_pullback =
        xmvb::vb::backpropagate_ri_packed_active_two_electron_gradient(
            coordinate_adjoint,
            cache);
    require(
        std::abs(
            packed_integral_direction[packed_index] -
            (coordinate_pullback.array() * direction.array()).sum()) <=
            1.0e-10,
        "RI packed coordinate pullback has a symmetry-factor error");
  }

  const auto plus_cache =
      xmvb::vb::build_ri_active_two_electron_response_cache(
          factorization,
          plus_result,
          n_active);
  const auto minus_cache =
      xmvb::vb::build_ri_active_two_electron_response_cache(
          factorization,
          minus_result,
          n_active);
  const Eigen::MatrixXd finite_difference_pullback_direction =
      (xmvb::vb::backpropagate_ri_packed_active_two_electron_gradient(
           packed_adjoint,
           plus_cache) -
       xmvb::vb::backpropagate_ri_packed_active_two_electron_gradient(
           packed_adjoint,
           minus_cache)) /
      (2.0 * step);
  const Eigen::MatrixXd fixed_adjoint_direction =
      xmvb::vb::
          apply_ri_packed_active_two_electron_adjoint_hessian_vector(
              packed_adjoint,
              cache,
              direction,
              &factor_direction);
  require(
      is_close(
          fixed_adjoint_direction,
          finite_difference_pullback_direction,
          1.0e-8),
      "RI fixed-adjoint orbital HVP failed finite difference");

  const Eigen::MatrixXd internally_built_factor_direction =
      xmvb::vb::
          apply_ri_packed_active_two_electron_adjoint_hessian_vector(
              packed_adjoint,
              cache,
              direction);
  require(
      is_close(
          internally_built_factor_direction,
          fixed_adjoint_direction,
          1.0e-13),
      "RI fixed-adjoint HVP changed when reusing factor direction");
  return 0;
}
