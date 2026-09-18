#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <Eigen/Core>

#include "vbscf/optimization/coupled/operator.hpp"
#include "vbscf/optimization/coupled/projection_cache.hpp"

namespace {

struct ActionCounts {
  int orbital_hessian = 0;
  int orbital_to_response = 0;
  int response_to_orbital = 0;
  int response_hessian = 0;
  int orbital_metric = 0;
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(
    const Eigen::Ref<const Eigen::MatrixXd>& actual,
    const Eigen::Ref<const Eigen::MatrixXd>& reference,
    double tolerance,
    const char* message) {
  require(
      actual.rows() == reference.rows() &&
          actual.cols() == reference.cols() &&
          (actual - reference).stableNorm() <=
              tolerance * std::max(1.0, reference.stableNorm()),
      message);
}

}  // namespace

int main() {
  try {
    Eigen::Matrix4d orbital_hessian;
    orbital_hessian << 3.0, 0.2, -0.1, 0.3,
                       0.2, 2.0, 0.4, -0.2,
                      -0.1, 0.4, 1.5, 0.1,
                       0.3, -0.2, 0.1, 2.5;
    Eigen::Matrix4d orbital_metric;
    orbital_metric << 1.5, 0.1, 0.0, 0.1,
                      0.1, 1.2, 0.1, 0.0,
                      0.0, 0.1, 0.9, 0.05,
                      0.1, 0.0, 0.05, 1.1;
    Eigen::Matrix<double, 3, 4> coupling;
    coupling << 0.3, -0.2, 0.1, 0.4,
                0.1, 0.5, -0.3, 0.2,
               -0.2, 0.1, 0.4, -0.1;
    Eigen::Matrix3d response_hessian;
    response_hessian << 2.0, 0.2, -0.1,
                        0.2, -1.0, 0.3,
                       -0.1, 0.3, 1.5;
    const auto counts = std::make_shared<ActionCounts>();
    const xmvb::vb::CoupledNewtonOperator coupled_operator(
        4,
        xmvb::vb::SelectedSubspaceResponseLayout(
            2,
            {xmvb::vb::SelectedStateCluster{1, 0.5}}),
        xmvb::vb::CoupledNewtonActions{
            [orbital_hessian, counts](const auto& x) {
              ++counts->orbital_hessian;
              return (orbital_hessian * x).eval();
            },
            [coupling, counts](const auto& x) {
              ++counts->orbital_to_response;
              return (coupling * x).eval();
            },
            [coupling, counts](const auto& x) {
              ++counts->response_to_orbital;
              return (coupling.transpose() * x).eval();
            },
            [response_hessian, counts](const auto& x) {
              ++counts->response_hessian;
              return (response_hessian * x).eval();
            },
            [orbital_metric, counts](const auto& x) {
              ++counts->orbital_metric;
              return (orbital_metric * x).eval();
            }});
    xmvb::vb::CoupledProjectionCache cache(coupled_operator);

    Eigen::Matrix<double, 4, 3> orbital_candidates;
    orbital_candidates.col(0) << 1.0, 2.0, 0.0, 0.0;
    orbital_candidates.col(1) = -2.0 * orbital_candidates.col(0);
    orbital_candidates.col(2) << 0.0, 1.0, 1.0, 0.5;
    require(cache.append_orbital_block(orbital_candidates) == 2,
            "orbital numerical-rank filter retained a dependent column");
    require(counts->orbital_metric == 1 &&
                counts->orbital_hessian == 1 &&
                counts->orbital_to_response == 1,
            "accepted orbital block did not use one action per operator");

    Eigen::Matrix<double, 3, 3> response_candidates;
    response_candidates.col(0) << 1.0, -1.0, 0.5;
    response_candidates.col(1) = 3.0 * response_candidates.col(0);
    response_candidates.col(2) << 0.0, 1.0, 2.0;
    require(cache.append_response_block(response_candidates) == 2,
            "response numerical-rank filter retained a dependent column");
    require(counts->response_to_orbital == 1 &&
                counts->response_hessian == 1,
            "accepted response block did not use one action per operator");

    require_close(
        cache.orbital_basis().transpose() * orbital_metric *
            cache.orbital_basis(),
        Eigen::Matrix2d::Identity(),
        2.0e-14,
        "orbital projection basis is not G-orthonormal");
    require_close(
        cache.response_basis().transpose() * cache.response_basis(),
        Eigen::Matrix2d::Identity(),
        2.0e-14,
        "response projection basis is not Euclidean-orthonormal");
    for (Eigen::Index column = 0;
         column < cache.orbital_basis().cols();
         ++column) {
      Eigen::Index pivot = 0;
      cache.orbital_basis().col(column).cwiseAbs().maxCoeff(&pivot);
      require(cache.orbital_basis()(pivot, column) >= 0.0,
              "orbital projection sign convention is not deterministic");
    }
    for (Eigen::Index column = 0;
         column < cache.response_basis().cols();
         ++column) {
      Eigen::Index pivot = 0;
      cache.response_basis().col(column).cwiseAbs().maxCoeff(&pivot);
      require(cache.response_basis()(pivot, column) >= 0.0,
              "response projection sign convention is not deterministic");
    }

    const Eigen::MatrixXd dependent_orbitals =
        cache.orbital_basis() *
        (Eigen::Matrix<double, 2, 2>() << 1.0, -0.4, 0.7, 2.0).finished();
    require(cache.append_orbital_block(dependent_orbitals) == 0,
            "dependent orbital reappend changed the cache rank");
    require(counts->orbital_metric == 2 &&
                counts->orbital_hessian == 1 &&
                counts->orbital_to_response == 1,
            "rejected orbital block evaluated unnecessary Hessian actions");
    const Eigen::MatrixXd dependent_responses =
        cache.response_basis() * Eigen::Vector2d(0.3, -1.1);
    require(cache.append_response_block(dependent_responses) == 0,
            "dependent response reappend changed the cache rank");
    require(counts->response_to_orbital == 1 &&
                counts->response_hessian == 1,
            "rejected response block evaluated operator actions");

    const Eigen::Vector4d new_orbital(0.0, 0.0, 0.0, 1.0);
    require(cache.append_orbital_block(new_orbital) == 1,
            "independent orbital expansion was rejected");
    const Eigen::Vector3d new_response(0.0, 0.0, 1.0);
    require(cache.append_response_block(new_response) == 1,
            "independent response expansion was rejected");
    require(counts->orbital_metric == 3 &&
                counts->orbital_hessian == 2 &&
                counts->orbital_to_response == 2 &&
                counts->response_to_orbital == 2 &&
                counts->response_hessian == 2,
            "cache expansion did not batch each required operator once");

    require_close(
        cache.projected_orbital_hessian(),
        cache.orbital_basis().transpose() * orbital_hessian *
            cache.orbital_basis(),
        2.0e-14,
        "cached projected orbital Hessian differs from dense reference");
    require_close(
        cache.projected_orbital_metric(),
        cache.orbital_basis().transpose() * orbital_metric *
            cache.orbital_basis(),
        2.0e-14,
        "cached projected metric differs from dense reference");
    require_close(
        cache.projected_response_hessian(),
        cache.response_basis().transpose() * response_hessian *
            cache.response_basis(),
        2.0e-14,
        "cached projected response Hessian differs from dense reference");
    require_close(
        cache.projected_coupling(),
        cache.response_basis().transpose() * coupling *
            cache.orbital_basis(),
        2.0e-14,
        "cached projected coupling differs from dense reference");
    require(cache.projected_coupling_adjoint_error() < 2.0e-14,
            "projected B/B^T audit failed");

    const Eigen::Vector3d orbital_coordinates(0.4, -0.2, 0.7);
    const Eigen::Vector3d response_coordinates(-0.3, 0.5, 0.1);
    const Eigen::Vector4d orbital_vector =
        cache.orbital_basis() * orbital_coordinates;
    const Eigen::Vector3d response_vector =
        cache.response_basis() * response_coordinates;
    const double shift = 0.35;
    const xmvb::vb::CoupledCachedBlocks image = cache.reconstruct_image(
        orbital_coordinates,
        response_coordinates,
        shift);
    require_close(
        image.orbital,
        orbital_hessian * orbital_vector +
            coupling.transpose() * response_vector +
            shift * orbital_metric * orbital_vector,
        2.0e-14,
        "cached orbital image reconstruction is inaccurate");
    require_close(
        image.response,
        coupling * orbital_vector + response_hessian * response_vector,
        2.0e-14,
        "cached response image reconstruction is inaccurate");
    require(image.packed().size() == 7,
            "cached block packing changed the coupled dimension");

    const double ritz_value = -0.6;
    const xmvb::vb::CoupledCachedBlocks residual =
        cache.reconstruct_ritz_residual(
            orbital_coordinates,
            response_coordinates,
            ritz_value);
    require_close(
        residual.orbital,
        orbital_hessian * orbital_vector +
            coupling.transpose() * response_vector -
            ritz_value * orbital_metric * orbital_vector,
        2.0e-14,
        "cached generalized Ritz residual is inaccurate");
    require_close(
        residual.response,
        coupling * orbital_vector + response_hessian * response_vector,
        2.0e-14,
        "cached response stationarity residual is inaccurate");

    std::cout << "two-space coupled projection cache: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
