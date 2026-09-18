#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "vbscf/optimization/coupled/operator.hpp"

namespace {

void require_close(
    const Eigen::Ref<const Eigen::MatrixXd>& actual,
    const Eigen::Ref<const Eigen::MatrixXd>& expected,
    double tolerance,
    const char* message) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
      (actual - expected).norm() > tolerance) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  try {
    using xmvb::vb::ClusterResponse;
    using xmvb::vb::CoupledNewtonActions;
    using xmvb::vb::CoupledNewtonOperator;
    using xmvb::vb::SelectedStateCluster;
    using xmvb::vb::SelectedSubspaceResponseLayout;

    SelectedSubspaceResponseLayout response_layout(
        3,
        std::vector<SelectedStateCluster>{{2, 0.25}, {1, 0.5}});
    if (response_layout.response_size() != 14 ||
        response_layout.coefficient_offset(0) != 0 ||
        response_layout.multiplier_offset(0) != 6 ||
        response_layout.coefficient_offset(1) != 10 ||
        response_layout.multiplier_offset(1) != 13 ||
        std::abs(response_layout.coordinate_scale(0) - std::sqrt(0.5)) >
            1.0e-15) {
      throw std::runtime_error("selected-subspace layout offsets are wrong");
    }

    std::vector<ClusterResponse> cluster_responses(2);
    cluster_responses[0].coefficients.resize(3, 2);
    cluster_responses[0].coefficients << 1.0, 4.0, 2.0, 5.0, 3.0, 6.0;
    cluster_responses[0].multipliers.resize(2, 2);
    cluster_responses[0].multipliers << 7.0, 9.0, 8.0, 10.0;
    cluster_responses[1].coefficients = Eigen::Vector3d(11.0, 12.0, 13.0);
    cluster_responses[1].multipliers = Eigen::MatrixXd::Constant(1, 1, 14.0);
    const Eigen::VectorXd packed = response_layout.pack(cluster_responses);
    const std::vector<ClusterResponse> unpacked =
        response_layout.unpack(packed);
    require_close(
        unpacked[0].coefficients,
        cluster_responses[0].coefficients,
        1.0e-14,
        "cluster coefficient packing changed values");
    require_close(
        unpacked[0].multipliers,
        cluster_responses[0].multipliers,
        1.0e-14,
        "full cluster multiplier packing changed values");

    // Build an actual two-state generalized-eigen cluster. The full 2 x 2
    // multiplier block enforces the horizontal selected-subspace gauge.
    SelectedSubspaceResponseLayout cluster_layout(
        3,
        std::vector<SelectedStateCluster>{{2, 0.25}});
    const int n_orbitals = 3;
    const int n_response = cluster_layout.response_size();
    Eigen::Matrix3d orbital_hessian;
    orbital_hessian << 3.0, 0.2, -0.1,
                       0.2, 2.4, 0.3,
                      -0.1, 0.3, 1.8;
    Eigen::Matrix3d orbital_metric;
    orbital_metric << 1.4, 0.1, 0.0,
                      0.1, 1.1, 0.1,
                      0.0, 0.1, 0.9;

    const Eigen::Matrix3d overlap =
        (Eigen::Vector3d(1.0, 1.0, 2.0)).asDiagonal();
    const Eigen::Matrix3d hamiltonian =
        (Eigen::Vector3d(1.0, 2.0, 8.0)).asDiagonal();
    Eigen::Matrix<double, 3, 2> selected =
        Eigen::Matrix<double, 3, 2>::Zero();
    selected(0, 0) = 1.0;
    selected(1, 1) = 1.0;
    const Eigen::Matrix2d selected_energies =
        (Eigen::Vector2d(1.0, 2.0)).asDiagonal();

    const auto response_action =
        [cluster_layout, overlap, hamiltonian, selected, selected_energies](
            const Eigen::Ref<const Eigen::MatrixXd>& x) {
          Eigen::MatrixXd images(cluster_layout.response_size(), x.cols());
          for (Eigen::Index column = 0; column < x.cols(); ++column) {
            const ClusterResponse response =
                cluster_layout.unpack(x.col(column)).front();
            ClusterResponse image;
            image.coefficients =
                hamiltonian * response.coefficients -
                overlap * response.coefficients * selected_energies +
                overlap * selected * response.multipliers;
            image.multipliers =
                selected.transpose() * overlap * response.coefficients;
            images.col(column) = cluster_layout.pack({image});
          }
          return images;
        };
    const Eigen::MatrixXd response_hessian = response_action(
        Eigen::MatrixXd::Identity(n_response, n_response));
    if ((response_hessian - response_hessian.transpose()).norm() > 1.0e-14 ||
        !response_hessian.fullPivLu().isInvertible()) {
      throw std::runtime_error(
          "bordered selected-subspace response block is not symmetric and invertible");
    }

    // Build Bp from synthetic directional integral matrices:
    // [(delta H - delta S Lambda) C; C^T delta S C / 2].
    Eigen::MatrixXd coupling(n_response, n_orbitals);
    for (int orbital = 0; orbital < n_orbitals; ++orbital) {
      Eigen::Matrix3d delta_hamiltonian;
      Eigen::Matrix3d delta_overlap;
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          delta_hamiltonian(row, column) =
              0.03 * (orbital + 1) * (row + column + 2);
          delta_overlap(row, column) =
              0.002 * (orbital + 2) * (row + column + 1);
        }
      }
      ClusterResponse forcing;
      forcing.coefficients =
          delta_hamiltonian * selected -
          delta_overlap * selected * selected_energies;
      forcing.multipliers =
          0.5 * selected.transpose() * delta_overlap * selected;
      coupling.col(orbital) = cluster_layout.pack({forcing});
    }

    CoupledNewtonOperator op(
        n_orbitals,
        cluster_layout,
        CoupledNewtonActions{
            [orbital_hessian](const auto& x) {
              return (orbital_hessian * x).eval();
            },
            [coupling](const auto& x) { return (coupling * x).eval(); },
            [coupling](const auto& x) {
              return (coupling.transpose() * x).eval();
            },
            response_action,
            [orbital_metric](const auto& x) {
              return (orbital_metric * x).eval();
            }});

    const double shift = 0.35;
    Eigen::MatrixXd explicit_operator = Eigen::MatrixXd::Zero(
        op.size(), op.size());
    explicit_operator.topLeftCorner(n_orbitals, n_orbitals) =
        orbital_hessian + shift * orbital_metric;
    explicit_operator.topRightCorner(n_orbitals, n_response) =
        coupling.transpose();
    explicit_operator.bottomLeftCorner(n_response, n_orbitals) = coupling;
    explicit_operator.bottomRightCorner(n_response, n_response) =
        response_hessian;

    Eigen::MatrixXd probes(op.size(), 2);
    probes.col(0) = Eigen::VectorXd::LinSpaced(op.size(), -0.4, 0.7);
    probes.col(1) = Eigen::VectorXd::LinSpaced(op.size(), 0.9, -0.2);
    require_close(
        op.apply_block(probes, shift),
        explicit_operator * probes,
        1.0e-13,
        "coupled block action disagrees with the explicit reference");
    require_close(
        op.apply(probes.col(0), shift),
        explicit_operator * probes.col(0),
        1.0e-13,
        "coupled vector action disagrees with the explicit reference");

    const Eigen::Vector3d orbital_probe(0.3, -0.5, 0.7);
    const Eigen::VectorXd response_probe = probes.col(0).tail(n_response);
    if (op.coupling_adjoint_error(orbital_probe, response_probe) > 1.0e-14) {
      throw std::runtime_error("coupled blocks are not adjoint consistent");
    }

    // Eliminating the response block must recover the relaxed orbital
    // Hessian A-B^T C^{-1}B, including the orbital-only trust shift.
    const Eigen::MatrixXd relaxed =
        orbital_hessian - coupling.transpose() *
            response_hessian.fullPivLu().solve(coupling) +
        shift * orbital_metric;
    const Eigen::Vector3d orbital_direction(0.4, -0.2, 0.6);
    const Eigen::VectorXd response_direction =
        -response_hessian.fullPivLu().solve(coupling * orbital_direction);
    Eigen::VectorXd coupled_direction(op.size());
    coupled_direction.head(n_orbitals) = orbital_direction;
    coupled_direction.tail(n_response) = response_direction;
    const Eigen::VectorXd coupled_image = op.apply(coupled_direction, shift);
    require_close(
        coupled_image.head(n_orbitals),
        relaxed * orbital_direction,
        1.0e-13,
        "coupled elimination did not recover the relaxed orbital Hessian");
    if (coupled_image.tail(n_response).norm() > 1.0e-13) {
      throw std::runtime_error("eliminated response equation is not zero");
    }

    std::cout << "Coupled orbital--subspace Newton operator: passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
