#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "vbscf/optimization/coupled/preconditioner.hpp"
#include "vbscf/optimization/krylov/minres.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

double numerical_floor(double scale, int dimension) {
  return std::max(
      std::numeric_limits<double>::epsilon() *
          std::max(1, dimension) * scale,
      std::numeric_limits<double>::min());
}

}  // namespace

int main() {
  try {
    using xmvb::vb::SelectedStateCluster;
    using xmvb::vb::SelectedSubspaceResponseLayout;
    using xmvb::vb::StructureResponsePreconditionerData;

    constexpr int n_orbitals = 6;
    constexpr int n_structures = 8;
    const SelectedSubspaceResponseLayout layout(
        n_structures,
        {SelectedStateCluster{2, 0.125}, SelectedStateCluster{1, 0.5}});
    StructureResponsePreconditionerData data;
    data.hamiltonian_diagonal =
        Eigen::VectorXd::LinSpaced(n_structures, 2.0, 9.0);
    data.overlap_diagonal =
        Eigen::VectorXd::LinSpaced(n_structures, 0.8, 1.2);
    data.selected_energies.resize(3);
    data.selected_energies << 0.2, 0.7, 1.1;
    data.overlap_selected.resize(n_structures, 3);
    for (int row = 0; row < n_structures; ++row) {
      const double x = static_cast<double>(row + 1) / n_structures;
      data.overlap_selected(row, 0) = 1.0 + 0.1 * x;
      data.overlap_selected(row, 1) = x - 0.2;
      data.overlap_selected(row, 2) = 0.7 - 0.3 * x;
    }

    Eigen::VectorXd orbital_diagonal(n_orbitals);
    orbital_diagonal << 0.03, 0.2, 1.0, 7.0, 40.0, 300.0;
    const auto inverse =
        xmvb::vb::make_coupled_block_inverse_preconditioner(
            n_orbitals,
            layout,
            data,
            [orbital_diagonal](const Eigen::VectorXd& residual) {
              return (residual.array() / orbital_diagonal.array()).matrix();
            });
    const auto response_inverse =
        xmvb::vb::make_structure_response_inverse_preconditioner(
            layout,
            data);

    // Build the exact dense SPD model represented by the matrix-free inverse.
    const int size = n_orbitals + layout.response_size();
    Eigen::MatrixXd preconditioner = Eigen::MatrixXd::Zero(size, size);
    preconditioner.topLeftCorner(n_orbitals, n_orbitals) =
        orbital_diagonal.asDiagonal();
    Eigen::MatrixXd saddle = Eigen::MatrixXd::Zero(size, size);
    saddle.topLeftCorner(n_orbitals, n_orbitals) =
        orbital_diagonal.asDiagonal();
    int first_state = 0;
    for (int cluster = 0; cluster < layout.n_clusters(); ++cluster) {
      const int width = layout.cluster(cluster).n_states;
      const Eigen::MatrixXd selected_overlap =
          data.overlap_selected.middleCols(first_state, width);
      for (int state = 0; state < width; ++state) {
        const double energy = data.selected_energies[first_state + state];
        const Eigen::VectorXd gap =
            data.hamiltonian_diagonal - energy * data.overlap_diagonal;
        const double scale = std::max(
            data.hamiltonian_diagonal.cwiseAbs().maxCoeff(),
            std::abs(energy) * data.overlap_diagonal.cwiseAbs().maxCoeff());
        const Eigen::VectorXd diagonal = gap.cwiseAbs().cwiseMax(
            numerical_floor(scale, n_structures));
        const Eigen::MatrixXd schur =
            selected_overlap.transpose() *
            diagonal.cwiseInverse().asDiagonal() * selected_overlap;
        const int coefficient = n_orbitals +
            layout.coefficient_offset(cluster) + state * n_structures;
        const int multiplier = n_orbitals +
            layout.multiplier_offset(cluster) + state * width;
        preconditioner.block(
            coefficient, coefficient, n_structures, n_structures) =
            diagonal.asDiagonal();
        preconditioner.block(
            coefficient, multiplier, n_structures, width) = selected_overlap;
        preconditioner.block(
            multiplier, coefficient, width, n_structures) =
            selected_overlap.transpose();
        preconditioner.block(multiplier, multiplier, width, width) =
            2.0 * schur;

        saddle.block(
            coefficient, coefficient, n_structures, n_structures) =
            diagonal.asDiagonal();
        saddle.block(
            coefficient, multiplier, n_structures, width) = selected_overlap;
        saddle.block(
            multiplier, coefficient, width, n_structures) =
            selected_overlap.transpose();
      }
      first_state += width;
    }

    Eigen::VectorXd probe = Eigen::VectorXd::LinSpaced(size, -0.8, 1.1);
    const Eigen::VectorXd expected =
        preconditioner.ldlt().solve(probe);
    require(
        (inverse(probe) - expected).stableNorm() <=
            2.0e-11 * std::max(1.0, expected.stableNorm()),
        "coupled inverse disagrees with its dense SPD factorization");
    const Eigen::VectorXd response_probe = probe.tail(layout.response_size());
    require(
        (response_inverse(response_probe) -
         expected.tail(layout.response_size())).stableNorm() <=
            2.0e-11 * std::max(
                1.0,
                expected.tail(layout.response_size()).stableNorm()),
        "response-only inverse disagrees with the coupled response block");
    for (int sample = 1; sample <= 4; ++sample) {
      const Eigen::VectorXd direction =
          Eigen::VectorXd::LinSpaced(
              size, -0.3 * sample, 0.7 + 0.1 * sample)
              .array()
              .sin();
      require(
          direction.dot(inverse(direction)) > 0.0,
          "coupled inverse preconditioner is not positive definite");
      const Eigen::VectorXd response_direction =
          direction.tail(layout.response_size());
      require(
          response_direction.dot(response_inverse(response_direction)) > 0.0,
          "response-only inverse preconditioner is not positive definite");
    }

    bool rejected_wrong_dimension = false;
    try {
      static_cast<void>(response_inverse(
          Eigen::VectorXd::Zero(layout.response_size() + 1)));
    } catch (const std::invalid_argument&) {
      rejected_wrong_dimension = true;
    }
    require(
        rejected_wrong_dimension,
        "response-only inverse accepted the wrong coordinate dimension");

    xmvb::vb::MinresOptions options;
    options.relative_residual_tolerance = 2.0e-11;
    options.maximum_iterations = 4 * size;
    const Eigen::VectorXd rhs =
        Eigen::VectorXd::LinSpaced(size, -0.9, 0.6).array().cos();
    const auto identity_result = xmvb::vb::solve_symmetric_minres(
        [saddle](const Eigen::VectorXd& vector) {
          return (saddle * vector).eval();
        },
        rhs,
        options);
    const auto preconditioned_result = xmvb::vb::solve_symmetric_minres(
        [saddle](const Eigen::VectorXd& vector) {
          return (saddle * vector).eval();
        },
        rhs,
        options,
        inverse);
    require(identity_result.converged() && preconditioned_result.converged(),
            "synthetic coupled MINRES solve did not converge");
    require(
        preconditioned_result.operator_actions <
            identity_result.operator_actions,
        "coupled block inverse did not reduce operator actions");

    std::cout << "coupled Newton preconditioner: passed (identity actions="
              << identity_result.operator_actions
              << ", block actions="
              << preconditioned_result.operator_actions << ")\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
