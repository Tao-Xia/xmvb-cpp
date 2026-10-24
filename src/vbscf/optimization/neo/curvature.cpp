#include "vbscf/optimization/neo/curvature.hpp"

#include <cmath>
#include <limits>

#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/orbitals/charts/chart.hpp"

namespace xmvb::vb {

mcscf::NeoResult probe_neo_curvature(ExactHvpOperator& orbital_hessian,
                                     const StructureTangentOperator& structure_hessian,
                                     const OrbitalChart& chart) {
    const Eigen::Index n_orbital = chart.reduced_size();
    const Eigen::Index n_structure = structure_hessian.tangent_size();
    const mcscf::QuadraticProblem problem(
        Eigen::VectorXd::Zero(n_orbital + n_structure),
        [&](const Eigen::VectorXd& vector) -> Eigen::VectorXd {
            const auto orbital = orbital_hessian.apply_orbital_coupling(vector.head(n_orbital));
            const auto structure =
                structure_hessian.apply_coupling_coordinates(vector.tail(n_structure));
            Eigen::VectorXd image(vector.size());
            image.head(n_orbital) =
                orbital.orbital_hessian +
                orbital_hessian.apply_structure_coupling_adjoint(structure.coefficient_response,
                                                                 structure.adjoint_multipliers);
            image.tail(n_structure) =
                structure.hessian_coordinates +
                structure_hessian.project_coordinates(orbital.scaled_structure_forcing);
            return image;
        },
        [&](const Eigen::VectorXd& vector) -> Eigen::VectorXd {
            Eigen::VectorXd image = vector;
            image.head(n_orbital) = chart.apply_reduced_curvature(vector.head(n_orbital));
            return image;
        },
        [&](const Eigen::VectorXd& covector) {
            const Eigen::VectorXd orbital_inverse =
                chart.apply_inverse_reduced_block_preconditioner(covector.head(n_orbital));
            return std::sqrt(covector.head(n_orbital).dot(orbital_inverse) +
                             covector.tail(n_structure).squaredNorm());
        },
        [&](const Eigen::VectorXd& vector) -> Eigen::VectorXd {
            Eigen::VectorXd image(vector.size());
            image.head(n_orbital) =
                chart.apply_inverse_reduced_block_preconditioner(vector.head(n_orbital));
            image.tail(n_structure) = structure_hessian.apply_inverse_shifted_preconditioner(
                vector.tail(n_structure), 0.0);
            return image;
        });
    mcscf::NeoOptions options;
    options.stop_on_negative_curvature = true;
    options.relative_residual_tolerance = std::sqrt(std::numeric_limits<double>::epsilon());
    return mcscf::solve_neo(problem, options);
}

} // namespace xmvb::vb
