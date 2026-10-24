#include "tools/response_audit.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "linalg/eigen_response.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"

namespace xmvb::tools {

/** @brief Bounded dense reference for coordinate versus physical response spectra.
 * Diagnostic only: the explicit size limit is checked before square allocations.
 * This mode exits without evaluating an orbital HVP or changing production solves.
 */
void audit_structure_response(const xmvb::vb::StructureAction& action,
                              const xmvb::vb::AcceptedPointContext& context,
                              bool complete_space,
                              int limit,
                              const Eigen::MatrixXd& orbital_forcing) {
    const int n = action.n_structures();
    if (n < 2 || n > limit)
        throw std::invalid_argument("structure spectrum exceeds explicit diagnostic size limit");
    const auto matrices = action.apply(Eigen::MatrixXd::Identity(n, n));
    const auto symmetric = [](const Eigen::MatrixXd& matrix) -> Eigen::MatrixXd {
        return 0.5 * (matrix + matrix.transpose());
    };
    const Eigen::MatrixXd h = symmetric(matrices.hamiltonian);
    const Eigen::MatrixXd m = symmetric(matrices.overlap);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> metric_spectrum(m);
    if (metric_spectrum.info() != Eigen::Success || !(metric_spectrum.eigenvalues()[0] > 0.0))
        throw std::runtime_error("diagnostic structure overlap is not positive definite");
    std::cout << "structure_overlap_min = " << metric_spectrum.eigenvalues()[0]
              << "\nstructure_overlap_max = " << metric_spectrum.eigenvalues()[n - 1]
              << "\nstructure_overlap_condition = "
              << metric_spectrum.eigenvalues()[n - 1] / metric_spectrum.eigenvalues()[0] << '\n';
    const Eigen::LLT<Eigen::MatrixXd> root_metric(m);
    if (root_metric.info() != Eigen::Success)
        throw std::runtime_error("diagnostic root metric factorization failed");
    for (Eigen::Index state = 0; state < context.selected_state_eigenvectors.cols(); ++state) {
        const Eigen::VectorXd c = context.selected_state_eigenvectors.col(state);
        const double energy = context.selected_state_energies[state];
        const Eigen::VectorXd residual = (h - energy * m) * c;
        const double norm2 = c.dot(m * c);
        const double pairing = residual.dot(root_metric.solve(residual));
        if (!(norm2 > 0.0) || !(pairing >= 0.0))
            throw std::runtime_error("diagnostic root norm or residual is invalid");
        std::cout << "root_certificate_state = " << state << "\nroot_energy = " << energy
                  << "\nroot_coefficient_norm = " << c.norm()
                  << "\nroot_metric_norm = " << std::sqrt(norm2)
                  << "\nroot_coordinate_residual = " << residual.norm()
                  << "\nroot_coefficient_scaled_residual = " << residual.norm() / c.norm()
                  << "\nroot_physical_residual = " << std::sqrt(pairing / norm2) << '\n';
    }
    // Separate spin recoupling from the exterior orbital transform. Distinct
    // spatial occupation patterns are orthogonal before that transform.
    Eigen::VectorXd first = Eigen::VectorXd::Zero(n);
    first[0] = 1.0;
    const Eigen::MatrixXd first_grid = action.expand_structure_block(first);
    if (first_grid.size() > limit)
        throw std::invalid_argument("spin-product spectrum exceeds explicit diagnostic size limit");
    const Eigen::MatrixXd packed = action.expand_structure_block(Eigen::MatrixXd::Identity(n, n));
    const int n_alpha = static_cast<int>(first_grid.rows());
    const int n_beta = static_cast<int>(first_grid.cols());
    Eigen::MatrixXd expansion(first_grid.size(), n);
    std::map<std::vector<int>, std::vector<int>> occupation_groups;
    for (int column = 0; column < n; ++column) {
        const Eigen::MatrixXd grid = packed.middleCols(column * n_beta, n_beta);
        expansion.col(column) = Eigen::Map<const Eigen::VectorXd>(grid.data(), grid.size());
        std::optional<std::vector<int>> occupation;
        for (int beta = 0; beta < n_beta; ++beta) {
            for (int alpha = 0; alpha < n_alpha; ++alpha) {
                if (grid(alpha, beta) == 0.0)
                    continue;
                std::vector<int> pattern(context.n_active_orbitals, 0);
                for (int orbital :
                     context.same_spin_pair_cache.alpha_reuse_table.unique_determinants[alpha])
                    ++pattern.at(orbital);
                for (int orbital :
                     context.same_spin_pair_cache.beta_reuse_table.unique_determinants[beta])
                    ++pattern.at(orbital);
                if (occupation && *occupation != pattern)
                    throw std::runtime_error(
                        "one structure spans multiple spatial occupation patterns");
                occupation = std::move(pattern);
            }
        }
        if (!occupation)
            throw std::runtime_error("diagnostic structure expansion is empty");
        occupation_groups[*occupation].push_back(column);
    }
    const Eigen::MatrixXd spin_metric = expansion.transpose() * expansion;
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spin_spectrum(spin_metric);
    if (spin_spectrum.info() != Eigen::Success || !(spin_spectrum.eigenvalues()[0] > 0.0))
        throw std::runtime_error("diagnostic spin-recoupling metric is not positive definite");
    std::size_t block_cells = 0;
    std::size_t max_block = 0;
    for (const auto& [pattern, columns] : occupation_groups) {
        block_cells += columns.size() * columns.size();
        max_block = std::max(max_block, columns.size());
    }
    std::cout << "spin_recoupling_condition = "
              << spin_spectrum.eigenvalues()[n - 1] / spin_spectrum.eigenvalues()[0]
              << "\noccupation_block_count = " << occupation_groups.size()
              << "\noccupation_max_block = " << max_block
              << "\noccupation_block_cells = " << block_cells << '\n';
    if (complete_space && action.storage().orthogonal_direct_ci) {
        const std::size_t column_bytes =
            (static_cast<std::size_t>(first_grid.size()) + 4 * n) * sizeof(double);
        const xmvb::vb::StructureMetricInverse metric_inverse(
            action,
            context.same_spin_pair_cache.alpha_reuse_table.unique_determinants,
            context.same_spin_pair_cache.beta_reuse_table.unique_determinants,
            complete_space,
            64 * 1024 * 1024,
            column_bytes);
        const auto inverse_metric = [&](const Eigen::MatrixXd& rhs) {
            return metric_inverse.apply(rhs);
        };
        std::cout << "sparse_metric_factor_bytes = " << metric_inverse.factor_bytes()
                  << "\nsparse_metric_block_count = " << metric_inverse.block_count()
                  << "\nsparse_metric_max_block = " << metric_inverse.max_block_size() << '\n';
        Eigen::MatrixXd rhs(n, 3);
        for (int column = 0; column < rhs.cols(); ++column)
            for (int row = 0; row < n; ++row)
                rhs(row, column) = std::sin(0.17 * (row + 1) * (column + 2));
        const Eigen::MatrixXd inverse = inverse_metric(rhs);
        const Eigen::LLT<Eigen::MatrixXd> reference_factor(m);
        if (reference_factor.info() != Eigen::Success)
            throw std::runtime_error("reference metric factorization failed");
        const Eigen::MatrixXd reference = reference_factor.solve(rhs);
        const double residual = (m * inverse - rhs).norm() / rhs.norm();
        const double reference_error = (inverse - reference).norm() / reference.norm();
        const Eigen::MatrixXd pairings = rhs.transpose() * inverse;
        const double symmetry_error = (pairings - pairings.transpose()).norm() / pairings.norm();
        std::cout << "factorized_metric_inverse_residual = " << residual
                  << "\nfactorized_metric_inverse_reference_error = " << reference_error
                  << "\nfactorized_metric_inverse_symmetry_error = " << symmetry_error << '\n';
        if (!inverse.allFinite() || residual > 1.0e-8 || reference_error > 1.0e-8 ||
            symmetry_error > 1.0e-10 || pairings.diagonal().minCoeff() <= 0.0)
            throw std::runtime_error("factorized structure metric inverse failed its certificate");

        // Ground-root control only. The actual orbital coupling tests the same
        // horizontal equation and relative tolerance with both preconditioners.
        // No dense response inverse enters either operator application.
        if (context.selected_state_indices.size() == 1 && context.selected_state_indices[0] == 0) {
            if (orbital_forcing.rows() != n || orbital_forcing.cols() != 1 ||
                !orbital_forcing.allFinite() || !(orbital_forcing.norm() > 0.0))
                throw std::invalid_argument(
                    "ground response control needs a finite nonzero orbital forcing");
            const Eigen::MatrixXd& rhs = orbital_forcing;
            std::cout << "response_control_rhs_kind = orbital_coupling"
                      << "\nresponse_control_rhs_norm = " << rhs.norm() << '\n';
            const Eigen::VectorXd c = context.selected_state_eigenvectors.col(0);
            const Eigen::VectorXd mc = m * c;
            const Eigen::VectorXd normal = mc.normalized();
            const double root_norm = c.dot(mc);
            const double energy = context.selected_state_energies[0];
            const Eigen::VectorXd ritz = (h - energy * m) * c;
            const auto project = [&](const Eigen::MatrixXd& values) -> Eigen::MatrixXd {
                return values - normal * (normal.transpose() * values);
            };
            const auto precondition = [&](const Eigen::MatrixXd& values) -> Eigen::MatrixXd {
                // Restricted metric inverse: preserve (M c)^T x=0, not c^T x=0.
                return inverse_metric(values) - c * (c.transpose() * values) / root_norm;
            };
            Eigen::MatrixXd x = Eigen::MatrixXd::Zero(n, rhs.cols());
            Eigen::MatrixXd ax = x;
            Eigen::MatrixXd response_residual = project(rhs);
            Eigen::MatrixXd z = precondition(response_residual);
            Eigen::MatrixXd directions = z;
            Eigen::VectorXd rz =
                (response_residual.array() * z.array()).colwise().sum().transpose();
            std::vector<bool> converged(rhs.cols(), false);
            std::vector<int> iterations(rhs.cols(), 0);
            int actions = 0;
            const double tolerance =
                context.structure_solve_accuracy.response_backward_error_tolerance(
                    std::abs(energy));
            const auto bordered_relative = [&](int column, const Eigen::MatrixXd& images) {
                const double multiplier =
                    (c.dot(rhs.col(column)) - ritz.dot(x.col(column))) / root_norm;
                const Eigen::VectorXd top = images.col(column) + mc * multiplier - rhs.col(column);
                return std::hypot(top.norm(), mc.dot(x.col(column))) / rhs.col(column).norm();
            };
            for (int iteration = 0; iteration < n + 1; ++iteration) {
                const auto images = action.apply(directions);
                ++actions;
                const Eigen::MatrixXd raw_images = images.hamiltonian - energy * images.overlap;
                const Eigen::MatrixXd projected_images = project(raw_images);
                for (int column = 0; column < rhs.cols(); ++column) {
                    if (converged[column])
                        continue;
                    const double curvature =
                        directions.col(column).dot(projected_images.col(column));
                    if (!(curvature > 0.0) || !(rz[column] > 0.0))
                        throw std::runtime_error(
                            "ground-root metric-PCG control lost positive curvature");
                    const double step = rz[column] / curvature;
                    x.col(column).noalias() += step * directions.col(column);
                    ax.col(column).noalias() += step * raw_images.col(column);
                    response_residual.col(column).noalias() -= step * projected_images.col(column);
                    iterations[column] = iteration + 1;
                    converged[column] = bordered_relative(column, ax) <= tolerance;
                }
                if (std::all_of(
                        converged.begin(), converged.end(), [](bool value) { return value; }))
                    break;
                z = precondition(response_residual);
                for (int column = 0; column < rhs.cols(); ++column) {
                    if (converged[column]) {
                        directions.col(column).setZero();
                        continue;
                    }
                    const double next_rz = response_residual.col(column).dot(z.col(column));
                    directions.col(column) =
                        z.col(column) + (next_rz / rz[column]) * directions.col(column);
                    rz[column] = next_rz;
                }
            }
            const auto checked = action.apply(x);
            ++actions;
            const Eigen::MatrixXd checked_images = checked.hamiltonian - energy * checked.overlap;
            double max_relative = 0.0;
            for (int column = 0; column < rhs.cols(); ++column) {
                const double relative = bordered_relative(column, checked_images);
                if (!std::isfinite(relative))
                    throw std::runtime_error(
                        "metric-PCG control has a nonfinite bordered residual");
                max_relative = std::max(max_relative, relative);
            }
            std::cout << "factorized_response_iterations = ";
            for (int count : iterations)
                std::cout << count << ',';
            std::cout << "\nfactorized_response_block_actions = " << actions
                      << "\nfactorized_response_bordered_relative_residual = " << max_relative
                      << '\n';
            if (!std::isfinite(max_relative) || max_relative > tolerance)
                throw std::runtime_error(
                    "metric-PCG control failed its independent bordered certificate");
            const xmvb::linalg::GeneralizedEigenAction reference_action =
                [&](const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
                    auto image = action.apply(vectors);
                    return xmvb::linalg::GeneralizedEigenActionResult{std::move(image.hamiltonian),
                                                                      std::move(image.overlap)};
                };
            // Use the identical production MINRES recurrence and certificate. This
            // separates metric conditioning from the preceding PCG control.
            const xmvb::linalg::EigenResponsePreconditioner metric_preconditioner{
                [&](const Eigen::Ref<const Eigen::VectorXd>& residual,
                    Eigen::Index) -> Eigen::VectorXd {
                    const Eigen::MatrixXd column = residual;
                    return precondition(column).col(0);
                },
                1.0 / m.trace()};
            const auto metric_response = xmvb::linalg::solve_generalized_eigen_response(
                reference_action,
                h.diagonal(),
                m.diagonal(),
                Eigen::VectorXd::Constant(rhs.cols(), energy),
                c.replicate(1, rhs.cols()),
                mc.replicate(1, rhs.cols()),
                -rhs,
                Eigen::MatrixXd::Zero(n, rhs.cols()),
                {n + 1, tolerance},
                {},
                nullptr,
                &metric_preconditioner);
            std::cout << "metric_minres_control_iterations = ";
            for (int count : metric_response.iterations)
                std::cout << count << ',';
            std::cout << "\nmetric_minres_control_block_actions = " << metric_response.block_actions
                      << "\nmetric_minres_control_relative_residual = "
                      << metric_response.relative_residual_norms.maxCoeff() << '\n';
            try {
                const auto jacobi_response = xmvb::linalg::solve_generalized_eigen_response(
                    reference_action,
                    h.diagonal(),
                    m.diagonal(),
                    Eigen::VectorXd::Constant(rhs.cols(), energy),
                    c.replicate(1, rhs.cols()),
                    mc.replicate(1, rhs.cols()),
                    -rhs,
                    Eigen::MatrixXd::Zero(n, rhs.cols()),
                    {n + 1, tolerance});
                std::cout << "jacobi_control_iterations = ";
                for (int count : jacobi_response.iterations)
                    std::cout << count << ',';
                std::cout << "\njacobi_control_block_actions = " << jacobi_response.block_actions
                          << "\njacobi_control_relative_residual = "
                          << jacobi_response.relative_residual_norms.maxCoeff() << '\n';
            } catch (const std::runtime_error& error) {
                if (std::string(error.what())
                        .find("projected generalized-eigen response MINRES did not reach") ==
                    std::string::npos)
                    throw;
                std::cout << "jacobi_control_failed = " << error.what() << '\n';
            }
        }
    }
    const auto report = [](const char* name, const Eigen::VectorXd& values) {
        const double smallest = values[0];
        const double largest = values[values.size() - 1];
        std::cout << "RESPONSE_SPECTRUM,kind=" << name << ",min=" << smallest << ",max=" << largest
                  << ",positive_condition="
                  << (smallest > 0.0 ? largest / smallest : std::numeric_limits<double>::infinity())
                  << '\n';
    };
    for (Eigen::Index state = 0; state < context.selected_state_eigenvectors.cols(); ++state) {
        const Eigen::VectorXd c = context.selected_state_eigenvectors.col(state);
        const Eigen::VectorXd normal = (m * c).normalized();
        const Eigen::HouseholderQR<Eigen::MatrixXd> qr(normal);
        const Eigen::MatrixXd complement =
            (qr.householderQ() * Eigen::MatrixXd::Identity(n, n)).rightCols(n - 1);
        const Eigen::MatrixXd shifted = h - context.selected_state_energies[state] * m;
        const Eigen::MatrixXd projected = symmetric(complement.transpose() * shifted * complement);
        const Eigen::MatrixXd projected_metric = symmetric(complement.transpose() * m * complement);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> raw(projected);
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> physical(projected,
                                                                                 projected_metric);
        const Eigen::VectorXd diagonal = shifted.diagonal().cwiseAbs();
        const double sigma = diagonal.maxCoeff();
        const double floor = std::numeric_limits<double>::epsilon() * std::max(1.0, sigma);
        const Eigen::VectorXd inverse =
            (diagonal.array() + sigma * normal.array().square()).max(floor).inverse();
        const Eigen::MatrixXd projected_inverse =
            symmetric(complement.transpose() * inverse.asDiagonal() * complement);
        const Eigen::LLT<Eigen::MatrixXd> factor(projected_inverse);
        if (raw.info() != Eigen::Success || physical.info() != Eigen::Success ||
            factor.info() != Eigen::Success)
            throw std::runtime_error("diagnostic response spectral factorization failed");
        const Eigen::MatrixXd l = factor.matrixL();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> jacobi(
            symmetric(l.transpose() * projected * l));
        if (jacobi.info() != Eigen::Success)
            throw std::runtime_error("diagnostic preconditioned response spectrum failed");
        std::cout << "response_spectrum_state = " << state << '\n';
        report("coefficient", raw.eigenvalues());
        report("jacobi", jacobi.eigenvalues());
        report("overlap_metric", physical.eigenvalues());
    }
}

} // namespace xmvb::tools
