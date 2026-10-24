#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/orbital/preparation.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {

using xmvb::vb::AcceptedOrbitalPreparationCache;
using xmvb::vb::InactiveDensityProjection;
using xmvb::vb::OrbitalPreparationInput;

OrbitalPreparationInput make_input(bool with_inactive) {
    OrbitalPreparationInput input;
    input.n_basis_functions = 5;
    input.n_active_orbitals = 2;
    input.n_active_electrons = 2;
    input.n_total_electrons = with_inactive ? 6 : 2;
    input.n_orbitals = with_inactive ? 4 : 2;
    input.orbital_type = 1;

    Eigen::MatrixXd perturbation(5, 5);
    for (int column = 0; column < 5; ++column)
        for (int row = 0; row < 5; ++row)
            perturbation(row, column) = std::sin(0.37 * (row + 1) * (column + 2));
    input.ao_overlap_matrix =
        Eigen::MatrixXd::Identity(5, 5) + 0.08 * perturbation.transpose() * perturbation;

    const std::vector<std::vector<int>> supports = {
        {1, 2, 4, 5}, {1, 3, 4, 5}, {1, 2, 5}, {2, 3, 4, 5}};
    const std::vector<std::vector<double>> values = {
        {1.2, 0.3, -0.2, 0.1}, {0.35, 1.1, 0.25, -0.1}, {-0.2, 1.3, 0.15}, {0.1, -0.3, 1.2, 0.2}};
    const int first = with_inactive ? 0 : 2;
    for (int orbital = first; orbital < 4; ++orbital) {
        input.orbital_basis_counts.push_back(supports[orbital].size());
        for (std::size_t slot = 0; slot < supports[orbital].size(); ++slot) {
            input.orbital_basis_index_table.push_back(supports[orbital][slot]);
            input.orbital_value_table.push_back(values[orbital][slot]);
        }
        while (input.orbital_basis_index_table.size() <
               input.orbital_basis_counts.size() * input.n_basis_functions) {
            input.orbital_basis_index_table.push_back(0);
            input.orbital_value_table.push_back(0.0);
        }
    }
    return input;
}

// Independent small-matrix reference: differentiate P = C M^-1 C^T and
// X = (I - P S) C_active for each raw sparse slot explicitly.
std::vector<double> reference_raw_gradient(const OrbitalPreparationInput& input,
                                           const AcceptedOrbitalPreparationCache& cache,
                                           const Eigen::MatrixXd& active_gradient,
                                           const Eigen::MatrixXd& density_gradient) {
    const int n_basis = input.n_basis_functions;
    const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
    const Eigen::MatrixXd& s = input.ao_overlap_matrix;
    const Eigen::MatrixXd inactive = cache.normalized_orbitals.leftCols(n_inactive);
    const Eigen::MatrixXd active =
        cache.normalized_orbitals.middleCols(n_inactive, input.n_active_orbitals);
    const Eigen::MatrixXd inverse_gram =
        n_inactive > 0 ? (inactive.transpose() * s * inactive)
                             .ldlt()
                             .solve(Eigen::MatrixXd::Identity(n_inactive, n_inactive))
                             .eval()
                       : Eigen::MatrixXd(0, 0);
    const Eigen::MatrixXd duals = inactive * inverse_gram;
    const Eigen::MatrixXd projector =
        Eigen::MatrixXd::Identity(n_basis, n_basis) - duals * inactive.transpose() * s;
    std::vector<double> result(input.orbital_value_table.size(), 0.0);
    for (int orbital = 0; orbital < input.n_orbitals; ++orbital) {
        const Eigen::VectorXd normalized = cache.normalized_orbitals.col(orbital);
        const Eigen::VectorXd s_normalized = s * normalized;
        for (int slot = 0; slot < input.orbital_basis_counts[orbital]; ++slot) {
            const int row = input.orbital_basis_index_table[orbital * n_basis + slot] - 1;
            Eigen::VectorXd direction = Eigen::VectorXd::Zero(n_basis);
            direction[row] = 1.0;
            direction = cache.inverse_norms[orbital] * (direction - normalized * s_normalized[row]);
            double derivative = 0.0;
            if (orbital < n_inactive) {
                Eigen::MatrixXd delta_inactive = Eigen::MatrixXd::Zero(n_basis, n_inactive);
                delta_inactive.col(orbital) = direction;
                const Eigen::MatrixXd delta_gram = delta_inactive.transpose() * s * inactive +
                                                   inactive.transpose() * s * delta_inactive;
                const Eigen::MatrixXd delta_duals =
                    delta_inactive * inverse_gram - duals * delta_gram * inverse_gram;
                const Eigen::MatrixXd delta_density =
                    delta_duals * inactive.transpose() + duals * delta_inactive.transpose();
                const Eigen::MatrixXd delta_active_auxiliary = -delta_density * s * active;
                derivative = (density_gradient.array() * delta_density.array()).sum() +
                             (active_gradient.array() * delta_active_auxiliary.array()).sum();
            } else {
                derivative = active_gradient.col(orbital - n_inactive).dot(projector * direction);
            }
            result[orbital * n_basis + slot] = derivative;
        }
    }
    return result;
}

void check(bool with_inactive) {
    const OrbitalPreparationInput input = make_input(with_inactive);
    const int n_basis = input.n_basis_functions;
    const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
    Eigen::MatrixXd active_gradient(n_basis, input.n_active_orbitals);
    Eigen::MatrixXd density_gradient(n_basis, n_basis);
    for (int column = 0; column < active_gradient.cols(); ++column)
        for (int row = 0; row < n_basis; ++row)
            active_gradient(row, column) = std::cos(0.29 * (row + 2) * (column + 1));
    for (int column = 0; column < n_basis; ++column)
        for (int row = 0; row < n_basis; ++row)
            density_gradient(row, column) = std::sin(0.17 * (row + 1) * (column + 3));

    const std::vector<double> dense_weight(density_gradient.data(),
                                           density_gradient.data() + density_gradient.size());
    const auto cache =
        xmvb::vb::build_accepted_orbital_preparation_cache(input, active_gradient, dense_weight);
    const auto prepared = xmvb::vb::ActiveSpaceOrbitalPreparer().prepare(input);
    if (with_inactive) {
        const Eigen::MatrixXd c = cache.normalized_orbitals.leftCols(n_inactive);
        const Eigen::MatrixXd gram = c.transpose() * input.ao_overlap_matrix * c;
        if (std::abs(gram(0, 1)) < 1e-3)
            throw std::runtime_error("test inactive frame is accidentally orthogonal");
    }
    const Eigen::MatrixXd projected =
        n_inactive > 0
            ? ((density_gradient + density_gradient.transpose()) * cache.inactive_auxiliary).eval()
            : Eigen::MatrixXd(n_basis, 0);
    const auto full = xmvb::vb::backpropagate_active_space_orbital_gradient(
        active_gradient, density_gradient, input, prepared, cache);
    const auto compressed = xmvb::vb::backpropagate_active_space_orbital_gradient(
        active_gradient, InactiveDensityProjection{projected}, input, prepared, cache);
    const auto reference = reference_raw_gradient(input, cache, active_gradient, density_gradient);
    for (std::size_t slot = 0; slot < full.size(); ++slot) {
        const double scale = std::max(1.0, std::abs(reference[slot]));
        if (std::abs(full[slot] - compressed[slot]) > 1e-12 * scale ||
            std::abs(compressed[slot] - reference[slot]) > 2e-11 * scale)
            throw std::runtime_error(
                "projected orbital pullback disagrees with explicit projector derivative");
    }
}

} // namespace

int main() {
    try {
        check(true);
        check(false);
        std::cout << "projected orbital pullback: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
