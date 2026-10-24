#include "output/text/normalization.hpp"

#include <cmath>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_available_norms() {
    xmvb::vb::VbScfOptimizerResult result;
    result.scf_result.n_structures = 3;
    result.scf_result.structure_overlap_diagonal = {4.0, 9.0, 16.0};
    result.scf_result.structure_overlap_diagonal_exact = true;
    const auto cached = xmvb::output::structure_norms(result);
    require((cached.values - Eigen::Vector3d(2.0, 3.0, 4.0)).norm() == 0.0,
            "report failed to reuse exact diagonal without an orbital/CI rebuild");
    require(std::abs(cached.mean_overlap - 29.0 / 3.0) < 1e-14,
            "cached mean structure overlap is wrong");

    result.scf_result.structure_overlap_diagonal_exact = false;
    result.scf_result.structure_overlap_diagonal.assign(3, 99.0);
    Eigen::Matrix3d overlap = Eigen::Vector3d(4.0, 9.0, 16.0).asDiagonal();
    result.scf_result.structure_matrices.overlap_matrix.assign(overlap.data(),
                                                               overlap.data() + overlap.size());
    const auto available = xmvb::output::structure_norms(result);
    require((available.values - cached.values).norm() == 0.0,
            "report rebuilt norms rather than reusing an available exact matrix");
}

void check_selected_open_shell() {
    xmvb::vb::VbScfOptimizerResult result;
    result.scf_result.n_structures = 3;
    result.scf_result.structure_overlap_diagonal.assign(3, 99.0); // Not exact.
    auto& data = result.optimized_input.structure_data;
    data.n_structures = 3;
    data.alpha_strings = {{0, 1}, {0, 2}, {1, 2}};
    data.beta_strings = {{0}, {1}, {2}};
    data.pair_alpha = {0, 1, 2};
    data.pair_beta = {0, 1, 2};
    data.offsets = {0, 2, 4, 6};
    data.terms = {{0, 1.0}, {1, -0.5}, {0, 0.3}, {1, 0.8}, {1, -0.2}, {2, 1.3}};
    result.optimized_input.orbital_preparation_input.n_active_orbitals = 3;
    Eigen::Matrix3d s;
    s << 1.1, 0.2, -0.1, 0.2, 1.3, 0.15, -0.1, 0.15, 0.9;
    result.active_orbital_overlap_matrix.assign(s.data(), s.data() + s.size());
    // There is deliberately no AO/orbital/Hamiltonian input. The accepted
    // three-by-three overlap must suffice, even if export uses another gauge.
    const auto norms = xmvb::output::structure_norms(result);
    Eigen::Matrix3d product_overlap;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const auto& a = data.alpha()[i];
            const auto& b = data.alpha()[j];
            product_overlap(i, j) =
                (s(a[0], b[0]) * s(a[1], b[1]) - s(a[0], b[1]) * s(a[1], b[0])) *
                s(data.beta()[i][0], data.beta()[j][0]);
        }
    Eigen::Matrix3d w;
    w << 1.0, -0.5, 0.0, 0.3, 0.8, 0.0, 0.0, -0.2, 1.3;
    const Eigen::Matrix3d m = w.transpose() * product_overlap * w;
    require((norms.values.array().square().matrix() - m.diagonal()).norm() < 2e-14,
            "on-demand norms differ from independent selected-space overlap");
    require(std::abs(norms.mean_overlap - m.trace() / 3.0) < 1e-14,
            "on-demand mean overlap uses the approximate solver diagonal");
    require(!result.scf_result.structure_overlap_diagonal_exact &&
                result.scf_result.structure_overlap_diagonal.front() == 99.0,
            "output normalization mutated the solver's preconditioner data");
    // For two states, the same norms preserve the wavefunction and both
    // coefficient-based weights. No closure of the selected space is assumed.
    for (int state = 0; state < 2; ++state) {
        Eigen::Vector3d c(0.2 + state, -0.7, 0.4);
        c /= std::sqrt(c.dot(m * c));
        const Eigen::Matrix3d normalized_m =
            norms.values.cwiseInverse().asDiagonal() * m * norms.values.cwiseInverse().asDiagonal();
        const Eigen::Vector3d normalized_c = c.array() * norms.values.array();
        require(std::abs(normalized_c.dot(normalized_m * normalized_c) - 1.0) < 1e-14,
                "normalized structure coefficients changed the wavefunction norm");
        const Eigen::Vector3d raw_weights = c.array() * (m * c).array();
        const Eigen::Vector3d normalized_weights =
            normalized_c.array() * (normalized_m * normalized_c).array();
        require((raw_weights - normalized_weights).norm() < 1e-14,
                "Coulson-Chirgwin weights changed under report normalization");
    }
}

} // namespace

int main() {
    check_available_norms();
    check_selected_open_shell();
}
