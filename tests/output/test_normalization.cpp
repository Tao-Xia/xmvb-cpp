#include "output/text/normalization.hpp"

#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>

#include "output/text/sections.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_cached_structure_weights() {
    const Eigen::Vector2d coefficients(1.0, -0.2);
    Eigen::Matrix2d metric;
    metric << 1.0, 0.5, 0.5, 1.0;
    const Eigen::Vector2d overlap_product = metric * coefficients;
    const auto weights = xmvb::output::coulson_chirgwin_weights(coefficients, overlap_product);
    require((weights - Eigen::Vector2d(15.0 / 14.0, -1.0 / 14.0)).norm() < 1e-14,
            "CC weights must use the off-diagonal overlap and allow negative entries");
    require(std::abs(weights.sum() - 1.0) < 1e-14, "CC weights do not sum to unity");
    require(
        (xmvb::output::coulson_chirgwin_weights(-3 * coefficients, -3 * overlap_product) - weights)
                .norm() < 1e-14,
        "CC weights depend on the CI normalization or phase");
    const Eigen::Vector2d scale(2.0, 3.0);
    require((xmvb::output::coulson_chirgwin_weights(
                 (scale.array() * coefficients.array()).matrix(),
                 (overlap_product.array() / scale.array()).matrix()) -
             weights)
                    .norm() < 1e-14,
            "CC weights depend on individual structure normalization");
    xmvb::vb::RawStructureData structures;
    structures.n_structures = 2;
    structures.n_total_electrons = structures.n_active_electrons = 4;
    structures.spin_multiplicity = 1;
    structures.raw_structure_orbitals = {1, 2, 3, 4, 1, 3, 2, 4};
    std::ostringstream text;
    text << std::scientific << std::unitbuf << std::setprecision(16);
    const auto flags = text.flags();
    xmvb::output::print_coulson_chirgwin_weights(text, coefficients, overlap_product, structures);
    require(text.flags() == flags && text.precision() == 16,
            "weight output changed subsequent energy precision or buffering");
    require(text.str().find("WEIGHTS OF STRUCTURES") != std::string::npos &&
                text.str().find("Coulson-Chirgwin Weights") != std::string::npos &&
                text.str().find("1.07142857") != std::string::npos &&
                text.str().find("-0.07142857") != std::string::npos &&
                text.str().find("1-3") != std::string::npos &&
                text.str().find("1-2") != std::string::npos,
            "GPU/CPU shared weight output lost values or structure topology");
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
    check_cached_structure_weights();
    check_available_norms();
    check_selected_open_shell();
}
