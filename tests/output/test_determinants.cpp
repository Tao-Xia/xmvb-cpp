#include "output/text/determinants.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
void check_rows(bool shared) {
    xmvb::vb::StructureExpansion data;
    data.n_structures = 2;
    data.alpha_strings = {{0, 1}, {0, 2}, {1, 2}};
    data.beta_strings = shared ? data.alpha_strings : xmvb::mcscf::StringTable{{0}, {2}};
    data.pair_alpha = {0, 2, 1, 0, 0, 1};
    data.pair_beta = shared ? std::vector<xmvb::mcscf::StringIndex>{2, 0, 2, 1, 0, 1}
                            : std::vector<xmvb::mcscf::StringIndex>{0, 1, 0, 1, 0, 1};
    data.offsets = {0, 1, 2, 4, 5, 6, 6};
    data.terms = {{1, 2.0}, {0, 1.0}, {0, -1.0}, {1, 2.0}, {0, 0.3}, {0, 0.7}};
    const std::array<std::size_t, 6> order{1, 2, 3, 4, 0, 5};
    const std::array<const char*, 3> alpha{"1 2-3", "1 2 4", "1 3-4"};
    const std::array<const char*, 2> beta{"1 2", "1 4"};
    xmvb::output::DeterminantReport report(data, 1);
    for (double scale : {1.0, -2.0}) {
        const Eigen::Vector2d coefficients = scale * Eigen::Vector2d(0.5, -0.25);
        std::ostringstream expected, actual;
        expected << "\n\n          ******  UNNORMALIZED DETERMINANT COEFFICIENTS  ******\n\n"
                    "                                     INA   A\n"
                    "                                     INA   B\n";
        const std::array<double, 6> values{0.5, -1.0, 0.15, 0.35, -0.5, 0.0};
        for (std::size_t display = 0; display < order.size(); ++display) {
            const auto pair = order[display];
            // An empty expansion row sums to positive zero, independent of state phase.
            const double value = display == 5 ? 0.0 : scale * values[display];
            expected << std::setw(8) << display + 1 << std::fixed << std::setprecision(8)
                     << std::setw(17) << value << "  ******    " << alpha[data.pair_alpha[pair]]
                     << '\n'
                     << std::string(37, ' ')
                     << (shared ? alpha[data.pair_beta[pair]] : beta[data.pair_beta[pair]]) << '\n';
        }
        actual << std::scientific << std::setprecision(16);
        const auto flags = actual.flags();
        report.write(actual, coefficients);
        if (actual.str() != expected.str())
            throw std::runtime_error(
                "cached spin text changed determinant rows, stable ties or coefficients");
        if (actual.flags() != flags || actual.precision() != 16)
            throw std::runtime_error("determinant output changed the caller's number format");
    }
}

void check_one_off() {
    xmvb::vb::StructureExpansion data;
    data.n_structures = 1;
    data.alpha_strings = {{0, 2}};
    data.beta_strings = {{}};
    data.pair_alpha = data.pair_beta = {0};
    data.offsets = {0, 1};
    data.terms = {{0, 1.0}};
    xmvb::output::DeterminantReport report(data, 2);
    std::ostringstream text;
    const Eigen::VectorXd c = Eigen::VectorXd::Constant(1, 0.25);
    report.write(text, c);
    if (text.str().find("1-2 3 5\n" + std::string(37, ' ') + "1-2\n") == std::string::npos)
        throw std::runtime_error("one-off or empty spin string was formatted incorrectly");
}

void check_original_structure_coefficients() {
    // One raw singlet is D_ab - D_ba and has squared norm two in an
    // orthonormal orbital basis. Its normalized structure coefficient is one,
    // but the original-basis coefficient passed to the expansion is 1/sqrt(2).
    xmvb::vb::StructureExpansion data;
    data.n_structures = 1;
    data.alpha_strings = data.beta_strings = {{0}, {1}};
    data.pair_alpha = {0, 1};
    data.pair_beta = {1, 0};
    data.offsets = {0, 1, 2};
    data.terms = {{0, 1.0}, {0, -1.0}};
    xmvb::output::DeterminantReport report(data, 0);
    const Eigen::VectorXd raw = Eigen::VectorXd::Constant(1, std::sqrt(0.5));
    std::ostringstream text;
    report.write(text, raw);
    if (text.str().find("-0.70710678") == std::string::npos ||
        text.str().find(" 0.70710678") == std::string::npos)
        throw std::runtime_error("determinant amplitudes lost original-structure normalization");
}
} // namespace

int main() {
    check_rows(false);
    check_rows(true);
    check_one_off();
    check_original_structure_coefficients();
}
