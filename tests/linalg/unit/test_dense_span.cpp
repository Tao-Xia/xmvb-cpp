#include "linalg/eigensolver.hpp"

#include <array>
#include <cmath>
#include <stdexcept>

static_assert(__cplusplus > 202002L, "C++23 is required");

int main() {
    const std::array h{2.0, 0.0, 0.0, 6.0};
    const std::array m{2.0, 0.0, 0.0, 3.0};
    const xmvb::linalg::GeneralizedEigensolver solver;
    const auto result = solver.solve_dense(h, m, 2);
    const auto values = solver.solve_dense_eigenvalues(h, m, 2);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i] - (i + 1.0)) > 1e-14 ||
            std::abs(result.eigenvalues[i] - values[i]) > 1e-14) {
            throw std::runtime_error("span-based generalized eigensolve disagrees");
        }
    }
    bool rejected = false;
    try {
        solver.solve_dense(std::span(h).first(3), m, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("invalid matrix extent was accepted");
}
