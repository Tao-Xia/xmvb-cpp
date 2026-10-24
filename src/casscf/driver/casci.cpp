#include "casscf/driver/casci.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/spin.hpp"

namespace xmvb::casscf {
namespace {

// Keep the resource check ahead of determinant-string and product-carrier
// construction for the integrals entry point.
linalg::DavidsonOptions casci_budget(int dimension, const CasciOptions& options) {
    if (dimension < 1 || options.n_roots < 1)
        throw std::invalid_argument("invalid pure-spin CASCI problem");
    constexpr std::size_t sigma_bytes = 64ULL * 1024 * 1024;
    const std::size_t bytes_per_vector = sizeof(double) * static_cast<std::size_t>(dimension);
    const std::size_t fixed_vectors = 12ULL + 8ULL * options.n_roots;
    if (options.workspace_bytes <= sigma_bytes ||
        fixed_vectors > (options.workspace_bytes - sigma_bytes) / bytes_per_vector)
        throw std::length_error("CASCI exceeds its solver-workspace budget");
    const auto available =
        (options.workspace_bytes - sigma_bytes) / bytes_per_vector - fixed_vectors;
    const int maximum_basis = static_cast<int>(std::min<std::size_t>(dimension, available / 4));
    auto budget = linalg::make_davidson_options(dimension,
                                                options.n_roots,
                                                options.energy_tolerance,
                                                options.residual_tolerance,
                                                options.stationarity_tolerance);
    budget.max_subspace_dimension = std::min(budget.max_subspace_dimension, maximum_basis);
    if (budget.max_subspace_dimension < options.n_roots)
        throw std::length_error("CASCI budget leaves no Davidson expansion space");
    budget.identity_overlap = true;
    return budget;
}

} // namespace

CasciResult
solve_casci(const CasIntegrals& integrals, int electrons, int spin, const CasciOptions& options) {
    const int n = static_cast<int>(integrals.active.one_electron.rows());
    if (n < 1 || n > 63 || electrons < 1 || electrons > 2 * n || spin < 0 ||
        spin > std::min(electrons, 2 * n - electrons) || (electrons - spin) % 2 ||
        options.n_roots < 1 || !std::isfinite(integrals.core_energy))
        throw std::invalid_argument("invalid pure-spin CASCI problem");
    const int na = (electrons + spin) / 2, nb = electrons - na;
    const auto a = mcscf::string_count(n, na), b = mcscf::string_count(n, nb);
    if (b && a > static_cast<std::size_t>(std::numeric_limits<int>::max()) / b)
        throw std::length_error("CASCI product space exceeds index range");
    const int dimension = static_cast<int>(a * b);
    (void)casci_budget(dimension, options);
    mcscf::SpinSpace space(n, na, nb);
    mcscf::DirectCiSigmaAction sigma(space.alpha(), space.beta(), integrals.active);
    return solve_casci(
        sigma, space, integrals.active.one_electron, spin, integrals.core_energy, options);
}

CasciResult solve_casci(const mcscf::DirectCiSigmaAction& sigma,
                        const mcscf::SpinSpace& space,
                        const Eigen::MatrixXd& one_electron,
                        int spin,
                        double core_energy,
                        const CasciOptions& options,
                        const Eigen::MatrixXd& initial_guess) {
    const auto na = space.alpha().size(), nb = space.beta().size();
    if (options.n_roots < 1 || !std::isfinite(core_energy) || sigma.n_alpha_strings() != na ||
        sigma.n_beta_strings() != nb || na < 1 || nb < 1 ||
        static_cast<std::size_t>(na) >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) / nb)
        throw std::invalid_argument("invalid operator-backed pure-spin CASCI problem");
    const int dimension = na * nb;
    auto budget = casci_budget(dimension, options);
    const int spin_dimension = space.dimension(spin);
    if (options.n_roots > spin_dimension)
        throw std::invalid_argument("more CASCI roots requested than the target-spin dimension");
    if (budget.max_subspace_dimension < options.n_roots ||
        (spin_dimension > options.n_roots && budget.max_subspace_dimension <= options.n_roots))
        throw std::length_error("CASCI budget leaves no Davidson expansion space");

    const Eigen::MatrixXd diagonal = sigma.diagonal(one_electron);
    const Eigen::Map<const Eigen::VectorXd> hdiag(diagonal.data(), dimension);
    const Eigen::VectorXd sdiag = Eigen::VectorXd::Ones(dimension);
    const auto project = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        const Eigen::Map<const Eigen::MatrixXd> carrier(x.data(), na, nb);
        const Eigen::MatrixXd projected = space.project(carrier, spin);
        return Eigen::Map<const Eigen::VectorXd>(projected.data(), dimension);
    };
    // Follow the common Davidson initial-block convention. A warm eigenvector
    // alone may be an excited root in a disconnected invariant sector; retain
    // independent low-diagonal probes instead of treating its zero residual as
    // sufficient evidence for selecting the lowest root.
    const int n_initial =
        std::min({budget.max_subspace_dimension, spin_dimension, 2 * options.n_roots});
    Eigen::MatrixXd initial(dimension, n_initial);
    int admitted = 0;
    const auto admit = [&](Eigen::VectorXd seed) {
        seed = project(seed);
        for (int pass = 0; pass < 2; ++pass)
            for (int j = 0; j < admitted; ++j)
                seed.noalias() -= initial.col(j).dot(seed) * initial.col(j);
        const double norm = seed.norm();
        if (norm <= 64 * std::numeric_limits<double>::epsilon())
            return;
        initial.col(admitted++) = seed / norm;
    };
    if (initial_guess.rows() != 0 || initial_guess.cols() != 0) {
        if (initial_guess.rows() != na || initial_guess.cols() < nb ||
            initial_guess.cols() % nb != 0 || initial_guess.cols() / nb > options.n_roots ||
            !initial_guess.allFinite())
            throw std::invalid_argument("invalid CASCI initial coefficient block");
        for (int root = 0; root < initial_guess.cols() / nb; ++root) {
            const Eigen::Map<const Eigen::VectorXd> seed(initial_guess.data() + root * dimension,
                                                         dimension);
            admit(seed);
        }
    }
    if (admitted < n_initial) {
        std::vector<int> order(dimension);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(
            order.begin(), order.end(), [&](int i, int j) { return hdiag[i] < hdiag[j]; });
        for (int i : order) {
            Eigen::VectorXd seed = Eigen::VectorXd::Zero(dimension);
            seed[i] = 1.0;
            admit(std::move(seed));
            if (admitted == n_initial)
                break;
        }
    }
    if (admitted != n_initial)
        throw std::runtime_error("CASCI spin-projected guesses are rank deficient");
    budget.residual_preconditioner = [&](const Eigen::VectorXd& r, double energy) {
        Eigen::VectorXd denominator = (hdiag.array() - energy).abs();
        const double floor = std::sqrt(std::numeric_limits<double>::epsilon()) *
                             std::max({1.0, hdiag.cwiseAbs().maxCoeff(), std::abs(energy)});
        denominator = denominator.cwiseMax(floor);
        return project(r.cwiseQuotient(denominator));
    };
    const linalg::GeneralizedEigenAction action = [&](const Eigen::Ref<const Eigen::MatrixXd>& x) {
        const Eigen::Map<const Eigen::MatrixXd> carrier(x.data(), na, nb * x.cols());
        const Eigen::MatrixXd hx = sigma.apply(carrier);
        linalg::GeneralizedEigenActionResult images;
        images.hamiltonian = Eigen::Map<const Eigen::MatrixXd>(hx.data(), dimension, x.cols());
        return images;
    };
    CasciResult result;
    result.core_energy = core_energy;
    result.active_ci =
        linalg::GeneralizedEigensolver{}.solve_davidson(action, hdiag, sdiag, initial, budget);
    const auto& vectors = result.active_ci.eigenpairs.eigenvector_matrix;
    for (int root = 0; root < options.n_roots; ++root) {
        const Eigen::Map<const Eigen::MatrixXd> x(vectors.data() + root * dimension, na, nb);
        const Eigen::MatrixXd sx = space.apply_squared(x);
        const double target = 0.25 * spin * (spin + 2);
        const double spin_error = (sx - target * x).norm();
        if (spin_error > 256 * std::numeric_limits<double>::epsilon() * dimension)
            throw std::runtime_error("CASCI eigenvector left the requested total-spin space");
        result.spin_squared.push_back(x.cwiseProduct(sx).sum());
    }
    return result;
}

} // namespace xmvb::casscf
