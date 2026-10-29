#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "linalg/metric_solve.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "integrals/active/two_electron.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "vbscf/structures/overlap/representation.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void combinations(
    int n, int k, int first, std::vector<int>* current, std::vector<std::vector<int>>* result) {
    if (static_cast<int>(current->size()) == k) {
        result->push_back(*current);
        return;
    }
    const int remaining = k - static_cast<int>(current->size());
    for (int orbital = first; orbital <= n - remaining; ++orbital) {
        current->push_back(orbital);
        combinations(n, k, orbital + 1, current, result);
        current->pop_back();
    }
}

std::vector<std::vector<int>> strings(int n, int k) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    combinations(n, k, 0, &current, &result);
    return result;
}

struct Fixture {
    std::vector<std::vector<int>> alpha, beta;
    Eigen::MatrixXd upper;
    Eigen::MatrixXd selected;
    std::vector<std::vector<int>> product_alpha, product_beta;
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms;
    xmvb::vb::SameSpinPairCacheContext topology;
    xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
    Eigen::MatrixXd one_electron;
    std::vector<double> overlap;

    Fixture(int n, int n_alpha, int n_beta, Eigen::MatrixXd columns)
        : alpha(strings(n, n_alpha)), beta(strings(n, n_beta)),
          upper(Eigen::MatrixXd::Identity(n, n)), selected(std::move(columns)),
          one_electron(Eigen::MatrixXd::Zero(n, n)) {
        for (int row = 0; row < n; ++row) {
            upper(row, row) = 0.9 + 0.08 * row;
            for (int col = row + 1; col < n; ++col)
                upper(row, col) = 0.04 * (row + 1) * (col + 1);
        }
        const Eigen::MatrixXd s = upper.transpose() * upper;
        overlap.assign(s.data(), s.data() + s.size());
        terms.resize(alpha.size() * beta.size());
        for (const auto& a : alpha)
            for (const auto& b : beta) {
                const int product = static_cast<int>(product_alpha.size());
                product_alpha.push_back(a);
                product_beta.push_back(b);
                for (int col = 0; col < selected.cols(); ++col)
                    if (selected(product, col) != 0.0)
                        terms[product].push_back({col, selected(product, col)});
            }
        topology = xmvb::vb::build_same_spin_pair_topology(product_alpha, product_beta, n);
        two_electron.packed_active_two_electron_integrals.assign(
            xmvb::integrals::packed_active_two_electron_integral_count(n), 0.0);
    }

    xmvb::vb::StructureAction
    action(xmvb::vb::StructureActionMode mode = xmvb::vb::StructureActionMode::ActionOnly,
           const xmvb::vb::StructureDiagonal* diagonal = nullptr) const {
        return xmvb::vb::StructureAction(terms,
                                         selected.cols(),
                                         topology,
                                         overlap,
                                         one_electron,
                                         two_electron,
                                         upper.rows(),
                                         diagonal,
                                         mode);
    }
};

void compare_dense(Fixture& fixture) {
    auto action = fixture.action();
    require(action.storage().orthogonal_direct_ci, "test carrier did not select exact direct CI");
    xmvb::vb::RepresentationOverlap representation(
        action, fixture.alpha, fixture.beta, fixture.upper.rows());
    const auto prepared_metric = representation.prepare_metric(fixture.upper);
    require(prepared_metric.selected_metric() != nullptr &&
                prepared_metric.selected_metric()->overlap.data() ==
                    prepared_metric.selected_matrix().data(),
            "bounded selected factor does not share the resident Gram");
    const Eigen::MatrixXd identity =
        Eigen::MatrixXd::Identity(fixture.selected.cols(), fixture.selected.cols());
    const auto exterior = [&](const std::vector<std::vector<int>>& strings) {
        Eigen::MatrixXd transform(strings.size(), strings.size());
        const int electrons = strings.front().size();
        Eigen::MatrixXd minor(electrons, electrons);
        for (int left = 0; left < transform.rows(); ++left)
            for (int right = 0; right < transform.cols(); ++right) {
                for (int i = 0; i < electrons; ++i)
                    for (int j = 0; j < electrons; ++j)
                        minor(i, j) = fixture.upper(strings[left][i], strings[right][j]);
                transform(left, right) = minor.determinant();
            }
        return transform;
    };
    // Independent tiny oracle: explicit minors and direct selected columns.
    // Production never materializes these full product or Gram matrices.
    const Eigen::MatrixXd alpha = exterior(fixture.alpha), beta = exterior(fixture.beta);
    Eigen::MatrixXd carrier =
        Eigen::MatrixXd::Zero(fixture.selected.rows(), fixture.selected.cols());
    for (int a = 0; a < alpha.rows(); ++a)
        for (int b = 0; b < beta.rows(); ++b)
            for (int i = 0; i < alpha.cols(); ++i)
                for (int j = 0; j < beta.cols(); ++j)
                    carrier.row(a * beta.rows() + b).noalias() +=
                        alpha(a, i) * beta(b, j) * fixture.selected.row(i * beta.cols() + j);
    const Eigen::MatrixXd metric = carrier.transpose() * carrier;
    require((prepared_metric(identity) - metric).norm() < 2.0e-11,
            "selected metric action disagrees with explicit exterior minors");
    const Eigen::LLT<Eigen::MatrixXd> inverse(prepared_metric.selected_matrix());
    require(inverse.info() == Eigen::Success, "selected CI metric factorization failed");
    Eigen::MatrixXd change = Eigen::MatrixXd::Identity(identity.rows(), identity.cols());
    for (int j = 0; j < change.cols(); ++j) {
        change(j, j) = 0.8 + 0.1 * j;
        for (int i = 0; i < j; ++i)
            change(i, j) = 0.13 * std::sin(i + 2 * j + 1);
    }
    const Eigen::MatrixXd transformed_metric = change.transpose() * metric * change;
    const Eigen::LLT<Eigen::MatrixXd> transformed_inverse(transformed_metric);
    Eigen::MatrixXd rhs(fixture.selected.cols(), 11);
    for (int col = 0; col < rhs.cols(); ++col)
        for (int row = 0; row < rhs.rows(); ++row)
            rhs(row, col) = std::sin(0.21 * (row + 1) * (col + 2));
    const Eigen::MatrixXd tangent = inverse.solve(rhs);
    const Eigen::MatrixXd transformed_rhs = change.transpose() * rhs;
    const Eigen::MatrixXd recovered = change * transformed_inverse.solve(transformed_rhs);
    require((metric * tangent - rhs).norm() < 2e-10 * std::max(1.0, rhs.norm()) &&
                (tangent - recovered).norm() < 2e-10 * std::max(1.0, tangent.norm()),
            "selected CI Riesz preconditioner is not invariant under basis congruence");
    const Eigen::VectorXd diagonal = metric.diagonal();
    Eigen::MatrixXd solution(rhs.rows(), rhs.cols());
    for (int col = 0; col < rhs.cols(); ++col) {
        xmvb::linalg::MetricSolveStats stats;
        solution.col(col) = xmvb::linalg::solve_metric(
            Eigen::VectorXd(rhs.col(col)),
            [&](const Eigen::VectorXd& x) { return Eigen::VectorXd(prepared_metric(x).col(0)); },
            [&](const Eigen::VectorXd& x) { return Eigen::VectorXd(x.cwiseQuotient(diagonal)); },
            &stats);
        require(stats.certifications > 0 &&
                    stats.relative_residual <= std::sqrt(std::numeric_limits<double>::epsilon()),
                "selected PCG did not certify its true metric residual");
    }
    const Eigen::MatrixXd reference = metric.fullPivLu().solve(rhs);
    require((solution - reference).norm() < 2.0e-10 * std::max(1.0, reference.norm()),
            "selected metric differs from independently solved selected Gram");
    require((metric * solution - rhs).norm() < 2.0e-10 * std::max(1.0, rhs.norm()),
            "selected metric failed selected-space residual");
    // Changing the orbital chart changes M, not the selected coefficient space.
    // The old positive diagonal remains a valid preconditioner for the new M.
    Eigen::MatrixXd trial = fixture.upper;
    trial(0, 0) *= 0.83;
    trial(0, trial.cols() - 1) += 0.17;
    const auto trial_action = representation.prepare_metric(trial);
    const Eigen::MatrixXd trial_metric = trial_action(identity);
    require(trial_action.selected_metric() != nullptr &&
                trial_action.selected_metric() != prepared_metric.selected_metric() &&
                (trial_metric - metric).norm() > 1e-5,
            "trial chart reused the accepted selected factor");
    for (int col = 0; col < rhs.cols(); ++col) {
        const Eigen::VectorXd x = xmvb::linalg::solve_metric(
            Eigen::VectorXd(rhs.col(col)),
            [&](const Eigen::VectorXd& v) { return Eigen::VectorXd(trial_action(v).col(0)); },
            [&](const Eigen::VectorXd& v) { return Eigen::VectorXd(v.cwiseQuotient(diagonal)); });
        const Eigen::VectorXd reference_trial = trial_metric.fullPivLu().solve(rhs.col(col));
        require((x - reference_trial).norm() < 2.0e-10 * std::max(1.0, reference_trial.norm()),
                "accepted Jacobi diagonal changed the trial-chart solution");
    }
}

void check_retained_ill_conditioned_metric() {
    Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(9, 3);
    selected(0, 0) = 1.0;
    selected(0, 1) = 1.0;
    selected(4, 1) = 8.0e-4;
    selected(8, 2) = 1.0;
    Fixture fixture(3, 1, 1, selected);
    const auto action = fixture.action();
    std::shared_ptr<const xmvb::vb::SelectedMetric> retained;
    {
        xmvb::vb::RepresentationOverlap representation(
            action, fixture.alpha, fixture.beta, fixture.upper.rows());
        const auto prepared = representation.prepare_metric(fixture.upper);
        retained = prepared.selected_metric();
    }
    require(retained != nullptr && retained->upper.allFinite(),
            "selected factor did not survive its prepared action");
    const Eigen::MatrixXd& metric = retained->overlap;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(metric);
    require(spectrum.info() == Eigen::Success && spectrum.eigenvalues()[0] > 0.0 &&
                spectrum.eigenvalues().tail(1)[0] / spectrum.eigenvalues()[0] > 1.0e6,
            "selected Gram fixture is not ill-conditioned and positive");
    const Eigen::VectorXd rhs = 1.0e-9 * Eigen::Vector3d(0.31, -0.27, 0.44);
    xmvb::linalg::MetricSolveStats stats;
    const Eigen::VectorXd solution = xmvb::linalg::solve_metric(
        rhs,
        [&](const Eigen::VectorXd& vector) {
            return Eigen::VectorXd(action.apply_overlap(vector).col(0));
        },
        [&](const Eigen::VectorXd& residual) { return Eigen::VectorXd(retained->solve(residual)); },
        &stats);
    const double relative_residual =
        (rhs - action.apply_overlap(solution).col(0)).norm() / rhs.norm();
    require(stats.certifications >= 1 &&
                relative_residual <= std::sqrt(std::numeric_limits<double>::epsilon()),
            "retained factor lost the original metric residual certificate");
}

/** @brief Resolve a finite physical state without squaring source conditioning. */
void check_physical_qr_factor() {
    constexpr double delta = 1e-6;
    Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(9, 3);
    selected(0, 0) = selected(0, 1) = selected(8, 2) = 1.0;
    selected(4, 1) = delta;
    Fixture fixture(3, 1, 1, selected);
    const auto action = fixture.action();
    const auto& view = action.orthogonal_view();
    const Eigen::Vector3d c(-1.0 / delta, 1.0 / delta, 0.0);
    // W*c is exactly the (alpha=1,beta=1) unit product. Its exterior
    // image is R(:,1) R(:,1)^T; no near-equal Gram entries are subtracted.
    const double expected = std::pow(fixture.upper.col(1).squaredNorm(), 2);
    const double bound = 64 * std::numeric_limits<double>::epsilon() / delta * expected;
    for (std::size_t bytes : {512UL, 8192UL}) {
        const auto metric = xmvb::vb::SelectedMetric::from_upper(
            action.selected_overlap_factor(view.alpha, view.beta, bytes));
        require(std::abs((metric.upper * c).squaredNorm() - expected) < bound,
                "physical QR lost a necessary direction through normal equations or tiling");
    }
}

/** @brief Different row partitions keep the same near-dependent selected span. */
void check_parallel_qr_factor() {
    constexpr double delta = 1e-6;
    Eigen::MatrixXd selected = Eigen::MatrixXd::Identity(36, 34);
    selected.col(1).setZero();
    selected(0, 1) = 1.0;
    selected(1, 1) = delta;
    Fixture fixture(4, 2, 2, selected);
    const auto action = fixture.action();
    const auto& view = action.orthogonal_view();
    Eigen::VectorXd c = Eigen::VectorXd::Zero(34);
    c[0] = -1.0 / delta;
    c[1] = 1.0 / delta;
    const Eigen::MatrixXd s = fixture.upper.transpose() * fixture.upper;
    // W*c is the product (alpha={0,1}, beta={0,2}). Its norm is the
    // product of the two occupied 2x2 Gram determinants.
    const double expected =
        (s(0, 0) * s(1, 1) - s(0, 1) * s(1, 0)) * (s(0, 0) * s(2, 2) - s(0, 2) * s(2, 0));
    const double bound = 64 * std::numeric_limits<double>::epsilon() / delta * expected;
    const std::size_t bytes = 4 * 34 * 34 * sizeof(double);
    for (auto budget : {bytes, 6 * bytes}) {
        const auto metric = xmvb::vb::SelectedMetric::from_upper(
            action.selected_overlap_factor(view.alpha, view.beta, budget));
        require(std::abs((metric.upper * c).squaredNorm() - expected) < bound,
                "TSQR partitions changed the physical norm of a selected direction");
    }
}

void check_selected_spin_metric() {
    // Two independent columns in one occupation block plus an ionic column.
    // Neither completeness, closure nor a particular spin sector is required
    // by the orbital-independent Gram factorization.
    Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(9, 3);
    selected(0, 0) = 1.3;
    selected(1, 1) = selected(3, 1) = 1.0;
    selected(1, 2) = 2.0;
    selected(3, 2) = -1.0;
    Fixture fixture(3, 1, 1, selected);
    auto action = fixture.action();
    const Eigen::MatrixXd gram = selected.transpose() * selected;
    constexpr std::size_t bytes = 5 * sizeof(double);
    require(xmvb::vb::SpinMetric::fits(action, fixture.alpha, fixture.beta, bytes) &&
                !xmvb::vb::SpinMetric::fits(action, fixture.alpha, fixture.beta, bytes - 1),
            "selected spin metric factor budget is not exact");
    const xmvb::vb::SpinMetric spin(action, fixture.alpha, fixture.beta, bytes);
    require(spin.block_count() == 2 && spin.max_block_size() == 2 && spin.factor_bytes() == bytes,
            "selected spin metric did not preserve occupation blocks");
    const Eigen::VectorXd rhs = Eigen::Vector3d(0.31, -0.47, 0.28);
    require((gram * spin.solve(rhs).col(0) - rhs).norm() < 1e-12 &&
                (spin.diagonal() - gram.diagonal()).norm() < 1e-12,
            "selected spin Gram solve disagrees with independent dense oracle");
    const Eigen::VectorXd weights = Eigen::Vector3d(0.8, 1.1, 1.1);
    const Eigen::MatrixXd diagonal_orbital_metric =
        weights.asDiagonal() * gram * weights.asDiagonal();
    require(
        (diagonal_orbital_metric * spin.precondition(rhs, diagonal_orbital_metric.diagonal()) - rhs)
                .norm() < 1e-12,
        "scaled spin inverse is not exact for occupation-diagonal orbital metrics");

    // At nonorthogonal orbitals this remains only a preconditioner. The exact
    // selected M action and original certified residual determine the solution.
    const xmvb::vb::RepresentationOverlap representation(
        action, fixture.alpha, fixture.beta, fixture.upper.rows());
    const auto metric = representation.prepare_metric(fixture.upper);
    const Eigen::MatrixXd physical = metric.selected_matrix();
    xmvb::linalg::MetricSolveStats stats;
    const auto solution = xmvb::linalg::solve_metric(
        rhs,
        [&](const Eigen::VectorXd& x) { return Eigen::VectorXd(physical * x); },
        [&](const Eigen::VectorXd& x) { return spin.precondition(x, physical.diagonal()); },
        &stats);
    require(stats.certifications > 0 &&
                stats.relative_residual <= std::sqrt(std::numeric_limits<double>::epsilon()) &&
                (physical * solution - rhs).norm() < 1e-10 * rhs.norm(),
            "spin-preconditioned selected solve lost the physical metric certificate");
    const Eigen::VectorXd other = Eigen::Vector3d(-0.1, 0.8, 0.34);
    const Eigen::VectorXd px = spin.precondition(rhs, physical.diagonal());
    const Eigen::VectorXd py = spin.precondition(other, physical.diagonal());
    require(rhs.dot(px) > 0.0 && std::abs(rhs.dot(py) - other.dot(px)) < 1e-12,
            "scaled spin preconditioner is not symmetric positive definite");

    const Eigen::Vector3d d(0.6, 1.4, 0.8);
    const Eigen::VectorXd k = (d.array() / gram.diagonal().array()).sqrt();
    const Eigen::MatrixXd model = k.asDiagonal() * gram * k.asDiagonal();
    const Eigen::MatrixXd inverse = spin.solve_diagonal_model(Eigen::Matrix3d::Identity(), d);
    require((model * inverse - Eigen::Matrix3d::Identity()).norm() < 2e-12 &&
                (inverse - inverse.transpose()).norm() < 2e-12 && rhs.dot(inverse * rhs) > 0.0,
            "diagonal-matched spin inverse differs from independent dense model");

    // The selected columns do not close under orbital changes. Check the
    // constrained preconditioner against a separately restricted dense model.
    const xmvb::vb::StructureDiagonal model_diagonal{Eigen::VectorXd::Zero(3), physical.diagonal()};
    auto modeled_action = fixture.action(xmvb::vb::StructureActionMode::Davidson, &model_diagonal);
    for (int states : {1, 2}) {
        auto point = std::make_shared<xmvb::vb::AcceptedPointContext>();
        point->n_structures = 3;
        point->selected_state_indices.resize(states);
        point->selected_state_energies.assign(states, 0.0);
        point->normalized_state_weights.assign(states, 1.0 / states);
        const Eigen::LLT<Eigen::MatrixXd> norm(physical);
        point->selected_state_eigenvectors =
            norm.matrixU().solve(Eigen::Matrix3d::Identity()).leftCols(states);
        const auto& c = point->selected_state_eigenvectors;
        point->selected_state_structure_images =
            xmvb::vb::StructureActionResult{Eigen::MatrixXd::Zero(3, states), physical * c};
        point->spin_metric = std::make_shared<xmvb::vb::SpinMetric>(
            modeled_action, fixture.alpha, fixture.beta, bytes);
        const xmvb::vb::StructureTangentOperator tangent(point, modeled_action);
        const int size = tangent.tangent_size();
        Eigen::MatrixXd z(3 * states, size), actual(size, size);
        for (int j = 0; j < size; ++j) {
            const auto axis = Eigen::VectorXd::Unit(size, j);
            const auto raw = tangent.expand(axis).scaled_coefficients;
            z.col(j) = Eigen::Map<const Eigen::VectorXd>(raw.data(), raw.size());
            actual.col(j) = tangent.apply_inverse_augmented_hessian_diagonal(axis, -0.07);
        }
        const auto& diagonal = modeled_action.preconditioner_diagonal();
        const Eigen::VectorXd positive =
            (diagonal.hamiltonian + 0.07 * diagonal.overlap).cwiseAbs().cwiseMax(1e-8);
        const Eigen::VectorXd scale = (positive.array() / gram.diagonal().array()).sqrt();
        const Eigen::MatrixXd block = scale.asDiagonal() * gram * scale.asDiagonal();
        Eigen::MatrixXd blocks = Eigen::MatrixXd::Zero(3 * states, 3 * states);
        for (int s = 0; s < states; ++s)
            blocks.block(3 * s, 3 * s, 3, 3) = block;
        const Eigen::MatrixXd expected = (z.transpose() * blocks * z).inverse();
        require((actual - expected).norm() < 3e-11 && (actual - actual.transpose()).norm() < 3e-12,
                "spin AH model inverse lost a selected-space tangent constraint");
    }
}

void singlet_case() {
    Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(9, 4);
    selected(0, 0) = 1.0;
    selected(4, 1) = 1.0;
    selected(8, 2) = 1.0;
    selected(1, 3) = selected(3, 3) = 1.0;
    Fixture fixture(3, 1, 1, selected);
    compare_dense(fixture);

    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(9, 6);
    full.leftCols(4) = selected;
    full(2, 4) = full(6, 4) = 1.0;
    full(5, 5) = full(7, 5) = 1.0;
    Eigen::MatrixXd compound(9, 9);
    for (int target_a = 0; target_a < 3; ++target_a)
        for (int target_b = 0; target_b < 3; ++target_b)
            for (int source_a = 0; source_a < 3; ++source_a)
                for (int source_b = 0; source_b < 3; ++source_b)
                    compound(target_a * 3 + target_b, source_a * 3 + source_b) =
                        fixture.upper(target_a, source_a) * fixture.upper(target_b, source_b);
    const Eigen::MatrixXd carrier_metric = compound.transpose() * compound;
    const Eigen::MatrixXd selected_metric = selected.transpose() * carrier_metric * selected;
    const Eigen::MatrixXd full_metric = full.transpose() * carrier_metric * full;
    const Eigen::MatrixXd naive_pp = full_metric.inverse().topLeftCorner(4, 4);
    require((selected_metric.inverse() - naive_pp).norm() > 1.0e-9,
            "test failed to distinguish the restricted inverse from N_PP");

    selected.col(3) *= -2.0;
    Fixture scaled(3, 1, 1, selected);
    scaled.upper(0, 0) = 0.35;
    scaled.upper(2, 2) = 2.8;
    const Eigen::MatrixXd scaled_overlap = scaled.upper.transpose() * scaled.upper;
    scaled.overlap.assign(scaled_overlap.data(), scaled_overlap.data() + scaled_overlap.size());
    compare_dense(scaled);

    selected(3, 3) = 0.0; // Not annihilated by S_+.
    Fixture impure(3, 1, 1, selected);
    compare_dense(impure);

    selected(3, 3) = -2.0;
    selected(0, 3) = 0.2; // One column now crosses occupation blocks.
    Fixture nonlocal(3, 1, 1, selected);
    compare_dense(nonlocal);
    // A single selected structure requires one scalar equation, no complement.
    Fixture single(3, 1, 1, selected.leftCols(1));
    compare_dense(single);
}

void doublet_case() {
    // Three orbitals, two alpha and one beta: six one-doubly-occupied
    // highest-weight products and two three-open-shell doublets.
    Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(9, 7);
    int col = 0;
    const auto alpha = strings(3, 2);
    const auto beta = strings(3, 1);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            if (std::find(alpha[a].begin(), alpha[a].end(), beta[b][0]) != alpha[a].end())
                selected(a * 3 + b, col++) = 1.0;
    require(col == 6, "doublet fixture has wrong paired-product count");
    // S_+ signs for (01;2) and (02;1) are + and -, respectively.
    selected(0 * 3 + 2, 6) = 1.0;
    selected(1 * 3 + 1, 6) = 1.0;
    Fixture fixture(3, 2, 1, selected);
    compare_dense(fixture);
}

} // namespace

int main() {
    check_physical_qr_factor();
    check_parallel_qr_factor();
    check_selected_spin_metric();
    check_retained_ill_conditioned_metric();
    singlet_case();
    doublet_case();
    std::cout << "selected metric checks passed\n";
}
