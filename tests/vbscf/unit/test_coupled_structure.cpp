#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "core/pair_index.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/ao/graph.hpp"
#include "vbscf/optimization/neo/coordinates.hpp"
#include "vbscf/optimization/coupled/affine.hpp"
#include "vbscf/optimization/coupled/ci_transport.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/optimization/trust_region/retraction.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "linalg/metric_solve.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

/** @brief Repeated Riesz maps reuse inverse images, not an approximate norm. */
void check_metric_recycling() {
    constexpr int n = 24;
    Eigen::MatrixXd transform = Eigen::MatrixXd::Identity(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            transform(i, j) = 0.03 * std::sin(i + 2 * j);
    const Eigen::VectorXd spectrum =
        Eigen::VectorXd::LinSpaced(n, 0.0, 4.0).array().unaryExpr([](double x) {
            return std::pow(10.0, x);
        });
    const Eigen::MatrixXd metric = transform * spectrum.asDiagonal() * transform.transpose();
    const auto action = [&](const Eigen::VectorXd& x) { return (metric * x).eval(); };
    const auto inverse = [&](const Eigen::VectorXd& x) {
        return (x.array() / metric.diagonal().array()).matrix().eval();
    };
    xmvb::linalg::MetricSolveWorkspace workspace(3);
    const Eigen::VectorXd a = Eigen::VectorXd::LinSpaced(n, -0.37, 0.62);
    const Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 0.29, -0.43);
    xmvb::linalg::MetricSolveStats stats;
    const auto xa = workspace.solve(a, action, inverse, &stats);
    const auto xb = workspace.solve(b, action, inverse, &stats);
    const Eigen::VectorXd rhs = 0.4 * a - 0.7 * b;
    const auto x = workspace.solve(rhs, action, inverse, &stats);
    require((metric * x - rhs).norm() <=
                std::sqrt(std::numeric_limits<double>::epsilon()) * rhs.norm(),
            "recycled Riesz solve changed its true-residual accuracy");
    require(stats.iterations == 0 && stats.actions == 1 && stats.certifications == 1,
            "a known inverse span launched another metric CG solve");
    require((x - (0.4 * xa - 0.7 * xb)).norm() < 1e-10,
            "recycled metric map is not linear on solved covectors");
    for (int j = 0; j < 6; ++j) {
        const Eigen::VectorXd probe = Eigen::VectorXd::Unit(n, j);
        const auto y = workspace.solve(probe, action, inverse, &stats);
        require((metric * y - probe).norm() <= std::sqrt(std::numeric_limits<double>::epsilon()),
                "new RHS lost the recycled metric solve certificate");
        require(workspace.size() <= 3, "metric reuse space exceeded its memory bound");
    }
    const auto zero = workspace.solve(Eigen::VectorXd::Zero(n), action, inverse, &stats);
    require(zero.isZero() && stats.actions == 0, "zero covector requires no metric action");
    xmvb::linalg::MetricSolveWorkspace observed(2);
    const Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(n, -0.42, 0.73);
    observed.observe(direction, action(direction));
    const auto known = observed.solve(action(direction), action, inverse, &stats);
    require(stats.iterations == 0 && stats.actions == 1 &&
                (metric * known - action(direction)).norm() <=
                    std::sqrt(std::numeric_limits<double>::epsilon()) * action(direction).norm(),
            "an existing AH metric image relaunched CG");

    // The captured component is exactly zero, but the complete residual is large.
    xmvb::linalg::MetricSolveWorkspace decisions(2);
    Eigen::Matrix3d diagonal = Eigen::Vector3d(2.0, 0.001, 7.0).asDiagonal();
    const Eigen::VectorXd e0 = Eigen::VectorXd::Unit(3, 0);
    decisions.observe(e0, diagonal * e0);
    const Eigen::VectorXd outside = Eigen::VectorXd::Unit(3, 1);
    int actions = 0;
    const auto metric_action = [&](const Eigen::VectorXd& v) {
        ++actions;
        return (diagonal * v).eval();
    };
    const auto jacobi = [&](const Eigen::VectorXd& v) {
        return (v.array() / diagonal.diagonal().array()).matrix().eval();
    };
    require(!decisions.norm_below(outside, 10.0, metric_action, jacobi) && actions == 1,
            "projected-zero residual was accepted or launched a full inverse solve");
    require(decisions.norm_below(outside, 40.0, metric_action, jacobi),
            "ambiguous lower bound did not use the complete metric solve");
    const double norm = std::sqrt(outside.dot(diagonal.inverse() * outside));
    require(!decisions.norm_below(outside, 0.999 * norm, metric_action, jacobi) &&
                decisions.norm_below(outside, 1.001 * norm, metric_action, jacobi),
            "decision comparison changed its threshold");
}

struct StructureFixture {
    xmvb::vb::StructureAction action;
    Eigen::MatrixXd hamiltonian;
    Eigen::MatrixXd overlap;
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
};

StructureFixture make_fixture() {
    constexpr int n_orbitals = 2;
    constexpr int n_structures = 4;
    const std::vector<std::vector<int>> alpha_strings{{0}, {1}};
    const std::vector<std::vector<int>> beta_strings{{0}, {1}};

    std::vector<std::vector<int>> determinant_alpha;
    std::vector<std::vector<int>> determinant_beta;
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> expansion(n_structures);
    for (int alpha = 0; alpha < 2; ++alpha) {
        for (int beta = 0; beta < 2; ++beta) {
            const int determinant = 2 * alpha + beta;
            determinant_alpha.push_back(alpha_strings[alpha]);
            determinant_beta.push_back(beta_strings[beta]);
            expansion[determinant].push_back({determinant, 1.0});
        }
    }
    const auto topology =
        xmvb::vb::build_same_spin_pair_topology(determinant_alpha, determinant_beta, n_orbitals);

    Eigen::Matrix2d active_overlap;
    active_overlap << 1.0, 0.18, 0.18, 1.24;
    const std::vector<double> packed_overlap(active_overlap.data(),
                                             active_overlap.data() + active_overlap.size());
    Eigen::Matrix2d one_electron;
    one_electron << -1.1, 0.16, 0.16, -0.45;

    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    Eigen::MatrixXd pair_factor(3, n_pairs);
    pair_factor << 0.31, 0.07, -0.02, 0.04, 0.26, 0.08, -0.03, 0.05, 0.22;
    const Eigen::MatrixXd pair_kernel = pair_factor.transpose() * pair_factor;
    xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
    two_electron.packed_active_two_electron_integrals.resize(
        xmvb::integrals::packed_active_two_electron_integral_count(n_orbitals));
    for (int column = 0; column < n_pairs; ++column) {
        for (int row = 0; row <= column; ++row) {
            two_electron.packed_active_two_electron_integrals
                [xmvb::core::PairIndex::packed_pair_of_pairs_index(row, column)] =
                pair_kernel(row, column);
        }
    }

    xmvb::vb::StructureDiagonal diagonal;
    diagonal.hamiltonian = Eigen::VectorXd::Zero(n_structures);
    diagonal.overlap = Eigen::VectorXd::Ones(n_structures);
    xmvb::vb::StructureAction action(expansion,
                                     n_structures,
                                     topology,
                                     packed_overlap,
                                     one_electron,
                                     two_electron,
                                     n_orbitals,
                                     &diagonal);

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_structures, n_structures);
    const xmvb::vb::StructureActionResult matrices = action.apply(identity);
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(matrices.hamiltonian,
                                                                          matrices.overlap);
    require(eigensolver.info() == Eigen::Success, "fixture generalized eigensolve failed");
    return StructureFixture{std::move(action),
                            matrices.hamiltonian,
                            matrices.overlap,
                            eigensolver.eigenvalues(),
                            eigensolver.eigenvectors()};
}

std::shared_ptr<xmvb::vb::AcceptedPointContext>
accepted_point(const StructureFixture& fixture, int n_states, std::vector<double> weights) {
    auto accepted = std::make_shared<xmvb::vb::AcceptedPointContext>();
    accepted->n_structures = static_cast<int>(fixture.hamiltonian.rows());
    accepted->selected_state_indices.resize(n_states);
    accepted->selected_state_energies.resize(n_states);
    for (int state = 0; state < n_states; ++state) {
        accepted->selected_state_indices[state] = state;
        accepted->selected_state_energies[state] = fixture.eigenvalues[state];
    }
    accepted->normalized_state_weights = std::move(weights);
    accepted->selected_state_eigenvectors = fixture.eigenvectors.leftCols(n_states);
    accepted->selected_state_structure_images =
        xmvb::vb::StructureActionResult{fixture.hamiltonian * accepted->selected_state_eigenvectors,
                                        fixture.overlap * accepted->selected_state_eigenvectors};
    auto& selected = accepted->selected_state_matrices;
    selected.n_structures = accepted->n_structures;
    selected.n_unique_alpha = selected.n_unique_beta = 2;
    selected.normalized_state_weights = accepted->normalized_state_weights;
    selected.states.resize(n_states);
    for (int i = 0; i < n_states; ++i) {
        selected.states[i].normalized_state_weight = accepted->normalized_state_weights[i];
        selected.states[i].coefficient_matrix =
            fixture.action.expand_structure_block(accepted->selected_state_eigenvectors.col(i));
    }
    return accepted;
}

double block_dot(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) {
    return (left.array() * right.array()).sum();
}

xmvb::vb::OrbitalPreparationInput make_orbital_input() {
    xmvb::vb::OrbitalPreparationInput input;
    input.n_basis_functions = 2;
    input.n_orbitals = 1;
    input.n_active_orbitals = 1;
    input.n_active_electrons = 1;
    input.n_total_electrons = 1;
    input.orbital_type = 1;
    input.ao_overlap_matrix.resize(2, 2);
    input.ao_overlap_matrix << 1.0, 0.12, 0.12, 1.18;
    Eigen::Vector2d orbital(1.0, 0.28);
    orbital /= std::sqrt(orbital.dot(input.ao_overlap_matrix * orbital));
    input.orbital_basis_counts = {2};
    input.orbital_value_table = {orbital[0], orbital[1]};
    input.orbital_basis_index_table = {1, 2};
    return input;
}

void check_single_state(const StructureFixture& fixture) {
    const auto accepted = accepted_point(fixture, 1, {1.0});
    const xmvb::vb::StructureTangentOperator structure(accepted, fixture.action);
    require(structure.n_structures() == 4 && structure.n_states() == 1,
            "single-state coupled structure dimensions are wrong");

    Eigen::Vector4d raw;
    raw << 0.7, -0.4, 0.25, 0.8;
    const xmvb::vb::StructureTangent tangent = structure.project(raw);
    const Eigen::MatrixXd overlap_tangent = fixture.overlap * tangent.scaled_coefficients;
    require((accepted->selected_state_eigenvectors.transpose() * overlap_tangent).norm() < 2.0e-13,
            "projected structure tangent is not horizontal");

    const Eigen::MatrixXd coefficient_response = 0.6 * raw;
    const xmvb::vb::StructureTangent scaled =
        structure.from_coefficient_response(coefficient_response);
    const Eigen::MatrixXd expected_scaled =
        structure.project(std::sqrt(2.0) * coefficient_response).scaled_coefficients;
    require((scaled.scaled_coefficients - expected_scaled).norm() < 2.0e-13,
            "coefficient response was not scaled by sqrt(2w)");
    const Eigen::MatrixXd round_trip = structure.coefficient_response(scaled);
    const Eigen::MatrixXd expected_response =
        structure.project(coefficient_response).scaled_coefficients;
    require((round_trip - expected_response).norm() < 2.0e-13,
            "scaled structure coordinate round trip is inconsistent");

    const double expected_norm =
        block_dot(tangent.scaled_coefficients, fixture.overlap * tangent.scaled_coefficients);
    require(std::abs(structure.squared_norm(tangent) - expected_norm) < 2.0e-13,
            "structure physical norm differs from z^T S z");
    require(expected_norm > 0.0, "test structure tangent has zero physical norm");
    const Eigen::VectorXd covector = structure.coordinates(tangent);
    Eigen::MatrixXd metric(covector.size(), covector.size());
    for (Eigen::Index col = 0; col < metric.cols(); ++col)
        metric.col(col) =
            structure.apply_metric_coordinates(Eigen::VectorXd::Unit(covector.size(), col));
    const double dense_dual = std::sqrt(covector.dot(metric.ldlt().solve(covector)));
    require(std::abs(structure.dual_norm(covector) - dense_dual) < 1e-10,
            "complete structure dual norm differs from dense Riesz reference");
    require(structure.dual_norm(Eigen::VectorXd::Zero(covector.size())) == 0.0,
            "zero structure covector has nonzero dual norm");
}

void check_equal_weight_multistate(const StructureFixture& fixture) {
    const auto accepted = accepted_point(fixture, 2, {0.5, 0.5});
    const xmvb::vb::StructureTangentOperator structure(accepted, fixture.action);
    Eigen::Matrix<double, 4, 2> x;
    x << 0.2, -0.5, 0.7, 0.1, -0.3, 0.8, 0.6, -0.2;
    Eigen::Matrix<double, 4, 2> y;
    y << -0.4, 0.3, 0.5, -0.7, 0.9, 0.2, -0.1, 0.6;
    const auto horizontal_x = structure.project(x);
    const auto horizontal_y = structure.project(y);
    require((structure.project_coordinates(x) - structure.coordinates(horizontal_x)).norm() <
                3.0e-13,
            "direct projected structure coordinates are inconsistent");
    require((accepted->selected_state_eigenvectors.transpose() * fixture.overlap *
             horizontal_x.scaled_coefficients)
                    .norm() < 3.0e-13,
            "multistate tangent retains a selected-subspace component");

    const auto hx = structure.apply_hessian(horizontal_x);
    const auto hy = structure.apply_hessian(horizontal_y);
    const auto coordinate_action =
        structure.apply_coupling_coordinates(structure.coordinates(horizontal_x));
    xmvb::vb::StructureActiveIntegralAdjoint joint_adjoint;
    const auto joint =
        structure.apply_coupling_coordinates(structure.coordinates(horizontal_x), &joint_adjoint);
    require((joint.hessian_coordinates - coordinate_action.hessian_coordinates).norm() < 3e-13 &&
                (joint.metric_coordinates - coordinate_action.metric_coordinates).norm() < 3e-13 &&
                (joint.adjoint_multipliers - coordinate_action.adjoint_multipliers).norm() < 3e-13,
            "joint response changed horizontal Hessian, metric or multipliers");
    const auto separate_adjoint =
        fixture.action.active_integral_response_adjoint(*accepted->structure_adjoint_state,
                                                        coordinate_action.coefficient_response,
                                                        coordinate_action.adjoint_multipliers);
    require((joint_adjoint.overlap - separate_adjoint.overlap).norm() < 3e-13 &&
                (joint_adjoint.one_electron - separate_adjoint.one_electron).norm() < 3e-13 &&
                (joint_adjoint.pair_kernel - separate_adjoint.pair_kernel).norm() < 3e-13,
            "joint structure response changed the integral adjoint");
    const auto coordinate_y =
        structure.apply_coupling_coordinates(structure.coordinates(horizontal_y));
    Eigen::MatrixXd coordinate_block(structure.tangent_size(), 2);
    coordinate_block.col(0) = structure.coordinates(horizontal_x);
    coordinate_block.col(1) = structure.coordinates(horizontal_y);
    const auto block_action = structure.apply_coupling_coordinate_block(coordinate_block);
    require(
        (block_action.hessian_coordinates.col(0) - coordinate_action.hessian_coordinates).norm() <
                3.0e-13 &&
            (block_action.hessian_coordinates.col(1) - coordinate_y.hessian_coordinates).norm() <
                3.0e-13,
        "block structure Hessian differs from scalar actions");
    require((block_action.metric_coordinates.col(0) - coordinate_action.metric_coordinates).norm() <
                    3.0e-13 &&
                (block_action.metric_coordinates.col(1) - coordinate_y.metric_coordinates).norm() <
                    3.0e-13,
            "block structure metric differs from scalar actions");
    require(
        (block_action.coefficient_responses[0] - coordinate_action.coefficient_response).norm() <
                3.0e-13 &&
            (block_action.adjoint_multipliers[1] - coordinate_y.adjoint_multipliers).norm() <
                3.0e-13,
        "block structure adjoint data differ from scalar actions");
    require((coordinate_action.hessian_coordinates - structure.coordinates(hx)).norm() < 3.0e-13,
            "coordinate-space structure Hessian action is inconsistent");
    require((coordinate_action.metric_coordinates -
             structure.coordinates(structure.apply_metric(horizontal_x)))
                    .norm() < 3.0e-13,
            "shared H/S structure metric action is inconsistent");
    const auto ambient_action = structure.apply_coupling(horizontal_x);
    require((coordinate_action.coefficient_response - ambient_action.coefficient_response).norm() <
                3.0e-13,
            "coordinate-space coefficient response is inconsistent");
    require((coordinate_action.adjoint_multipliers - ambient_action.adjoint_multipliers).norm() <
                3.0e-13,
            "coordinate-space structure multipliers are inconsistent");
    const double h_xy = block_dot(horizontal_x.scaled_coefficients, hy.scaled_coefficients);
    const double h_yx = block_dot(hx.scaled_coefficients, horizontal_y.scaled_coefficients);
    require(std::abs(h_xy - h_yx) < 2.0e-12 * std::max({1.0, std::abs(h_xy), std::abs(h_yx)}),
            "coupled structure Hessian is not symmetric");

    const auto mx = structure.apply_metric(horizontal_x);
    const auto my = structure.apply_metric(horizontal_y);
    const double m_xy = block_dot(horizontal_x.scaled_coefficients, my.scaled_coefficients);
    const double m_yx = block_dot(mx.scaled_coefficients, horizontal_y.scaled_coefficients);
    require(std::abs(m_xy - m_yx) < 2.0e-12 * std::max({1.0, std::abs(m_xy), std::abs(m_yx)}),
            "coupled structure metric is not symmetric");
    require(std::abs(structure.squared_norm(horizontal_x) -
                     block_dot(horizontal_x.scaled_coefficients,
                               fixture.overlap * horizontal_x.scaled_coefficients)) < 3.0e-13,
            "multistate physical norm is inconsistent");
}

void check_unequal_weights_rejected(const StructureFixture& fixture) {
    const auto accepted = accepted_point(fixture, 2, {0.6, 0.4});
    bool rejected = false;
    try {
        check_metric_recycling();
        (void)xmvb::vb::StructureTangentOperator(accepted, fixture.action);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "unequal multistate structure tangent was accepted");
}

void check_coupled_neo_coordinates(const StructureFixture& fixture) {
    const auto accepted = accepted_point(fixture, 2, {0.5, 0.5});
    const xmvb::vb::StructureTangentOperator structure(accepted, fixture.action);

    const xmvb::vb::OrbitalPreparationInput input = make_orbital_input();
    const xmvb::vb::SparseParameterLayout layout(input);
    Eigen::Matrix<double, 2, 1> orbitals;
    orbitals << input.orbital_value_table[0], input.orbital_value_table[1];
    const xmvb::vb::OrbitalChart chart(input, layout, orbitals, orbitals);
    const xmvb::vb::NonredundantRetractionMetric orbital_metric(chart, layout, input);
    const xmvb::vb::CoupledNeoCoordinates coordinates(chart, orbital_metric, structure);

    require(coordinates.orbital_size() == chart.reduced_size(),
            "coupled NEO orbital coordinate dimension is wrong");
    require(coordinates.structure_size() == 4,
            "coupled NEO retained ambient structure null coordinates");

    Eigen::Matrix<double, 4, 2> ambient;
    ambient << 0.2, -0.5, 0.7, 0.1, -0.3, 0.8, 0.6, -0.2;
    xmvb::vb::CoupledDirection direction;
    direction.orbital = Eigen::VectorXd::LinSpaced(chart.reduced_size(), 0.31, 0.31);
    direction.structure = structure.project(ambient);
    const Eigen::VectorXd packed = coordinates.flatten(direction);
    const xmvb::vb::CoupledDirection restored = coordinates.unflatten(packed);
    require((restored.orbital - direction.orbital).norm() < 2.0e-14,
            "coupled NEO orbital coordinates do not round trip");
    require(
        (restored.structure.scaled_coefficients - direction.structure.scaled_coefficients).norm() <
            3.0e-13,
        "coupled NEO structure coordinates do not round trip");

    const Eigen::VectorXd metric_image = coordinates.apply_metric(packed);
    xmvb::vb::CoupledDirection expected_metric;
    expected_metric.orbital = orbital_metric.apply(direction.orbital);
    expected_metric.structure = structure.apply_metric(direction.structure);
    require((metric_image - coordinates.flatten(expected_metric)).norm() < 3.0e-13,
            "coupled NEO physical metric action is inconsistent");
    require(coordinates.squared_norm(packed) > 0.0, "coupled NEO physical metric is not positive");

    const Eigen::VectorXd other = Eigen::VectorXd::LinSpaced(coordinates.size(), -0.4, 0.7);
    const double xy = packed.dot(coordinates.apply_metric(other));
    const double yx = coordinates.apply_metric(packed).dot(other);
    require(std::abs(xy - yx) < 3.0e-12 * std::max({1.0, std::abs(xy), std::abs(yx)}),
            "coupled NEO physical metric is not symmetric");

    bool rejected = false;
    try {
        xmvb::vb::CoupledDirection nonhorizontal = direction;
        nonhorizontal.structure.scaled_coefficients.col(0) +=
            fixture.overlap * accepted->selected_state_eigenvectors.col(0);
        (void)coordinates.flatten(nonhorizontal);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "coupled NEO accepted an ambient selected-state gauge direction");

    rejected = false;
    try {
        (void)coordinates.unflatten(Eigen::VectorXd::Zero(coordinates.size() + 1));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "coupled NEO accepted an invalid coordinate dimension");
}

/** Compare the horizontal factorized inverse to an independent dense metric. */
void check_metric_factorization(const StructureFixture& fixture) {
    for (const int n_states : {1, 2}) {
        auto accepted =
            accepted_point(fixture, n_states, std::vector<double>(n_states, 1.0 / n_states));
        accepted->complete_active_space = true;
        accepted->same_spin_pair_cache.alpha_reuse_table.unique_determinants = {{0}, {1}};
        accepted->same_spin_pair_cache.beta_reuse_table.unique_determinants = {{0}, {1}};
        const xmvb::vb::StructureTangentOperator structure(accepted, fixture.action);
        const auto* factors = accepted->metric_inverse(fixture.action);
        require(factors != nullptr && factors == accepted->metric_inverse(fixture.action),
                "complete metric factors are not shared at the accepted point");
        require(factors->factor_bytes() < 4 * 4 * sizeof(double),
                "horizontal inverse allocated a full structure matrix");
        Eigen::MatrixXd dense(structure.tangent_size(), structure.tangent_size());
        for (int j = 0; j < dense.cols(); ++j)
            dense.col(j) =
                structure.apply_metric_coordinates(Eigen::VectorXd::Unit(dense.cols(), j));
        const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(dense.rows(), -0.37, 0.62);
        const Eigen::VectorXd expected = dense.llt().solve(rhs);
        const Eigen::VectorXd actual = structure.apply_inverse_metric_model(rhs);
        require((expected - actual).norm() < 1e-12 && (dense * actual - rhs).norm() < 1e-12,
                "complete horizontal overlap inverse differs from its dense reference");
        require(std::abs(structure.dual_norm(rhs) - std::sqrt(rhs.dot(expected))) < 1e-12,
                "factorized structure norm changed the physical metric");
        xmvb::linalg::MetricSolveStats stats;
        const auto certified = xmvb::linalg::solve_metric(
            rhs,
            [&](const Eigen::VectorXd& x) { return structure.apply_metric_coordinates(x); },
            [&](const Eigen::VectorXd& x) { return structure.apply_inverse_metric_model(x); },
            &stats);
        require(stats.iterations <= 2 && stats.certifications >= 1 &&
                    stats.relative_residual <= std::sqrt(std::numeric_limits<double>::epsilon()) &&
                    (certified - expected).norm() < 1e-12,
                "factorized metric inverse lost its true-residual certification");
    }
    auto selected = accepted_point(fixture, 1, {1.0});
    require(selected->metric_inverse(fixture.action) == nullptr,
            "complete spin carrier incorrectly admitted a noncomplete VB space");
}

/** @brief Carrier inverse needs both horizontal Riesz maps in a nonorthogonal basis. */
void check_carrier_diagonal(const StructureFixture& fixture) {
    const auto expanded = fixture.action.orthogonalize_structure_block(Eigen::Matrix4d::Identity());
    Eigen::Matrix4d f;
    for (int j = 0; j < 4; ++j) {
        const Eigen::Matrix2d block = expanded.middleCols(2 * j, 2);
        f.col(j) = Eigen::Map<const Eigen::Vector4d>(block.data());
    }
    const Eigen::Matrix4d fi = f.inverse();
    const Eigen::Vector4d diagonal = (fi.transpose() * fixture.hamiltonian * fi).diagonal();
    const double shift = -0.07;
    for (const int n_states : {1, 2}) {
        auto accepted =
            accepted_point(fixture, n_states, std::vector<double>(n_states, 1.0 / n_states));
        accepted->complete_active_space = true;
        accepted->same_spin_pair_cache.alpha_reuse_table.unique_determinants = {{0}, {1}};
        accepted->same_spin_pair_cache.beta_reuse_table.unique_determinants = {{0}, {1}};
        const xmvb::vb::StructureTangentOperator structure(accepted, fixture.action);
        const int dimension = structure.tangent_size();
        Eigen::MatrixXd l(4 * n_states, dimension);
        for (int j = 0; j < dimension; ++j) {
            const auto raw =
                structure.expand(Eigen::VectorXd::Unit(dimension, j)).scaled_coefficients;
            for (int s = 0; s < n_states; ++s)
                l.block(4 * s, j, 4, 1) = f * raw.col(s);
        }
        const Eigen::MatrixXd g = l.transpose() * l;
        const Eigen::MatrixXd gi = g.inverse();
        Eigen::VectorXd inverse_diagonal(4 * n_states);
        for (int s = 0; s < n_states; ++s)
            inverse_diagonal.segment(4 * s, 4) =
                (diagonal.array() - fixture.eigenvalues[s] - shift).inverse().matrix();
        const Eigen::MatrixXd expected =
            gi * l.transpose() * inverse_diagonal.asDiagonal() * l * gi;
        Eigen::MatrixXd actual(dimension, dimension);
        for (int j = 0; j < dimension; ++j)
            actual.col(j) = structure.apply_inverse_augmented_hessian_diagonal(
                Eigen::VectorXd::Unit(dimension, j), shift);
        require((actual - expected).norm() < 2e-11,
                "carrier preconditioner did not apply both horizontal Riesz maps");
        require((actual - actual.transpose()).norm() < 2e-12,
                "carrier preconditioner lost coordinate-space symmetry");
    }
}

/** Check the forcing split without assuming a stationary carrier root. */
void check_connection(const StructureFixture& fixture) {
    require(fixture.action.supports_integral_direction(), "fixture lacks direct CI");
    Eigen::Matrix2d overlap_direction, one_direction;
    overlap_direction << 0.03, -0.04, -0.04, 0.02;
    one_direction << 0.11, 0.02, 0.02, -0.07;
    const std::vector<double> overlap(overlap_direction.data(), overlap_direction.data() + 4);
    const std::vector<double> one(one_direction.data(), one_direction.data() + 4);
    std::vector<double> two(xmvb::integrals::packed_active_two_electron_integral_count(2), 0.0);
    for (const bool stationary : {true, false}) {
        Eigen::VectorXd coefficients = fixture.eigenvectors.col(0);
        if (!stationary)
            coefficients += 0.4 * fixture.eigenvectors.col(1);
        coefficients /= std::sqrt(coefficients.dot(fixture.overlap * coefficients));
        const double energy = coefficients.dot(fixture.hamiltonian * coefficients);
        const auto state = fixture.action.prepare_active_adjoint(coefficients, {1.0}, {energy});
        xmvb::vb::StructureIntegralDirection captured, plain;
        captured.capture_connection = true;
        const auto images =
            fixture.action.apply_integral_direction(state, overlap, one, two, &captured);
        const auto reference =
            fixture.action.apply_integral_direction(state, overlap, one, two, &plain);
        require(captured.connection.has_value() && !plain.connection.has_value(),
                "connection diagnostic capture contract failed");
        require((images.hamiltonian - reference.hamiltonian).norm() < 1e-14 &&
                    (images.overlap - reference.overlap).norm() < 1e-14,
                "connection capture changed the action");
        require((images.hamiltonian - energy * images.overlap -
                 captured.connection->representation_forcing -
                 captured.connection->integral_forcing)
                        .norm() < 1e-12,
                "connection forcing decomposition lost a carrier residual term");
        require(captured.connection->squared_norm[0] >= -1e-14,
                "connection tangent squared norm is invalid");
        if (stationary) {
            const auto accepted = accepted_point(fixture, 1, {1.0});
            xmvb::vb::StructureTangentOperator tangent(accepted, fixture.action);
            const double scale = std::sqrt(2.0);
            const Eigen::VectorXd rhs =
                tangent.project_coordinates(scale * captured.connection->overlap);
            Eigen::MatrixXd metric(tangent.tangent_size(), tangent.tangent_size());
            for (int j = 0; j < metric.cols(); ++j)
                metric.col(j) =
                    tangent.apply_metric_coordinates(Eigen::VectorXd::Unit(metric.cols(), j));
            const Eigen::VectorXd gamma = metric.llt().solve(rhs);
            require(std::abs(2.0 * captured.connection->squared_norm[0] - rhs.dot(gamma)) < 1e-12,
                    "complete carrier connection has a spurious closure defect");
            require(
                (tangent.project_coordinates(scale * captured.connection->representation_forcing) -
                 tangent.apply_hessian_coordinates(gamma))
                        .norm() < 1e-12,
                "complete stationary carrier connection is not C gamma");
        } else {
            const auto image = fixture.action.apply(state.structure_coefficients);
            require((image.hamiltonian - energy * image.overlap).norm() > 1e-3,
                    "nonstationary connection test accidentally has zero carrier residual");
        }
    }
}

xmvb::vb::VbScfInput coupling_input() {
    xmvb::vb::VbScfInput input;
    input.standard_two_electron_mode = xmvb::vb::StandardTwoElectronMode::Exact;
    auto& orbitals = input.orbital_preparation_input;
    orbitals.n_basis_functions = 3;
    orbitals.n_orbitals = orbitals.n_active_orbitals = 2;
    orbitals.n_total_electrons = orbitals.n_active_electrons = 2;
    orbitals.orbital_value_table = {1.0, 0.13, -0.17, 0.21, 0.94, 0.12};
    orbitals.orbital_basis_index_table = {1, 2, 3, 1, 2, 3};
    orbitals.orbital_basis_counts = {3, 3};
    orbitals.ao_overlap_matrix = Eigen::Matrix3d::Identity();
    orbitals.ao_overlap_matrix(0, 1) = orbitals.ao_overlap_matrix(1, 0) = 0.1;
    auto& ao = input.ao_integral_input;
    ao.n_basis_functions = 3;
    ao.ao_core_hamiltonian_matrix.resize(3, 3);
    ao.ao_core_hamiltonian_matrix << -1.1, 0.13, 0.04, 0.13, -0.6, -0.08, 0.04, -0.08, 0.2;
    std::vector<int> left, right;
    std::vector<double> values;
    for (int p = 0; p < 6; ++p) {
        for (int q = 0; q <= p; ++q) {
            left.push_back(p);
            right.push_back(q);
            values.push_back(0.07 * std::cos(0.2 * (p - q)));
        }
    }
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, 3);
    auto& data = input.structure_data;
    data.alpha_strings = {{0}, {1}};
    data.beta_strings = {{0}, {1}};
    data.offsets = {0};
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            data.pair_alpha.push_back(a);
            data.pair_beta.push_back(b);
            data.terms.push_back({data.n_structures++, 1.0});
            data.offsets.push_back(data.terms.size());
        }
    }
    return input;
}

/** Compare the complete orbital/CI coupling action with its separate calls. */
void check_joint_coupling_action() {
    const auto input = coupling_input();
    const auto& orbitals = input.orbital_preparation_input;

    const auto accepted =
        xmvb::vb::OrbitalGradientEvaluator().evaluate(input, {0, 3}, {0.5, 0.5}, 0.4);
    const xmvb::vb::SparseParameterLayout layout(orbitals);
    const auto& prepared = accepted.orbital_preparation_result;
    const xmvb::vb::OrbitalChart chart(
        orbitals,
        layout,
        prepared.auxiliary_orbital_matrix.leftCols(orbitals.n_orbitals),
        prepared.physical_orbital_frame.normalized_orbital_matrix,
        nullptr,
        false,
        false,
        nullptr,
        xmvb::vb::OrbitalChart::Coordinates::Unwhitened);
    const xmvb::vb::ExactHvpOperator hessian(accepted.second_order_context, &input, layout, &chart);
    const xmvb::vb::StructureTangentOperator structure(accepted.second_order_context,
                                                       hessian.structure_action());
    require(hessian.structure_action().supports_integral_direction() && chart.reduced_size() > 0 &&
                structure.tangent_size() > 0,
            "joint coupling fixture lacks orbital or direct-CI directions");
    Eigen::VectorXd orbital = Eigen::VectorXd::LinSpaced(chart.reduced_size(), 0.23, 0.57);
    orbital.normalize();
    const Eigen::VectorXd ci = Eigen::VectorXd::LinSpaced(structure.tangent_size(), -0.31, 0.42);
    const auto close = [](const auto& actual, const auto& expected) {
        return (actual - expected).norm() <= 3e-11 * std::max(1.0, expected.norm());
    };
    for (const int direction : {0, 1, 2}) {
        Eigen::VectorXd p = orbital;
        Eigen::VectorXd q = ci;
        if (direction == 1)
            p.setZero();
        if (direction == 2)
            q.setZero();
        const auto separate_p = hessian.apply_orbital_coupling(p);
        xmvb::vb::StructureActiveIntegralAdjoint separate_adjoint;
        const auto separate_ci = structure.apply_coupling_coordinates(q, &separate_adjoint);
        const Eigen::MatrixXd raw = structure.coefficient_response_coordinates(q);
        xmvb::vb::StructureResponseRequest request{raw};
        const auto joint_p = hessian.apply_orbital_coupling(p, &request);
        xmvb::vb::StructureActiveIntegralAdjoint joint_adjoint;
        const auto joint_ci =
            structure.finish_coupling_coordinates(raw, std::move(request.result), &joint_adjoint);
        require(close(joint_p.orbital_hessian, separate_p.orbital_hessian) &&
                    close(joint_p.scaled_structure_forcing, separate_p.scaled_structure_forcing),
                "joint coupling changed orbital Hessian or structure forcing");
        require(close(joint_ci.hessian_coordinates, separate_ci.hessian_coordinates) &&
                    close(joint_ci.metric_coordinates, separate_ci.metric_coordinates) &&
                    close(joint_ci.coefficient_response, separate_ci.coefficient_response) &&
                    close(joint_ci.adjoint_multipliers, separate_ci.adjoint_multipliers),
                "joint coupling changed CI Hessian, metric, response, or multipliers");
        require(close(hessian.apply_structure_coupling_adjoint(joint_adjoint),
                      hessian.apply_structure_coupling_adjoint(separate_adjoint)),
                "joint coupling changed the orbital image of the CI adjoint");
    }
}

/** Production CIAH HVP against analytic keyframe-gradient differences, not FD-vs-FD. */
void check_affine_keyframe_hvp(int n_states, bool selected, bool stationary) {
    using namespace xmvb::vb;
    auto input = coupling_input();
    if (selected) {
        // Three of four products: the selected subspace is not orbital-transform closed.
        input.structure_data.terms.pop_back();
        input.structure_data.offsets.back() = input.structure_data.terms.size();
        input.structure_data.n_structures = 3;
    }
    const auto& orbitals = input.orbital_preparation_input;
    const SparseParameterLayout layout(orbitals);
    const OrbitalGradientEvaluator evaluator;
    std::vector<int> roots(n_states);
    for (int i = 0; i < n_states; ++i)
        roots[i] = i;
    const std::vector<double> weights(n_states, 1.0 / n_states);
    const auto solved = evaluator.evaluate(input, roots, weights, 0.4);
    const auto& spectrum = solved.second_order_context->root_eigenvectors;
    Eigen::MatrixXd c = spectrum.leftCols(n_states);
    if (!stationary) {
        for (int i = 0; i < n_states; ++i)
            c.col(i) += (0.23 + 0.11 * i) * spectrum.col(n_states);
    }
    // Rebuild every physical gradient/cache at prescribed nonstationary CI;
    // never perturb C while leaving accepted densities/adjoints stale.
    const auto fixed =
        evaluator.evaluate_fixed_structure(input, roots, weights, 0.4, c, nullptr, true);
    const auto context = fixed.gradient.second_order_context;
    const double residual = fixed.structure_residuals.norm();
    require(stationary ? residual < 1e-10 : residual > 1e-2,
            "finite-residual HVP fixture did not attain the requested CI residual");
    const auto& prepared = fixed.gradient.orbital_preparation_result;
    const OrbitalChart chart(orbitals,
                             layout,
                             prepared.auxiliary_orbital_matrix.leftCols(orbitals.n_orbitals),
                             prepared.physical_orbital_frame.normalized_orbital_matrix,
                             nullptr,
                             false,
                             false,
                             nullptr,
                             OrbitalChart::Coordinates::Unwhitened);
    const ExactHvpOperator hessian(context, &input, layout, &chart);
    const StructureTangentOperator structure(context, hessian.structure_action());
    const AffineCoupling affine(*context, hessian, structure);
    const CiTransport transport(
        orbitals, layout, chart, *context, hessian.structure_action(), structure);
    const int no = chart.reduced_size(), ns = structure.tangent_size();
    const auto model = [&](const Eigen::VectorXd& x) {
        const auto point = transport.retract(x.head(no), x.tail(ns));
        auto trial = input;
        trial.orbital_preparation_input = point.input;
        const auto value = evaluator.evaluate_fixed_structure(
            trial, roots, weights, 0.4, point.coefficients, point.selected_metric.get());
        VbScfCoupledKeyframeEvaluation keyframe;
        keyframe.energy = value.gradient.scf_result.total_energy;
        keyframe.orbital_gradient =
            layout.gather_from_full(value.gradient.sparse_orbital_energy_gradient);
        keyframe.normalized_structure_coefficients = value.normalized_coefficients;
        keyframe.structure_residuals = value.structure_residuals;
        keyframe.coefficient_gradient = value.coefficient_gradient;
        return transport.evaluate_model(point, keyframe);
    };
    const auto action = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        const auto blocks = affine.apply(x.head(no), x.tail(ns));
        Eigen::VectorXd result(no + ns);
        result << blocks.orbital.orbital_hessian + blocks.adjoint,
            blocks.forcing + blocks.ci.hessian_coordinates;
        return result;
    };
    Eigen::VectorXd p = Eigen::VectorXd::LinSpaced(no, -0.23, 0.71).normalized();
    Eigen::VectorXd q = Eigen::VectorXd::LinSpaced(ns, 0.63, -0.29).normalized();
    double worst = 0.0, old_error = 0.0, coarse_error = 0.0;
    for (int mode : {0, 1, 2}) {
        Eigen::VectorXd direction(no + ns);
        direction << p, q;
        if (mode == 0)
            direction.tail(ns).setZero();
        if (mode == 1)
            direction.head(no).setZero();
        const Eigen::VectorXd analytic = action(direction);
        for (double step : {2e-4, 1e-4}) {
            const Eigen::VectorXd fd =
                (model(step * direction).gradient - model(-step * direction).gradient) / (2 * step);
            const double error = (analytic - fd).norm() / std::max(1.0, fd.norm());
            std::cout << "affine_hvp states=" << n_states << " selected=" << selected
                      << " stationary=" << stationary << " mode=" << mode << " step=" << step
                      << " residual=" << residual << " error=" << error << '\n';
            if (step == 1e-4)
                worst = std::max(worst, error);
            else
                coarse_error = std::max(coarse_error, error);
            if (mode == 2 && step == 1e-4) {
                // Negative control: the old, symmetric response coupling misses RD/2.
                const auto orbital = hessian.apply_orbital_coupling(p);
                const auto ci = structure.apply_coupling_coordinates(q);
                Eigen::VectorXd old(no + ns);
                old << orbital.orbital_hessian +
                           hessian.apply_structure_coupling_adjoint(ci.coefficient_response,
                                                                    ci.adjoint_multipliers),
                    structure.project_coordinates(orbital.scaled_structure_forcing) +
                        ci.hessian_coordinates;
                old_error = (old - fd).norm() / std::max(1.0, fd.norm());
            }
        }
    }
    require(worst < 2e-7, "production affine HVP differs from keyframe-gradient FD");
    require(worst < 0.4 * coarse_error || worst < 2e-9,
            "keyframe-gradient differences do not show central-FD convergence");
    require(stationary ? old_error < 2e-7 : old_error > 1e-4,
            "negative control did not expose the finite-residual normalization mismatch");
    Eigen::VectorXd x(no + ns), y(no + ns);
    x << p, 0.3 * q;
    y << -0.4 * p, q;
    require(std::abs(x.dot(action(y)) - y.dot(action(x))) < 2e-10,
            "affine forward and reverse coupling are not adjoints");
    // Independently check the actual balanced keyframe energy derivative.
    const auto z = 0.015 * x;
    constexpr double step = 1e-5;
    require(std::abs((model(z + step * y).energy - model(z - step * y).energy) / (2 * step) -
                     model(z).gradient.dot(y)) < 2e-7,
            "physical keyframe gradient fails model-energy finite differences");
    std::cout << "affine_summary states=" << n_states << " selected=" << selected
              << " stationary=" << stationary << " error=" << worst << " old_error=" << old_error
              << '\n';
}

} // namespace

int main() {
    try {
        const StructureFixture fixture = make_fixture();
        check_single_state(fixture);
        check_equal_weight_multistate(fixture);
        check_unequal_weights_rejected(fixture);
        check_coupled_neo_coordinates(fixture);
        check_connection(fixture);
        check_carrier_diagonal(fixture);
        check_metric_factorization(fixture);
        check_joint_coupling_action();
        for (int n_states : {1, 2})
            for (bool selected : {false, true})
                for (bool stationary : {false, true})
                    check_affine_keyframe_hvp(n_states, selected, stationary);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
