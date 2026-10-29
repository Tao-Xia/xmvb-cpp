#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "core/timing.hpp"

#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/derivatives/hessian/coupled/structure.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "integrals/active/two_electron.hpp"
#include "vbscf/optimization/coupled/ci_transport.hpp"
#include "vbscf/optimization/objective/function.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"
#include "vbscf/orbitals/preparation/cholesky.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "vbscf/structures/overlap/representation.hpp"

namespace {

using namespace xmvb::vb;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Actual, typename Reference>
double relative_error(const Eigen::MatrixBase<Actual>& actual,
                      const Eigen::MatrixBase<Reference>& reference) {
    return (actual - reference).norm() / std::max(1.0, reference.norm());
}

OrbitalPreparationInput make_input() {
    OrbitalPreparationInput input;
    input.n_basis_functions = 5;
    input.n_orbitals = 4;
    input.n_active_orbitals = 3;
    input.n_active_electrons = 2;
    input.n_total_electrons = 4;
    input.orbital_type = 1;
    input.ao_overlap_matrix = Eigen::MatrixXd::Identity(5, 5);
    input.ao_overlap_matrix.array() += 0.05;
    input.ao_overlap_matrix(0, 2) += 0.04;
    input.ao_overlap_matrix(2, 0) += 0.04;
    input.ao_overlap_matrix(1, 4) -= 0.03;
    input.ao_overlap_matrix(4, 1) -= 0.03;
    input.orbital_basis_counts = {5, 5, 5, 5};
    input.orbital_basis_index_table = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    input.orbital_value_table = {1.20,  0.14, -0.10, 0.08, 0.03,  0.11, 1.10,  0.23, -0.16, 0.09,
                                 -0.16, 0.19, 1.18,  0.26, -0.12, 0.07, -0.18, 0.25, 1.07,  0.34};
    return input;
}

Eigen::MatrixXd selected_w() {
    Eigen::MatrixXd w(9, 3);
    for (int product = 0; product < 9; ++product) {
        w(product, 0) = std::sin(0.41 * (product + 1));
        w(product, 1) = std::cos(0.29 * (product + 2));
        w(product, 2) = std::sin(0.17 * (product + 3)) + 0.37 * std::cos(0.53 * (product + 1));
    }
    return w;
}

Eigen::Matrix3d active_overlap(const OrbitalPreparationResult& prepared) {
    require(prepared.active_orbital_overlap_matrix.size() == 9,
            "unexpected active-overlap dimension");
    return Eigen::Map<const Eigen::Matrix3d>(prepared.active_orbital_overlap_matrix.data());
}

std::shared_ptr<const SelectedMetric> selected_factor(const StructureAction& action,
                                                      const SameSpinPairCacheContext& topology,
                                                      const OrbitalPreparationResult& prepared) {
    RepresentationOverlap representation(action,
                                         topology.alpha_reuse_table.unique_determinants,
                                         topology.beta_reuse_table.unique_determinants,
                                         3);
    const auto metric = representation.prepare_metric(overlap_upper(active_overlap(prepared)));
    require(metric.selected_metric() != nullptr, "small selected metric did not retain its factor");
    return metric.selected_metric();
}

void check_horizontal_inverse(const StructureTangentOperator& structure,
                              const Eigen::MatrixXd& ambient_metric) {
    const int dimension = structure.tangent_size();
    Eigen::MatrixXd metric(dimension, dimension);
    auto& phases = xmvb::profiling::phase_totals();
    const bool enabled = phases.enabled;
    phases.enabled = true;
    const auto phase = static_cast<std::size_t>(xmvb::profiling::Phase::CiProductGenerate);
    const auto before = phases.calls[phase].load();
    for (int column = 0; column < dimension; ++column) {
        const Eigen::VectorXd axis = Eigen::VectorXd::Unit(dimension, column);
        metric.col(column) = structure.apply_metric_coordinates(axis);
        const auto raw = structure.expand(axis).scaled_coefficients;
        require(relative_error(metric.col(column),
                               structure.project_coordinates(ambient_metric * raw)) < 2e-11,
                "shared Gram changed the physical horizontal metric");
    }
    const auto after = phases.calls[phase].load();
    phases.enabled = enabled;
    require(before == after, "shared metric replayed a product transform");
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(dimension, 0.27, -0.41);
    const Eigen::VectorXd actual = structure.apply_inverse_metric_model(rhs);
    const Eigen::VectorXd expected = metric.llt().solve(rhs);
    require(relative_error(actual, expected) < 2e-11 && (metric * actual - rhs).norm() < 2e-11,
            "shared selected factor changed the horizontal inverse");
}

// The 1-alpha/1-beta full ambient carrier has nine indexed spin products.
// Form its selected columns directly from the independent LLT and W entries.
Eigen::MatrixXd explicit_carrier(const OrbitalPreparationInput& input, const Eigen::MatrixXd& w) {
    const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
    Eigen::LLT<Eigen::Matrix3d> factor(active_overlap(prepared));
    require(factor.info() == Eigen::Success, "active overlap lost positive rank");
    const Eigen::Matrix3d r = factor.matrixU();
    Eigen::MatrixXd f = Eigen::MatrixXd::Zero(9, 3);
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    f.row(3 * a + b).noalias() += r(a, i) * r(b, j) * w.row(3 * i + j);
                }
            }
        }
    }
    return f;
}

Eigen::Matrix3d model_hamiltonian() {
    Eigen::Matrix3d k;
    k << -1.10, 0.27, -0.13, 0.27, -0.37, 0.22, -0.13, 0.22, 0.61;
    return k;
}

double fixed_energy(const OrbitalPreparationInput& input,
                    const SparseParameterLayout& layout,
                    const Eigen::Vector3d& coefficients,
                    const Eigen::MatrixXd& w,
                    const Eigen::Matrix3d& k) {
    const Eigen::MatrixXd f = explicit_carrier(input, w);
    const double norm_squared = (f * coefficients).squaredNorm();
    require(norm_squared > 1e-8, "selected state lost positive norm");
    return coefficients.dot(k * coefficients) / norm_squared +
           0.1 * layout.pack(input).squaredNorm();
}

Eigen::VectorXd fixed_raw_gradient(const OrbitalPreparationInput& input,
                                   const SparseParameterLayout& layout,
                                   const Eigen::Vector3d& coefficients,
                                   const Eigen::MatrixXd& w,
                                   const Eigen::Matrix3d& k) {
    constexpr double h = 1e-5;
    Eigen::VectorXd gradient(layout.size());
    const Eigen::VectorXd packed = layout.pack(input);
    for (int j = 0; j < layout.size(); ++j) {
        auto plus = input;
        auto minus = input;
        const Eigen::VectorXd axis = Eigen::VectorXd::Unit(layout.size(), j);
        layout.unpack(packed + h * axis, &plus);
        layout.unpack(packed - h * axis, &minus);
        gradient[j] = (fixed_energy(plus, layout, coefficients, w, k) -
                       fixed_energy(minus, layout, coefficients, w, k)) /
                      (2.0 * h);
    }
    return gradient;
}

StructureAction make_action(const Eigen::MatrixXd& w,
                            const Eigen::Matrix3d& overlap_matrix,
                            const StructureDiagonal& diagonal,
                            SameSpinPairCacheContext* topology_out) {
    std::vector<std::vector<int>> alpha, beta;
    std::vector<std::vector<StructureExpansionTerm>> terms(9);
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            const int product = 3 * a + b;
            alpha.push_back({a});
            beta.push_back({b});
            for (int s = 0; s < 3; ++s)
                terms[product].push_back({s, w(product, s)});
        }
    }
    *topology_out = build_same_spin_pair_topology(alpha, beta, 3);
    const std::vector<double> packed_overlap(overlap_matrix.data(),
                                             overlap_matrix.data() + overlap_matrix.size());
    xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
    two_electron.packed_active_two_electron_integrals.assign(
        xmvb::integrals::packed_active_two_electron_integral_count(3), 0.0);
    return StructureAction(terms,
                           3,
                           *topology_out,
                           packed_overlap,
                           Eigen::Matrix3d::Zero(),
                           two_electron,
                           3,
                           &diagonal,
                           StructureActionMode::Davidson);
}

VbScfCoupledKeyframeEvaluation fake_fixed_evaluation(const CiTransportPoint& point,
                                                     const SparseParameterLayout& layout,
                                                     const Eigen::MatrixXd& w,
                                                     const Eigen::Matrix3d& k) {
    const Eigen::Vector3d c = point.coefficients.col(0);
    const Eigen::MatrixXd f = explicit_carrier(point.input, w);
    const Eigen::Matrix3d m = f.transpose() * f;
    const double norm = std::sqrt(c.dot(m * c));
    const double e_ci = c.dot(k * c) / (norm * norm);
    VbScfCoupledKeyframeEvaluation fixed;
    fixed.energy = e_ci + 0.1 * layout.pack(point.input).squaredNorm();
    fixed.normalized_structure_coefficients = c / norm;
    fixed.structure_residuals = (k - e_ci * m) * fixed.normalized_structure_coefficients;
    fixed.coefficient_gradient = (2.0 / norm) * fixed.structure_residuals;
    fixed.orbital_gradient = fixed_raw_gradient(point.input, layout, c, w, k);
    return fixed;
}

void check_selected_transport(bool stationary) {
    const auto input = make_input();
    const SparseParameterLayout layout(input);
    const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
    const Eigen::MatrixXd w = selected_w();
    const Eigen::MatrixXd f0 = explicit_carrier(input, w);
    const Eigen::Matrix3d m0 = f0.transpose() * f0;
    const Eigen::Matrix3d k = model_hamiltonian();
    Eigen::Vector3d c0(0.64, -0.39, 0.52);
    c0 /= std::sqrt(c0.dot(m0 * c0));
    if (stationary) {
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3d> roots(k, m0);
        require(roots.info() == Eigen::Success, "stationary CI fixture failed");
        c0 = roots.eigenvectors().col(0);
    }
    const double e0 = c0.dot(k * c0);
    const Eigen::Vector3d gc0 = 2.0 * (k - e0 * m0) * c0;
    require(stationary ? gc0.norm() < 1e-11 : gc0.norm() > 0.1,
            "accepted CI fixture has the wrong stationarity");

    const OrbitalChart chart(input,
                             layout,
                             prepared.auxiliary_orbital_matrix.leftCols(4),
                             prepared.physical_orbital_frame.normalized_orbital_matrix,
                             nullptr,
                             false,
                             false,
                             nullptr,
                             OrbitalChart::Coordinates::Unwhitened);
    require(chart.reduced_size() > 0, "empty orbital chart");
    StructureDiagonal diagonal;
    diagonal.hamiltonian = k.diagonal();
    diagonal.overlap = m0.diagonal();
    SameSpinPairCacheContext topology;
    const StructureAction action = make_action(w, active_overlap(prepared), diagonal, &topology);
    require((action.apply_overlap(Eigen::Matrix3d::Identity()) - m0).norm() < 2e-11,
            "production action disagrees with explicit origin overlap");

    auto accepted = std::make_shared<AcceptedPointContext>();
    accepted->complete_active_space = false;
    accepted->prepared_active_space.orbital_result = prepared;
    accepted->prepared_active_space.active_space_two_electron_result.orbital_transform =
        overlap_upper(active_overlap(prepared));
    accepted->same_spin_pair_cache = topology;
    accepted->n_active_orbitals = 3;
    accepted->n_structures = 3;
    accepted->selected_state_indices = {0};
    accepted->normalized_state_weights = {1.0};
    accepted->selected_state_energies = {e0};
    accepted->selected_state_eigenvectors = c0;
    accepted->selected_state_structure_images = StructureActionResult{k * c0, m0 * c0};
    accepted->selected_metric = selected_factor(action, topology, prepared);
    require(relative_error(accepted->selected_metric->overlap, m0) < 2e-11,
            "accepted selected factor differs from the physical overlap");
    const StructureTangentOperator structure(accepted, action);
    require(structure.tangent_size() == 2, "selected three-structure tangent dimension is wrong");
    check_horizontal_inverse(structure, m0);
    const CiTransport transport(input, layout, chart, *accepted, action, structure);

    // Cache an independent five-point derivative of the explicit carrier map.
    // This is the origin tangent A[p] in the finite target, held fixed for all
    // outer finite differences below.
    constexpr double tangent_step = 2e-3;
    std::vector<Eigen::MatrixXd> carrier_basis(chart.reduced_size());
    for (int j = 0; j < chart.reduced_size(); ++j) {
        const Eigen::VectorXd axis = Eigen::VectorXd::Unit(chart.reduced_size(), j);
        const auto carrier_at = [&](double scale) {
            return explicit_carrier(chart.retract_step(input, axis, scale), w);
        };
        carrier_basis[j] = (-carrier_at(2.0 * tangent_step) + 8.0 * carrier_at(tangent_step) -
                            8.0 * carrier_at(-tangent_step) + carrier_at(-2.0 * tangent_step)) /
                           (12.0 * tangent_step);
    }
    const auto independent_coefficient = [&](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
        const Eigen::MatrixXd f = explicit_carrier(chart.retract_step(input, p), w);
        Eigen::VectorXd target = f0 * (c0 + structure.coefficient_response_coordinates(q).col(0));
        for (int j = 0; j < p.size(); ++j)
            target.noalias() += p[j] * carrier_basis[j] * c0;
        return Eigen::Vector3d((f.transpose() * f).ldlt().solve(f.transpose() * target));
    };
    const auto independent_energy = [&](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
        const Eigen::Vector3d c = independent_coefficient(p, q);
        const Eigen::Vector3d affine = c0 + structure.coefficient_response_coordinates(q).col(0);
        return fixed_energy(chart.retract_step(input, p), layout, c, w, k) - gc0.dot(c - affine);
    };
    const auto transport_gradient = [&](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
        const CiTransportPoint point = transport.retract(p, q);
        return transport.evaluate_model(point, fake_fixed_evaluation(point, layout, w, k)).gradient;
    };
    const auto old_gradient = [&](const Eigen::VectorXd& p, const Eigen::VectorXd& q) {
        const auto trial = chart.retract_step(input, p);
        const Eigen::Vector3d c = c0 + structure.coefficient_response_coordinates(q).col(0);
        const Eigen::MatrixXd f = explicit_carrier(trial, w);
        const Eigen::Matrix3d m = f.transpose() * f;
        const double norm = std::sqrt(c.dot(m * c));
        const double e_ci = c.dot(k * c) / (norm * norm);
        const Eigen::Vector3d gc = (2.0 / norm) * (k - e_ci * m) * (c / norm);
        Eigen::VectorXd result(chart.reduced_size() + structure.tangent_size());
        result.head(chart.reduced_size()) =
            chart.project_reduced_gradient(fixed_raw_gradient(trial, layout, c, w, k));
        result.tail(structure.tangent_size()) =
            structure.project_coordinates(Eigen::MatrixXd(gc / std::sqrt(2.0)));
        return result;
    };

    Eigen::VectorXd p = Eigen::VectorXd::LinSpaced(chart.reduced_size(), -0.7, 0.9);
    p.normalize();
    p *= 0.12;
    Eigen::Vector2d q(0.065, -0.047);
    const CiTransportPoint point = transport.retract(p, q);
    const auto trial_factor = RepresentationOverlap(action,
                                                    topology.alpha_reuse_table.unique_determinants,
                                                    topology.beta_reuse_table.unique_determinants,
                                                    3)
                                  .prepare_metric(point.upper)
                                  .selected_metric();
    require(trial_factor != nullptr && trial_factor != accepted->selected_metric &&
                (trial_factor->overlap - accepted->selected_metric->overlap).norm() > 1e-5,
            "changed trial chart reused the accepted metric factor");
    require(point.selected_metric && point.selected_metric != accepted->selected_metric &&
                relative_error(point.selected_metric->overlap, trial_factor->overlap) < 2e-12,
            "transport did not retain the exact trial-chart Gram for its keyframe");
    const auto retained_metric = point.selected_metric;
    const auto next_point = transport.retract(0.7 * p, 0.5 * q);
    require(next_point.selected_metric && next_point.selected_metric != retained_metric &&
                relative_error(retained_metric->overlap, trial_factor->overlap) < 2e-12,
            "a later retraction invalidated an earlier trial's shared Gram");
    const Eigen::Vector3d reference_c = independent_coefficient(p, q);
    require(relative_error(point.coefficients.col(0), reference_c) < 2e-9,
            "finite selected transport differs from explicit least squares");
    const auto model = transport.evaluate_model(point, fake_fixed_evaluation(point, layout, w, k));
    require(std::abs(model.energy - independent_energy(p, q)) < 2e-9,
            "balanced model energy differs from independent selected transport");
    if (!stationary)
        require(std::abs(gc0.dot(reference_c - point.base_coefficients.col(0))) > 1e-6,
                "nonstationary fixture does not exercise the balancing counterterm");
    const Eigen::MatrixXd fp = explicit_carrier(point.input, w);
    const Eigen::VectorXd target = f0 * point.base_coefficients.col(0) + [&]() {
        Eigen::VectorXd ap = Eigen::VectorXd::Zero(9);
        for (int j = 0; j < p.size(); ++j)
            ap.noalias() += p[j] * carrier_basis[j] * c0;
        return ap;
    }();
    const Eigen::VectorXd residual = target - fp * reference_c;
    require(residual.norm() > 1e-5, "arbitrary selected-space fixture accidentally closes");
    require((fp.transpose() * residual).norm() < 2e-10,
            "finite selected transport violates its normal equation");

    constexpr double h = 1e-5;
    const Eigen::VectorXd actual_gradient = transport_gradient(p, q);
    Eigen::VectorXd finite_gradient(actual_gradient.size());
    for (int j = 0; j < finite_gradient.size(); ++j) {
        Eigen::VectorXd pp = p, pm = p;
        Eigen::VectorXd qp = q, qm = q;
        if (j < p.size()) {
            pp[j] += h;
            pm[j] -= h;
        } else {
            qp[j - p.size()] += h;
            qm[j - p.size()] -= h;
        }
        finite_gradient[j] = (independent_energy(pp, qp) - independent_energy(pm, qm)) / (2.0 * h);
    }
    require(relative_error(actual_gradient, finite_gradient) < 5e-7,
            "balanced orbital/CI gradient fails independent model-energy FD");

    const Eigen::VectorXd zero_p = Eigen::VectorXd::Zero(chart.reduced_size());
    const Eigen::VectorXd zero_q = Eigen::VectorXd::Zero(structure.tangent_size());
    require(relative_error(transport_gradient(zero_p, zero_q), old_gradient(zero_p, zero_q)) < 4e-7,
            "transport changed the origin first derivative");
    Eigen::VectorXd up = Eigen::VectorXd::LinSpaced(chart.reduced_size(), 0.8, -0.6);
    up.normalize();
    Eigen::Vector2d uq(0.48, -0.71);
    const auto ci_gradient = transport.origin_structure_gradient();
    require(stationary ? ci_gradient.norm() < 1e-11 : ci_gradient.norm() > 1e-3,
            "balanced model erased the actual origin CI residual");
    constexpr double hvp_step = 2e-4;
    const Eigen::VectorXd new_hvp = (transport_gradient(hvp_step * up, hvp_step * uq) -
                                     transport_gradient(-hvp_step * up, -hvp_step * uq)) /
                                    (2.0 * hvp_step);
    const Eigen::VectorXd old_hvp = (old_gradient(hvp_step * up, hvp_step * uq) -
                                     old_gradient(-hvp_step * up, -hvp_step * uq)) /
                                    (2.0 * hvp_step);
    require(relative_error(old_hvp, new_hvp) < 2e-5,
            "balanced origin HVP differs from the affine physical HVP");
}

/** @brief Two-state transport on a nonclosed three-structure subspace. */
void check_multistate_transport(bool stationary) {
    const auto input = make_input();
    const SparseParameterLayout layout(input);
    const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
    const Eigen::MatrixXd w = selected_w();
    const Eigen::MatrixXd f0 = explicit_carrier(input, w);
    const Eigen::Matrix3d m0 = f0.transpose() * f0;
    const Eigen::Matrix3d k = model_hamiltonian();
    Eigen::MatrixXd c0(3, 2);
    c0 << 0.64, 0.21, -0.39, 0.73, 0.52, -0.31;
    Eigen::LLT<Eigen::MatrixXd> gram(c0.transpose() * m0 * c0);
    c0 = (c0 * gram.matrixU().solve(Eigen::Matrix2d::Identity())).eval();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> roots(c0.transpose() * k * c0);
    c0 = (c0 * roots.eigenvectors()).eval();
    Eigen::Vector2d energies = roots.eigenvalues();
    if (stationary) {
        const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::Matrix3d> full_roots(k, m0);
        require(full_roots.info() == Eigen::Success, "stationary multistate CI fixture failed");
        c0 = full_roots.eigenvectors().leftCols(2);
        energies = full_roots.eigenvalues().head(2);
    }
    const Eigen::MatrixXd gc0 = k * c0 - m0 * c0 * (c0.transpose() * k * c0);
    require(stationary ? gc0.norm() < 1e-11 : gc0.norm() > 1e-3,
            "multistate CI fixture has the wrong stationarity");
    const OrbitalChart chart(input,
                             layout,
                             prepared.auxiliary_orbital_matrix.leftCols(4),
                             prepared.physical_orbital_frame.normalized_orbital_matrix,
                             nullptr,
                             false,
                             false,
                             nullptr,
                             OrbitalChart::Coordinates::Unwhitened);
    StructureDiagonal diagonal;
    diagonal.hamiltonian = k.diagonal();
    diagonal.overlap = m0.diagonal();
    SameSpinPairCacheContext topology;
    const auto action = make_action(w, active_overlap(prepared), diagonal, &topology);
    auto accepted = std::make_shared<AcceptedPointContext>();
    accepted->prepared_active_space.orbital_result = prepared;
    accepted->prepared_active_space.active_space_two_electron_result.orbital_transform =
        overlap_upper(active_overlap(prepared));
    accepted->same_spin_pair_cache = topology;
    accepted->n_active_orbitals = accepted->n_structures = 3;
    accepted->selected_state_indices = {0, 1};
    accepted->normalized_state_weights = {0.5, 0.5};
    accepted->selected_state_energies = {energies[0], energies[1]};
    accepted->selected_state_eigenvectors = c0;
    accepted->selected_state_structure_images = StructureActionResult{k * c0, m0 * c0};
    accepted->selected_metric = selected_factor(action, topology, prepared);
    const StructureTangentOperator structure(accepted, action);
    check_horizontal_inverse(structure, m0);
    const CiTransport transport(input, layout, chart, *accepted, action, structure);
    const auto energy = [&](const OrbitalPreparationInput& trial, const Eigen::MatrixXd& c) {
        const Eigen::MatrixXd f = explicit_carrier(trial, w);
        const Eigen::MatrixXd g = c.transpose() * f.transpose() * f * c;
        return 0.5 * g.ldlt().solve(Eigen::MatrixXd(c.transpose() * k * c)).trace() +
               0.1 * layout.pack(trial).squaredNorm();
    };
    const auto fixed = [&](const CiTransportPoint& point) {
        VbScfCoupledKeyframeEvaluation result;
        const Eigen::MatrixXd& c = point.coefficients;
        const Eigen::MatrixXd f = explicit_carrier(point.input, w);
        const Eigen::MatrixXd m = f.transpose() * f;
        const Eigen::MatrixXd inverse =
            (c.transpose() * m * c).ldlt().solve(Eigen::Matrix2d::Identity());
        result.energy = energy(point.input, c);
        result.coefficient_gradient = (k * c - m * c * inverse * (c.transpose() * k * c)) * inverse;
        const auto packed = layout.pack(point.input);
        result.orbital_gradient.resize(packed.size());
        for (Eigen::Index j = 0; j < packed.size(); ++j) {
            Eigen::VectorXd plus = packed, minus = packed;
            plus[j] += 2e-6;
            minus[j] -= 2e-6;
            auto ip = point.input, im = point.input;
            layout.unpack(plus, &ip);
            layout.unpack(minus, &im);
            result.orbital_gradient[j] = (energy(ip, c) - energy(im, c)) / 4e-6;
        }
        return result;
    };
    Eigen::VectorXd x =
        Eigen::VectorXd::LinSpaced(chart.reduced_size() + structure.tangent_size(), -0.04, 0.05);
    const auto value = [&](const Eigen::VectorXd& v) {
        auto point =
            transport.retract(v.head(chart.reduced_size()), v.tail(structure.tangent_size()));
        return energy(point.input, point.coefficients) -
               (gc0.array() * (point.coefficients - point.base_coefficients).array()).sum();
    };
    const auto point =
        transport.retract(x.head(chart.reduced_size()), x.tail(structure.tangent_size()));
    const auto model = transport.evaluate_model(point, fixed(point));
    const Eigen::VectorXd gradient = model.gradient;
    require(std::abs(model.energy - value(x)) < 2e-11,
            "multistate balanced energy differs from its counterterm definition");
    Eigen::VectorXd numerical(x.size());
    for (Eigen::Index j = 0; j < x.size(); ++j) {
        Eigen::VectorXd plus = x, minus = x;
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        numerical[j] = (value(plus) - value(minus)) / 2e-5;
    }
    require(relative_error(gradient, numerical) < 2e-6,
            "multistate balanced transport gradient fails model-energy finite differences");
    const auto derivative = [&](const Eigen::VectorXd& v, bool transported) {
        CiTransportPoint p;
        if (transported) {
            p = transport.retract(v.head(chart.reduced_size()), v.tail(structure.tangent_size()));
            return transport.evaluate_model(p, fixed(p)).gradient;
        }
        p.input = chart.retract_step(input, v.head(chart.reduced_size()));
        p.coefficients =
            c0 + structure.coefficient_response_coordinates(v.tail(structure.tangent_size()));
        const auto f = fixed(p);
        Eigen::VectorXd g(v.size());
        g.head(chart.reduced_size()) = chart.project_reduced_gradient(f.orbital_gradient);
        g.tail(structure.tangent_size()) = structure.project_coordinates(f.coefficient_gradient);
        return g;
    };
    const Eigen::VectorXd direction = x.normalized();
    const Eigen::VectorXd before =
        (derivative(2e-4 * direction, false) - derivative(-2e-4 * direction, false)) / 4e-4;
    const Eigen::VectorXd after =
        (derivative(2e-4 * direction, true) - derivative(-2e-4 * direction, true)) / 4e-4;
    require(relative_error(before, after) < 4e-5,
            "multistate balanced origin HVP differs from the affine physical HVP");
}

} // namespace

int main() {
    try {
        for (const bool stationary : {false, true}) {
            check_selected_transport(stationary);
            check_multistate_transport(stationary);
        }
        std::cout << "Selected-space balanced CI transport energy, gradient and HVP: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
