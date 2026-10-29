#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>

#include "core/pair_index.hpp"
#include "integrals/active/exact.hpp"
#include "integrals/active/matrix_adjoint.hpp"
#include "integrals/active/one_electron.hpp"
#include "integrals/active/raw.hpp"
#include "integrals/active/ri.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/active/transform.hpp"
#include "integrals/ao/graph.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/ri.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/orbitals/preparation/cholesky.hpp"

namespace {

using xmvb::integrals::ActiveSpaceTwoElectronResult;
using xmvb::integrals::AoIntegralInput;
using xmvb::integrals::RiAoFactorization;

void require_close(const Eigen::MatrixXd& actual,
                   const Eigen::MatrixXd& expected,
                   double tolerance,
                   const char* label) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
        (actual - expected).norm() > tolerance * std::max(1.0, expected.norm())) {
        throw std::runtime_error(label);
    }
}

std::vector<double> pack_pair_adjoint(const Eigen::MatrixXd& gradient) {
    const int pairs = gradient.rows();
    std::vector<double> packed(pairs * (pairs + 1) / 2);
    for (int column = 0; column < pairs; ++column)
        for (int row = 0; row <= column; ++row)
            packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(row, column)] =
                row == column ? gradient(row, column)
                              : gradient(row, column) + gradient(column, row);
    return packed;
}

ActiveSpaceTwoElectronResult exact_at(const AoIntegralInput& ao,
                                      const Eigen::MatrixXd& coefficients) {
    const int n_basis = coefficients.rows();
    const int n_active = coefficients.cols();
    std::vector<std::size_t> row_offsets(n_basis + 1, 0);
    std::vector<int> orbital_indices;
    std::vector<double> values;
    for (int row = 0; row < n_basis; ++row) {
        for (int column = 0; column < n_active; ++column) {
            const double value = coefficients(row, column);
            if (value != 0.0) {
                orbital_indices.push_back(column);
                values.push_back(value);
            }
        }
        row_offsets[row + 1] = values.size();
    }
    return xmvb::integrals::transform_exact_active_integrals(
        ao, row_offsets, orbital_indices, values, n_active);
}

void check_frame(bool ri_backend) {
    constexpr int n_basis = 4;
    constexpr int n_active = 3;
    constexpr int n_aux = 5;
    constexpr int ao_pairs = n_basis * (n_basis + 1) / 2;
    constexpr int active_pairs = n_active * (n_active + 1) / 2;
    RiAoFactorization ri;
    ri.n_basis_functions = n_basis;
    ri.n_auxiliary_functions = n_aux;
    ri.n_packed_ao_pairs = ao_pairs;
    ri.metric_whitened_ao_pair_factors.resize(n_aux, ao_pairs);
    for (int column = 0; column < ao_pairs; ++column)
        for (int row = 0; row < n_aux; ++row)
            ri.metric_whitened_ao_pair_factors(row, column) =
                std::sin(0.19 * (row + 1) * (column + 2));
    const Eigen::MatrixXd ao_kernel =
        ri.metric_whitened_ao_pair_factors.transpose() * ri.metric_whitened_ao_pair_factors;
    std::vector<xmvb::integrals::PairAddress> left, right;
    std::vector<double> values;
    for (int p = 0; p < ao_pairs; ++p)
        for (int q = 0; q <= p; ++q) {
            left.push_back(p);
            right.push_back(q);
            values.push_back(ao_kernel(p, q));
        }
    AoIntegralInput ao;
    ao.n_basis_functions = n_basis;
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, n_basis);

    Eigen::MatrixXd c(n_basis, n_active), dc(n_basis, n_active);
    c << 1.1, 0.2, 0.0, 0.3, 1.0, -0.1, 0.0, 0.2, 0.9, -0.2, 0.0, 0.4;
    for (int column = 0; column < n_active; ++column)
        for (int row = 0; row < n_basis; ++row)
            dc(row, column) = 0.04 * std::cos(0.31 * (row + 2) * (column + 1));
    Eigen::MatrixXd ao_overlap = Eigen::MatrixXd::Identity(n_basis, n_basis);
    for (int column = 0; column < n_basis; ++column)
        for (int row = 0; row < n_basis; ++row)
            ao_overlap(row, column) += 0.03 * std::cos(0.2 * (row + 1) * (column + 1));
    const Eigen::MatrixXd s = c.transpose() * ao_overlap * c;
    const Eigen::MatrixXd ds = dc.transpose() * ao_overlap * c + c.transpose() * ao_overlap * dc;
    const Eigen::MatrixXd r = s.llt().matrixU();
    const Eigen::MatrixXd q =
        r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(n_active, n_active));
    const Eigen::MatrixXd cbar = c * q;
    Eigen::MatrixXd ao_h(n_basis, n_basis);
    for (int column = 0; column < n_basis; ++column)
        for (int row = 0; row < n_basis; ++row)
            ao_h(row, column) = 0.2 * std::sin(0.3 * (row + column + 2));
    ao_h = 0.5 * (ao_h + ao_h.transpose()).eval();
    const xmvb::integrals::ActiveSpaceOneElectronBuilder one_builder;
    const auto raw_one_electron = one_builder.build(ao_h, c);
    const auto framed_one_electron = one_builder.build(ao_h, cbar);
    const Eigen::MatrixXd& h = raw_one_electron.h1e_act;
    const Eigen::MatrixXd dh = dc.transpose() * ao_h * c + c.transpose() * ao_h * dc;
    const Eigen::MatrixXd& framed_h = framed_one_electron.h1e_act;
    require_close(framed_one_electron.ao_image,
                  ao_h * cbar,
                  1e-12,
                  "framed one-electron builder did not retain F Cbar");
    const Eigen::MatrixXd framed_dh =
        (dc * q).transpose() * ao_h * cbar + cbar.transpose() * ao_h * (dc * q);
    require_close(framed_h, q.transpose() * h * q, 1e-12, "fixed-frame one-electron input differs");
    require_close(
        framed_dh, q.transpose() * dh * q, 1e-12, "fixed-frame one-electron direction differs");
    const auto forward = [&](const Eigen::MatrixXd& coefficients) {
        return ri_backend ? xmvb::integrals::transform_ri_active_integrals(ri, coefficients)
                          : exact_at(ao, coefficients);
    };
    const auto raw = forward(c);
    auto framed = forward(cbar);
    framed.orbital_transform = r;
    const std::vector<double> overlap(s.data(), s.data() + s.size());
    const std::vector<double> overlap_direction(ds.data(), ds.data() + ds.size());
    const auto raw_orth = xmvb::vb::orthogonalize_active_integrals(overlap, h, raw, n_active);
    const auto framed_orth =
        xmvb::vb::orthogonalize_active_integrals(overlap, framed_h, framed, n_active);
    require_close(framed_orth.one_electron,
                  raw_orth.one_electron,
                  1e-12,
                  "framed one-electron integrals differ");
    require_close(framed_orth.pair_kernel,
                  raw_orth.pair_kernel,
                  2e-11,
                  "framed two-electron integrals differ");
    if (ri_backend)
        require_close(framed_orth.ri_pair_factors,
                      raw_orth.ri_pair_factors,
                      2e-11,
                      "framed RI factors differ");

    // Raw pair-provider cotangents are packed in the unframed kernel
    // coordinates. Verify their adjoint map against a scalar contraction.
    Eigen::MatrixXd framed_pair_direction(active_pairs, active_pairs);
    for (int column = 0; column < active_pairs; ++column)
        for (int row = 0; row < active_pairs; ++row)
            framed_pair_direction(row, column) = 0.5 * (std::sin(0.23 * (row + 1) * (column + 2)) +
                                                        std::sin(0.23 * (column + 1) * (row + 2)));
    std::vector<double> raw_pair_gradient(active_pairs * (active_pairs + 1) / 2);
    for (std::size_t index = 0; index < raw_pair_gradient.size(); ++index)
        raw_pair_gradient[index] = std::cos(0.29 * (index + 1));
    const auto framed_pair_gradient =
        xmvb::integrals::materialize_framed_pair_kernel_adjoint(raw_pair_gradient, r);
    const Eigen::MatrixXd raw_pair_direction =
        xmvb::integrals::transform_pair_kernel(framed_pair_direction, r);
    const auto pack_kernel = [](const Eigen::MatrixXd& kernel) {
        std::vector<double> packed(kernel.rows() * (kernel.rows() + 1) / 2);
        for (int column = 0; column < kernel.cols(); ++column)
            for (int row = 0; row <= column; ++row)
                packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(row, column)] =
                    kernel(row, column);
        return packed;
    };
    const auto packed_raw_delta = pack_kernel(raw_pair_direction);
    const auto packed_framed_delta = pack_kernel(framed_pair_direction);
    double raw_contraction = 0.0, framed_contraction = 0.0;
    for (std::size_t index = 0; index < raw_pair_gradient.size(); ++index) {
        raw_contraction += raw_pair_gradient[index] * packed_raw_delta[index];
        framed_contraction += framed_pair_gradient[index] * packed_framed_delta[index];
    }
    if (std::abs(raw_contraction - framed_contraction) >
        1e-11 * std::max(1.0, std::abs(raw_contraction)))
        throw std::runtime_error("packed pair adjoint frame changed scalar contraction");

    Eigen::MatrixXd one_adjoint(n_active, n_active), pair_adjoint(active_pairs, active_pairs),
        generator(n_active, n_active);
    for (int column = 0; column < n_active; ++column)
        for (int row = 0; row < n_active; ++row) {
            one_adjoint(row, column) = std::sin(0.13 * (row + 2) * (column + 1));
            generator(row, column) = std::cos(0.17 * (row + 1) * (column + 3));
        }
    for (int column = 0; column < active_pairs; ++column)
        for (int row = 0; row < active_pairs; ++row)
            pair_adjoint(row, column) = std::cos(0.11 * (row + 1) * (column + 2));
    const auto raw_adjoint = xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
        raw_orth, one_adjoint, pair_adjoint, generator);
    const auto framed_adjoint = xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
        framed_orth, one_adjoint, pair_adjoint, generator);
    require_close(
        framed_adjoint.overlap, raw_adjoint.overlap, 1e-12, "framed overlap adjoint differs");
    require_close(framed_adjoint.one_electron,
                  raw_adjoint.one_electron,
                  1e-12,
                  "framed one-electron adjoint did not return the raw input cotangent");
    const Eigen::MatrixXd symmetric_one_adjoint = 0.5 * (one_adjoint + one_adjoint.transpose());
    const Eigen::MatrixXd expected_raw_one_adjoint = q * symmetric_one_adjoint * q.transpose();
    require_close(framed_adjoint.one_electron,
                  expected_raw_one_adjoint,
                  1e-12,
                  "framed one-electron adjoint has the wrong raw frame");
    const Eigen::MatrixXd raw_one_coefficient_gradient =
        ao_h * c * (raw_adjoint.one_electron + raw_adjoint.one_electron.transpose());
    const Eigen::MatrixXd framed_one_coefficient_gradient =
        ao_h * cbar * (2.0 * symmetric_one_adjoint) * q.transpose();
    require_close(framed_one_coefficient_gradient,
                  raw_one_coefficient_gradient,
                  1e-12,
                  "framed one-electron physical orbital pullback differs");
    require_close(cbar * symmetric_one_adjoint * cbar.transpose(),
                  c * raw_adjoint.one_electron * c.transpose(),
                  1e-12,
                  "framed one-electron AO cotangent differs");
    const std::vector<double> overlap_weight(
        raw_adjoint.overlap.data(), raw_adjoint.overlap.data() + raw_adjoint.overlap.size());
    const std::vector<double> one_weight(raw_adjoint.one_electron.data(),
                                         raw_adjoint.one_electron.data() +
                                             raw_adjoint.one_electron.size());
    Eigen::MatrixXd raw_auxiliary = Eigen::MatrixXd::Identity(n_basis, n_basis);
    raw_auxiliary.leftCols(n_active) = c;
    const auto independent_raw_pullback =
        xmvb::integrals::ActiveSpaceMatrixBackpropagator{}.backpropagate(
            overlap_weight, one_weight, ao_overlap, ao_h, raw_auxiliary, n_basis, 0, n_active);
    const auto framed_pullback = xmvb::integrals::pullback_active_matrices(
        overlap_weight, one_weight, ao_overlap, c, cbar, framed_one_electron, r, true);
    require_close(framed_pullback.active_auxiliary_orbital_gradient,
                  independent_raw_pullback.active_auxiliary_orbital_gradient,
                  1e-12,
                  "fused active-matrix orbital pullback differs from the raw builder");
    if (framed_pullback.ao_effective_one_electron_gradient.size() !=
        static_cast<std::size_t>(n_basis * n_basis)) {
        throw std::runtime_error("fused active-matrix AO pullback has the wrong size");
    }
    const Eigen::Map<const Eigen::MatrixXd> framed_ao_pullback(
        framed_pullback.ao_effective_one_electron_gradient.data(), n_basis, n_basis);
    const Eigen::Map<const Eigen::MatrixXd> raw_ao_pullback(
        independent_raw_pullback.ao_effective_one_electron_gradient.data(), n_basis, n_basis);
    require_close(framed_ao_pullback,
                  raw_ao_pullback,
                  1e-12,
                  "fused active-matrix AO pullback differs from the raw builder");
    const auto framed_pullback_without_ao = xmvb::integrals::pullback_active_matrices(
        overlap_weight, one_weight, ao_overlap, c, cbar, framed_one_electron, r, false);
    require_close(framed_pullback_without_ao.active_auxiliary_orbital_gradient,
                  independent_raw_pullback.active_auxiliary_orbital_gradient,
                  1e-12,
                  "fused active-matrix orbital pullback changed without AO adjoint");
    if (!framed_pullback_without_ao.ao_effective_one_electron_gradient.empty()) {
        throw std::runtime_error("fused active-matrix pullback built an unrequested AO adjoint");
    }
    require_close(framed_adjoint.pair_kernel,
                  0.5 * (pair_adjoint + pair_adjoint.transpose()),
                  1e-12,
                  "framed pair adjoint was transformed twice");

    Eigen::MatrixXd raw_coefficient_gradient, framed_coefficient_gradient;
    if (ri_backend) {
        const auto raw_cache =
            xmvb::integrals::build_ri_active_two_electron_response_cache(ri, raw, n_active);
        const auto framed_cache =
            xmvb::integrals::build_ri_active_two_electron_response_cache(ri, framed, n_active);
        raw_coefficient_gradient =
            xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                pack_pair_adjoint(raw_adjoint.pair_kernel), raw_cache);
        framed_coefficient_gradient =
            xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                pack_pair_adjoint(framed_adjoint.pair_kernel), framed_cache);
    } else {
        raw_coefficient_gradient =
            xmvb::integrals::backpropagate_exact_packed_active_two_electron_gradient(
                pack_pair_adjoint(raw_adjoint.pair_kernel), ao, raw);
        framed_coefficient_gradient =
            xmvb::integrals::backpropagate_exact_packed_active_two_electron_gradient(
                pack_pair_adjoint(framed_adjoint.pair_kernel), ao, framed);
    }
    require_close(framed_coefficient_gradient * q.transpose(),
                  raw_coefficient_gradient,
                  2e-10,
                  "framed physical orbital pullback differs");
    const Eigen::MatrixXd raw_full_coefficient_gradient =
        raw_one_coefficient_gradient +
        ao_overlap * c * (raw_adjoint.overlap + raw_adjoint.overlap.transpose()) +
        raw_coefficient_gradient;
    const Eigen::MatrixXd framed_full_coefficient_gradient =
        framed_one_coefficient_gradient +
        ao_overlap * c * (framed_adjoint.overlap + framed_adjoint.overlap.transpose()) +
        framed_coefficient_gradient * q.transpose();
    require_close(framed_full_coefficient_gradient,
                  raw_full_coefficient_gradient,
                  2e-10,
                  "framed full physical orbital gradient differs");

    constexpr double step = 1e-6;
    const auto raw_plus = forward(c + step * dc);
    const auto raw_minus = forward(c - step * dc);
    const auto framed_plus = forward(cbar + step * dc * q);
    const auto framed_minus = forward(cbar - step * dc * q);
    std::vector<double> raw_delta, framed_delta;
    Eigen::MatrixXd raw_factor_delta, framed_factor_delta;
    if (ri_backend) {
        raw_factor_delta =
            (raw_plus.ri_active_pair_factors - raw_minus.ri_active_pair_factors) / (2 * step);
        framed_factor_delta =
            (framed_plus.ri_active_pair_factors - framed_minus.ri_active_pair_factors) / (2 * step);
    } else {
        raw_delta.resize(raw_plus.packed_active_two_electron_integrals.size());
        framed_delta.resize(framed_plus.packed_active_two_electron_integrals.size());
        for (std::size_t index = 0; index < raw_delta.size(); ++index) {
            raw_delta[index] = (raw_plus.packed_active_two_electron_integrals[index] -
                                raw_minus.packed_active_two_electron_integrals[index]) /
                               (2 * step);
            framed_delta[index] = (framed_plus.packed_active_two_electron_integrals[index] -
                                   framed_minus.packed_active_two_electron_integrals[index]) /
                                  (2 * step);
        }
    }
    const auto raw_jet =
        xmvb::vb::orthogonalize_active_integral_direction(raw_orth,
                                                          overlap_direction,
                                                          dh,
                                                          raw_delta,
                                                          n_active,
                                                          ri_backend ? &raw_factor_delta : nullptr);
    const auto framed_jet = xmvb::vb::orthogonalize_active_integral_direction(
        framed_orth,
        overlap_direction,
        framed_dh,
        framed_delta,
        n_active,
        ri_backend ? &framed_factor_delta : nullptr);
    require_close(framed_jet.orbital_transform,
                  raw_jet.orbital_transform,
                  1e-12,
                  "framed Cholesky direction differs");
    require_close(framed_jet.one_electron,
                  raw_jet.one_electron,
                  1e-12,
                  "framed one-electron direction differs");
    const Eigen::MatrixXd plus_c = c + step * dc;
    const Eigen::MatrixXd minus_c = c - step * dc;
    const Eigen::MatrixXd plus_s = plus_c.transpose() * ao_overlap * plus_c;
    const Eigen::MatrixXd minus_s = minus_c.transpose() * ao_overlap * minus_c;
    const Eigen::MatrixXd plus_q =
        plus_s.llt().matrixU().solve(Eigen::MatrixXd::Identity(n_active, n_active));
    const Eigen::MatrixXd minus_q =
        minus_s.llt().matrixU().solve(Eigen::MatrixXd::Identity(n_active, n_active));
    const Eigen::MatrixXd plus_h = (plus_c * plus_q).transpose() * ao_h * (plus_c * plus_q);
    const Eigen::MatrixXd minus_h = (minus_c * minus_q).transpose() * ao_h * (minus_c * minus_q);
    require_close(framed_jet.one_electron,
                  (plus_h - minus_h) / (2 * step),
                  2e-8,
                  "framed one-electron direction missed the changing Cholesky frame");
    require_close(
        framed_jet.pair_kernel, raw_jet.pair_kernel, 2e-8, "framed two-electron direction differs");

    const Eigen::MatrixXd zero_one = Eigen::MatrixXd::Zero(n_active, n_active);
    const Eigen::MatrixXd zero_pair = Eigen::MatrixXd::Zero(active_pairs, active_pairs);
    const auto raw_adjoint_jet =
        xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(
            raw_orth, raw_jet, one_adjoint, pair_adjoint, generator, zero_one, zero_pair, zero_one);
    const auto framed_adjoint_jet =
        xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(framed_orth,
                                                                             framed_jet,
                                                                             one_adjoint,
                                                                             pair_adjoint,
                                                                             generator,
                                                                             zero_one,
                                                                             zero_pair,
                                                                             zero_one);
    require_close(framed_adjoint_jet.overlap,
                  raw_adjoint_jet.overlap,
                  2e-8,
                  "framed overlap-adjoint direction differs");
    require_close(framed_adjoint_jet.one_electron,
                  raw_adjoint_jet.one_electron,
                  2e-8,
                  "framed one-electron raw-cotangent direction differs");
    const Eigen::MatrixXd k = framed_jet.orbital_transform * q;
    const Eigen::MatrixXd expected_raw_one_adjoint_direction =
        q * (-(k * symmetric_one_adjoint + symmetric_one_adjoint * k.transpose())) * q.transpose();
    require_close(framed_adjoint_jet.one_electron,
                  expected_raw_one_adjoint_direction,
                  2e-8,
                  "framed one-electron raw-cotangent direction missed the frame term");
    require_close(xmvb::integrals::pullback_pair_kernel(framed_adjoint_jet.pair_kernel, q),
                  raw_adjoint_jet.pair_kernel,
                  2e-8,
                  "framed pair-adjoint direction differs");
}

/** @brief A resolved near-dependent frame must retain its physical AO metric. */
void check_conditioned_frame() {
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(5, 3);
    phi.col(0) << 1, .2, .3, -.1, .4;
    phi.col(1) = phi.col(0);
    phi(1, 1) += 1e-4;
    phi.col(2) << .1, -.4, .3, .6, -.2;
    const Eigen::MatrixXd metric =
        (Eigen::VectorXd(5) << 1., 1.5, 2., 2.5, 3.).finished().asDiagonal();
    const Eigen::MatrixXd gram = phi.transpose() * metric * phi;
    const auto frame = xmvb::vb::active_frame(phi, metric, gram);
    require_close(frame.coefficients.transpose() * metric * frame.coefficients,
                  Eigen::MatrixXd::Identity(3, 3),
                  2e-13,
                  "refined active frame lost orthogonality");
    require_close(frame.coefficients * frame.upper,
                  phi,
                  2e-14,
                  "refined active frame changed physical orbitals");
    // Positive diagonal fixes the same smooth Cholesky gauge, including its VJP.
    const Eigen::MatrixXd dphi = .01 * Eigen::MatrixXd::Identity(5, 3);
    const Eigen::MatrixXd dgram = dphi.transpose() * metric * phi + phi.transpose() * metric * dphi;
    const Eigen::MatrixXd analytic = xmvb::vb::overlap_upper_direction(frame.upper, dgram);
    const double step = 1e-7;
    const Eigen::MatrixXd plus = phi + step * dphi, minus = phi - step * dphi;
    const auto rp = xmvb::vb::active_frame(plus, metric, plus.transpose() * metric * plus);
    const auto rm = xmvb::vb::active_frame(minus, metric, minus.transpose() * metric * minus);
    require_close((rp.upper - rm.upper) / (2 * step),
                  analytic,
                  2e-6,
                  "refined frame direction differs from physical finite difference");
}

} // namespace

int main() {
    try {
        check_conditioned_frame();
        check_frame(false);
        check_frame(true);
        std::cout << "fused exact/RI integral frame: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
