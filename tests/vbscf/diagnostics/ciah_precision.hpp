#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <array>
#include <fstream>
#include <iostream>
#include <span>
#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "ciah_internal.hpp"
#include "ciah_state_metric.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"

namespace xmvb::test {
using ExtendedMatrix = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;

/** @brief Independent tiny one-body action with canonical fermion signs. */
inline ExtendedMatrix extended_generator(const mcscf::SpinStrings& alpha,
                                         const mcscf::SpinStrings& beta,
                                         const ExtendedMatrix& state,
                                         const ExtendedMatrix& generator) {
    ExtendedMatrix result = ExtendedMatrix::Zero(state.rows(), state.cols());
    for (bool use_beta : {false, true}) {
        const auto& strings = use_beta ? beta : alpha;
        for (Eigen::Index source = 0; source < strings.size(); ++source) {
            const auto mask = strings.masks()[source];
            for (int q : strings.occupied(source)) {
                const auto removed = mask ^ (std::uint64_t{1} << q);
                const int annihilation = std::popcount(mask & ((std::uint64_t{1} << q) - 1));
                for (int p = 0; p < strings.orbitals(); ++p) {
                    if (removed & (std::uint64_t{1} << p))
                        continue;
                    const auto target = strings.index(removed | (std::uint64_t{1} << p));
                    if (target < 0)
                        throw std::runtime_error(
                            "precision reference requires complete spin strings");
                    const int creation = std::popcount(removed & ((std::uint64_t{1} << p) - 1));
                    const long double value =
                        ((creation + annihilation) % 2 ? -1 : 1) * generator(p, q);
                    if (use_beta)
                        result.col(target) += value * state.col(source);
                    else
                        result.row(target) += value * state.row(source);
                }
            }
        }
    }
    return result;
}

/** @brief Precision attribution, never a molecular execution path.
 * Fix the same rounded active Hamiltonian and compare ordinary arithmetic
 * against extended-precision selected expansion, compounds and generators.
 * A dense CI operator is explicitly bounded to 512 amplitudes; no orbital
 * Hessian is assembled and no AO response is repeated.
 */
inline void audit_internal_precision(const vb::OrbitalPreparationInput& input,
                                     const vb::OrbitalPreparationResult& prepared,
                                     const vb::AcceptedPointContext& context,
                                     const vb::StructureAction& action,
                                     const vb::StructureTangentOperator& structure,
                                     const Eigen::MatrixXd& coefficients,
                                     const Eigen::MatrixXd& state,
                                     const Eigen::MatrixXd& basis,
                                     std::span<const InternalStateDirection> jets,
                                     const Eigen::VectorXd& production_gradient) {
    const auto view = action.orthogonal_view();
    const auto rows = state.rows(), cols = state.cols(), size = state.size();
    if (size > 512) {
        std::cout << "PRECISION_REFERENCE,skipped=bounded_carrier" << std::endl;
        return;
    }
    const auto na = input.n_active_orbitals;
    const auto nc = (input.n_total_electrons - input.n_active_electrons) / 2;
    const Eigen::MatrixXd phi = prepared.auxiliary_orbital_matrix.middleCols(nc, na);
    const ExtendedMatrix s = input.ao_overlap_matrix.cast<long double>();
    const ExtendedMatrix phi_extended = phi.cast<long double>();
    const Eigen::LLT<ExtendedMatrix> accurate(phi_extended.transpose() * s * phi_extended);
    if (accurate.info() != Eigen::Success)
        throw std::runtime_error("extended active Gram factorization failed");
    const ExtendedMatrix ir = accurate.matrixU().solve(ExtendedMatrix::Identity(na, na));
    const Eigen::LLT<Eigen::MatrixXd> ao_factor(input.ao_overlap_matrix);
    const Eigen::MatrixXd weighted = ao_factor.matrixU() * phi;
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(weighted);
    Eigen::MatrixXd r_qr = qr.matrixQR().topRows(na).triangularView<Eigen::Upper>();
    for (Eigen::Index i = 0; i < r_qr.rows(); ++i)
        if (r_qr(i, i) < 0)
            r_qr.row(i) *= -1;
    const auto& r = view.alpha.orbital_transform();
    const auto defect = [&](const Eigen::MatrixXd& upper) {
        const ExtendedMatrix inverse =
            upper.cast<long double>().triangularView<Eigen::Upper>().solve(
                ExtendedMatrix::Identity(na, na));
        const ExtendedMatrix chi = phi_extended * inverse;
        return (chi.transpose() * s * chi - ExtendedMatrix::Identity(na, na)).norm();
    };
    const Eigen::MatrixXd double_inverse =
        r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(na, na));
    const Eigen::MatrixXd first_orthogonal = phi * double_inverse;
    const Eigen::LLT<Eigen::MatrixXd> refinement(first_orthogonal.transpose() *
                                                 input.ao_overlap_matrix * first_orthogonal);
    const Eigen::MatrixXd refined_upper = refinement.matrixU() * r;
    const Eigen::MatrixXd refined_orthogonal =
        first_orthogonal * refinement.matrixU().solve(Eigen::MatrixXd::Identity(na, na));
    const ExtendedMatrix refined = refined_orthogonal.cast<long double>();
    std::cout << "PRECISION_FRAME,gram_orthogonality=" << defect(r)
              << ",qr_orthogonality=" << defect(r_qr)
              << ",refined_factor_orthogonality=" << defect(refined_upper)
              << ",refined_frame_orthogonality="
              << (refined.transpose() * s * refined - ExtendedMatrix::Identity(na, na)).norm()
              << ",gram_relative_factor="
              << (r.cast<long double>() * ir - ExtendedMatrix::Identity(na, na)).norm()
              << ",qr_relative_factor="
              << (r_qr.cast<long double>() * ir - ExtendedMatrix::Identity(na, na)).norm()
              << std::endl;

    const ExtendedMatrix ca = reference_compound(view.alpha.strings(), r.cast<long double>());
    const ExtendedMatrix cb = reference_compound(view.beta.strings(), r.cast<long double>());
    const auto expand = [&](const Eigen::MatrixXd& c) {
        ExtendedMatrix raw = ExtendedMatrix::Zero(rows, cols);
        for (std::size_t group = 0; group < view.products.size(); ++group) {
            long double value = 0;
            for (std::size_t j = view.offsets[group]; j < view.offsets[group + 1]; ++j)
                value += static_cast<long double>(view.terms[j].coefficient) *
                         c(view.terms[j].structure, 0);
            raw(view.products[group] / cols, view.products[group] % cols) = value;
        }
        return ExtendedMatrix(ca * raw * cb.transpose());
    };
    ExtendedMatrix y = expand(coefficients);
    const long double norm = y.norm();
    y /= norm;
    std::cout << "PRECISION_STATE,norm=" << norm
              << ",difference=" << (y - state.cast<long double>()).norm() << std::endl;
    Eigen::MatrixXd h(size, size);
    constexpr Eigen::Index block_width = 8;
    for (Eigen::Index first = 0; first < size; first += block_width) {
        const auto width = std::min(block_width, size - first);
        Eigen::MatrixXd inputs = Eigen::MatrixXd::Zero(rows, width * cols);
        for (Eigen::Index j = 0; j < width; ++j)
            inputs((first + j) % rows, j * cols + (first + j) / rows) = 1;
        const Eigen::MatrixXd output = view.sigma.apply(inputs);
        for (Eigen::Index j = 0; j < width; ++j)
            h.col(first + j) = Eigen::Map<const Eigen::VectorXd>(output.data() + j * size, size);
    }
    // Bounded fixed-span model for independent nonlinear minimization on Slurm.
    // All arrays are column-major; no molecular optimizer consumes this file.
    Eigen::MatrixXd bare = Eigen::MatrixXd::Zero(size, coefficients.rows());
    for (std::size_t group = 0; group < view.products.size(); ++group) {
        const auto row = view.products[group] / cols + rows * (view.products[group] % cols);
        for (auto term = view.offsets[group]; term < view.offsets[group + 1]; ++term)
            bare(row, view.terms[term].structure) += view.terms[term].coefficient;
    }
    std::ofstream file("internal_reference.bin", std::ios::binary);
    const std::array<std::uint64_t, 6> header = {0x5642494e54455231ULL,
                                                 na,
                                                 static_cast<std::uint64_t>(rows),
                                                 static_cast<std::uint64_t>(cols),
                                                 static_cast<std::uint64_t>(coefficients.rows()),
                                                 1};
    const auto write = [&](const auto* pointer, std::size_t count) {
        file.write(reinterpret_cast<const char*>(pointer), sizeof(*pointer) * count);
    };
    write(header.data(), header.size());
    write(view.alpha.strings().masks().data(), rows);
    write(view.beta.strings().masks().data(), cols);
    write(r.data(), r.size());
    write(h.data(), h.size());
    write(bare.data(), bare.size());
    write(coefficients.data(), coefficients.size());
    if (!file)
        throw std::runtime_error("cannot write bounded internal reference");
    const double skew = (h - h.transpose()).norm();
    const ExtendedMatrix hermitian = (.5 * (h + h.transpose())).eval().cast<long double>();
    const Eigen::Map<const Eigen::Matrix<long double, Eigen::Dynamic, 1>> flat(y.data(), size);
    const auto hy = (hermitian * flat).eval();
    const long double energy = flat.dot(hy);
    const auto residual = (hy - energy * flat).eval();
    const auto energy_at = [&](ExtendedMatrix value) {
        value /= value.norm();
        const Eigen::Map<const Eigen::Matrix<long double, Eigen::Dynamic, 1>> v(value.data(), size);
        return v.dot(hermitian * v);
    };
    long double maximum_difference = 0, maximum_generator_error = 0;
    if (context.structure_adjoint_state) {
        ExtendedMatrix moment(na, na), k = ExtendedMatrix::Zero(na, na);
        for (Eigen::Index p = 0; p < static_cast<Eigen::Index>(na); ++p)
            for (Eigen::Index q = 0; q < static_cast<Eigen::Index>(na); ++q) {
                k.setZero();
                k(p, q) = 1;
                const ExtendedMatrix image =
                    extended_generator(view.alpha.strings(), view.beta.strings(), y, k);
                moment(p, q) =
                    2 *
                    residual.dot(Eigen::Map<const Eigen::Matrix<long double, Eigen::Dynamic, 1>>(
                        image.data(), size));
            }
        const ExtendedMatrix difference =
            moment - context.structure_adjoint_state->generator_gradient.cast<long double>();
        std::cout << "PRECISION_INTERNAL_MOMENT,upper_error="
                  << difference.triangularView<Eigen::Upper>().toDenseMatrix().norm()
                  << ",undefined_lower_difference="
                  << difference.triangularView<Eigen::StrictlyLower>().toDenseMatrix().norm()
                  << ",operator_first=" << context.structure_adjoint_state->operator_first
                  << ",source_factors=" << context.structure_adjoint_state->source_factors
                  << std::endl;
    }
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(jets.size()); ++j) {
        const auto k = jets[j].generator.cast<long double>().eval();
        const ExtendedMatrix ci = expand(structure.coefficient_response_coordinates(
                                      basis.col(j).tail(structure.tangent_size()))) /
                                  norm;
        const ExtendedMatrix orbital =
            extended_generator(view.alpha.strings(), view.beta.strings(), y, k);
        const ExtendedMatrix u = orbital + ci;
        const Eigen::Map<const Eigen::Matrix<long double, Eigen::Dynamic, 1>> uf(u.data(), size);
        const long double reference = 2 * residual.dot(uf);
        const ExtendedMatrix control = extended_generator(
            view.alpha.strings(), view.beta.strings(), state.cast<long double>(), k);
        maximum_generator_error = std::max(maximum_generator_error,
                                           (control - jets[j].orbital.cast<long double>()).norm() /
                                               std::max(1.L, control.norm()));
        maximum_difference =
            std::max(maximum_difference, std::abs(reference - production_gradient[j]));
        std::cout << "PRECISION_GRADIENT,column=" << j << ",reference=" << reference
                  << ",production=" << production_gradient[j]
                  << ",difference=" << reference - production_gradient[j] << std::endl;
        if (j < 3 || j + 3 >= static_cast<Eigen::Index>(jets.size())) {
            for (long double mesh : {.003L, .001L}) {
                const long double step = mesh / std::max({1.L, k.norm(), ci.norm()});
                const ExtendedMatrix plus = ExtendedMatrix::Identity(na, na) + step * k;
                const ExtendedMatrix minus = ExtendedMatrix::Identity(na, na) - step * k;
                const ExtendedMatrix ap = reference_compound(view.alpha.strings(), plus),
                                     bp = reference_compound(view.beta.strings(), plus);
                const ExtendedMatrix am = reference_compound(view.alpha.strings(), minus),
                                     bm = reference_compound(view.beta.strings(), minus);
                const long double finite = (energy_at(ap * (y + step * ci) * bp.transpose()) -
                                            energy_at(am * (y - step * ci) * bm.transpose())) /
                                           (2 * step);
                std::cout << "PRECISION_GRADIENT_FD,column=" << j << ",step=" << step
                          << ",reference=" << reference << ",finite=" << finite
                          << ",error=" << finite - reference << std::endl;
            }
        }
    }
    std::cout << "PRECISION_SUMMARY,hamiltonian_skew=" << skew
              << ",maximum_generator_error=" << maximum_generator_error
              << ",maximum_gradient_difference=" << maximum_difference << std::endl;
}
} // namespace xmvb::test
