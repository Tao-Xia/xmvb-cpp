#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "integrals/active/raw.hpp"
#include "vbscf/structures/evaluation/evaluator.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(const Eigen::MatrixXd& actual,
                   const Eigen::MatrixXd& expected,
                   const char* message) {
    if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
        (actual - expected).norm() > 3e-11 * std::max(1.0, expected.norm())) {
        throw std::runtime_error(message);
    }
}

Eigen::MatrixXd fixture(int rows, int columns, double shift) {
    Eigen::MatrixXd result(rows, columns);
    for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row)
            result(row, column) = std::sin(shift + 0.21 * (row + 1) * (column + 2)) +
                                  0.17 * std::cos(shift + 0.43 * (row + 2) * (column + 1));
    return result;
}

std::vector<double> pack(const Eigen::MatrixXd& matrix) {
    const int pairs = static_cast<int>(matrix.rows());
    std::vector<double> result(pairs * (pairs + 1) / 2);
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q <= p; ++q)
            result[p * (p + 1) / 2 + q] = matrix(p, q);
    return result;
}

Eigen::MatrixXd unpack(const std::vector<double>& packed, int pairs) {
    Eigen::MatrixXd result(pairs, pairs);
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q <= p; ++q)
            result(p, q) = result(q, p) = packed[p * (p + 1) / 2 + q];
    return result;
}

// Independent packed-pair orbital map, including old off-diagonal pairs twice.
Eigen::MatrixXd pair_map(const Eigen::MatrixXd& r) {
    const int n = static_cast<int>(r.rows());
    const int pairs = n * (n + 1) / 2;
    Eigen::MatrixXd result(pairs, pairs);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            for (int a = 0; a < n; ++a)
                for (int b = 0; b <= a; ++b)
                    result(a * (a + 1) / 2 + b, i * (i + 1) / 2 + j) =
                        r(a, i) * r(b, j) + (a == b ? 0.0 : r(b, i) * r(a, j));
    return result;
}

void check_raw_consumers(int n,
                         const xmvb::integrals::ActiveSpaceTwoElectronResult& framed,
                         const xmvb::integrals::ActiveSpaceTwoElectronResult& raw) {
    xmvb::vb::VbScfInput input;
    input.structure_data.n_structures = 1;
    input.structure_data.alpha_strings = {{0, 1}};
    input.structure_data.beta_strings = {{1, 2}};
    input.structure_data.pair_alpha = {0};
    input.structure_data.pair_beta = {0};
    input.structure_data.offsets = {0, 1};
    input.structure_data.terms = {{0, 1.0}};
    input.orbital_preparation_input.n_active_orbitals = n;
    xmvb::vb::PreparedActiveSpaceContext prepared;
    const Eigen::MatrixXd overlap = framed.orbital_transform.transpose() * framed.orbital_transform;
    prepared.orbital_result.active_orbital_overlap_matrix =
        std::vector<double>(overlap.data(), overlap.data() + overlap.size());
    const Eigen::MatrixXd one = fixture(n, n, 0.91);
    const Eigen::MatrixXd raw_one = (one + one.transpose()).eval();
    const Eigen::MatrixXd inverse = framed.orbital_transform.triangularView<Eigen::Upper>().solve(
        Eigen::MatrixXd::Identity(n, n));
    prepared.active_space_one_electron_result.h1e_act = inverse.transpose() * raw_one * inverse;
    prepared.active_space_two_electron_result = framed;
    xmvb::vb::PreparedActiveSpaceContext raw_prepared = prepared;
    raw_prepared.active_space_one_electron_result.h1e_act = raw_one;
    raw_prepared.active_space_two_electron_result = raw;
    const Eigen::MatrixXd reconstructed_raw_one = xmvb::integrals::materialize_raw_one_electron(
        prepared.active_space_one_electron_result.h1e_act, framed.orbital_transform);
    require_close(
        reconstructed_raw_one, raw_one, "framed one-electron input did not recover the raw matrix");

    const xmvb::vb::StructureMatrixEvaluator evaluator;
    const auto framed_matrices = evaluator.evaluate(input, prepared);
    const auto raw_matrices = evaluator.evaluate(input, raw_prepared);
    require(
        framed_matrices.hamiltonian_matrix.size() == 1 &&
            raw_matrices.hamiltonian_matrix.size() == 1 &&
            std::abs(framed_matrices.hamiltonian_matrix[0] - raw_matrices.hamiltonian_matrix[0]) <
                3e-11 &&
            std::abs(framed_matrices.overlap_matrix[0] - raw_matrices.overlap_matrix[0]) < 3e-11,
        "raw structure builder changed the dense H/S result");

    const auto framed_cache = evaluator.build_pair_cache(input, prepared, {});
    const auto raw_cache = evaluator.build_pair_cache(input, raw_prepared, {});
    require(framed_cache.alpha_pair_cache.size() == raw_cache.alpha_pair_cache.size() &&
                framed_cache.beta_pair_cache_ref().size() ==
                    raw_cache.beta_pair_cache_ref().size() &&
                !raw_cache.alpha_pair_cache.empty(),
            "raw pair-cache adapter changed cache shape");
    for (std::size_t pair = 0; pair < raw_cache.alpha_pair_cache.size(); ++pair)
        require(std::abs(framed_cache.alpha_pair_cache[pair].total_hamiltonian -
                         raw_cache.alpha_pair_cache[pair].total_hamiltonian) < 3e-11,
                "raw pair-cache adapter changed alpha contractions");
    for (std::size_t pair = 0; pair < raw_cache.beta_pair_cache_ref().size(); ++pair)
        require(std::abs(framed_cache.beta_pair_cache_ref()[pair].total_hamiltonian -
                         raw_cache.beta_pair_cache_ref()[pair].total_hamiltonian) < 3e-11,
                "raw pair-cache adapter changed beta contractions");

    const auto topology = xmvb::vb::build_same_spin_pair_topology(
        input.structure_data.alpha(), input.structure_data.beta(), n);
    const xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder builder;
    const auto diagonal = [&](const xmvb::integrals::ActiveSpaceTwoElectronResult& integrals,
                              const Eigen::MatrixXd& one_electron) {
        return builder.build_diagonal(input.structure_data.alpha(),
                                      input.structure_data.beta(),
                                      input.structure_data.incidence(),
                                      prepared.orbital_result.active_orbital_overlap_matrix,
                                      one_electron,
                                      n,
                                      integrals,
                                      1,
                                      topology);
    };
    const auto framed_diagonal =
        diagonal(framed, prepared.active_space_one_electron_result.h1e_act);
    const auto raw_diagonal = diagonal(raw, raw_one);
    require_close(framed_diagonal.hamiltonian,
                  raw_diagonal.hamiltonian,
                  "raw structure diagonal adapter changed the Hamiltonian");
    require_close(framed_diagonal.overlap,
                  raw_diagonal.overlap,
                  "raw structure diagonal adapter changed the overlap");
    const auto jacobi = [&](const xmvb::integrals::ActiveSpaceTwoElectronResult& integrals,
                            const Eigen::MatrixXd& one_electron) {
        return builder.build_davidson_jacobi_preconditioner(
            input.structure_data.alpha(),
            input.structure_data.beta(),
            input.structure_data.incidence(),
            prepared.orbital_result.active_orbital_overlap_matrix,
            one_electron,
            n,
            integrals,
            1,
            topology);
    };
    const auto framed_jacobi = jacobi(framed, prepared.active_space_one_electron_result.h1e_act);
    const auto raw_jacobi = jacobi(raw, raw_one);
    require(std::abs(framed_jacobi.hamiltonian[0] - raw_jacobi.hamiltonian[0]) < 3e-11 &&
                std::abs(framed_jacobi.overlap[0] - raw_jacobi.overlap[0]) < 3e-11,
            "raw Davidson Jacobi adapter changed the determinant-diagonal model");
}

void check_size(int n) {
    const int pairs = n * (n + 1) / 2;
    const int n_aux = n + 3;
    Eigen::MatrixXd r = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        r(i, i) = 1.1 + 0.07 * i;
        for (int j = i + 1; j < n; ++j)
            r(i, j) = 0.12 * std::sin(0.37 * (i + 1) * (j + 2));
    }
    const Eigen::MatrixXd inverse =
        r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(n, n));
    const Eigen::MatrixXd t = pair_map(r);
    const Eigen::MatrixXd inverse_t = pair_map(inverse);
    const Eigen::MatrixXd cbar = fixture(n + 2, n, -0.23);

    Eigen::MatrixXd exact_kernel = fixture(pairs, pairs, 0.31);
    exact_kernel = (exact_kernel + exact_kernel.transpose()).eval();
    xmvb::integrals::ActiveSpaceTwoElectronResult exact;
    exact.packed_active_two_electron_integrals = pack(exact_kernel);
    exact.dense_active_coefficients = cbar;
    exact.orbital_transform = r;
    exact.ri_one_index_factors = std::make_shared<const Eigen::MatrixXd>(fixture(n, n, 0.17));
    exact.dense_ao_pair_products = fixture(n + 1, pairs, 0.47);
    const auto raw_exact = xmvb::integrals::materialize_raw_active_integrals(exact);
    const Eigen::MatrixXd raw_kernel =
        unpack(raw_exact.packed_active_two_electron_integrals, pairs);
    require_close(unpack(xmvb::integrals::materialize_raw_pair_kernel(
                             exact.packed_active_two_electron_integrals, r),
                         pairs),
                  raw_kernel,
                  "standalone raw pair-kernel conversion differs");
    require(xmvb::integrals::materialize_raw_pair_kernel(exact.packed_active_two_electron_integrals,
                                                         Eigen::MatrixXd{}) ==
                exact.packed_active_two_electron_integrals,
            "empty raw pair-kernel frame changed tensor");
    require_close(raw_kernel,
                  t.transpose() * exact_kernel * t,
                  "exact raw frame differs from explicit pair transform");
    require_close(inverse_t.transpose() * raw_kernel * inverse_t,
                  exact_kernel,
                  "exact frame roundtrip failed");
    require_close(
        raw_exact.dense_active_coefficients, cbar * r, "raw dense active coefficients are wrong");
    require(raw_exact.orbital_transform.size() == 0 && raw_exact.ri_one_index_factors == nullptr &&
                raw_exact.dense_ao_pair_products.size() == 0,
            "raw exact result retained framed metadata or products");
    require(exact.orbital_transform.size() != 0 && exact.ri_one_index_factors != nullptr &&
                exact.dense_ao_pair_products.size() != 0,
            "raw materialization changed the input");
    if (n == 3)
        check_raw_consumers(n, exact, raw_exact);

    xmvb::integrals::ActiveSpaceTwoElectronResult ri;
    ri.representation = xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    ri.n_auxiliary_functions = n_aux;
    ri.ri_active_pair_factors = fixture(n_aux, pairs, -0.52);
    const Eigen::MatrixXd ri_kernel =
        ri.ri_active_pair_factors.transpose() * ri.ri_active_pair_factors;
    ri.packed_active_two_electron_integrals = pack(ri_kernel);
    ri.dense_active_coefficients = cbar;
    ri.orbital_transform = r;
    ri.ri_one_index_factors = std::make_shared<const Eigen::MatrixXd>(fixture(n, n, -0.41));
    ri.dense_ao_pair_products = fixture(n + 1, pairs, 0.13);
    const auto raw_ri = xmvb::integrals::materialize_raw_active_integrals(ri);
    const Eigen::MatrixXd expected_factors = ri.ri_active_pair_factors * t;
    require_close(raw_ri.ri_active_pair_factors,
                  expected_factors,
                  "RI raw frame differs from explicit pair transform");
    require_close(raw_ri.ri_active_pair_factors * inverse_t,
                  ri.ri_active_pair_factors,
                  "RI factor frame roundtrip failed");
    require_close(unpack(raw_ri.packed_active_two_electron_integrals, pairs),
                  expected_factors.transpose() * expected_factors,
                  "RI raw cached tensor disagrees with raw factors");
    require(raw_ri.representation ==
                    xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity &&
                raw_ri.n_auxiliary_functions == n_aux && raw_ri.orbital_transform.size() == 0 &&
                raw_ri.ri_one_index_factors == nullptr && raw_ri.dense_ao_pair_products.size() == 0,
            "raw RI result retained invalid metadata or products");
    if (n == 3)
        check_raw_consumers(n, ri, raw_ri);
    ri.packed_active_two_electron_integrals.clear();
    bool missing_kernel_rejected = false;
    try {
        (void)xmvb::integrals::materialize_raw_active_integrals(ri);
    } catch (const std::invalid_argument&) {
        missing_kernel_rejected = true;
    }
    require(missing_kernel_rejected, "RI raw conversion accepted a missing four-index kernel");
    const auto copied_raw = xmvb::integrals::materialize_raw_active_integrals(raw_exact);
    require_close(copied_raw.dense_active_coefficients,
                  raw_exact.dense_active_coefficients,
                  "already-raw integral copy changed coefficients");
    require_close(unpack(copied_raw.packed_active_two_electron_integrals, pairs),
                  raw_kernel,
                  "already-raw integral copy changed tensor");
}

} // namespace

int main() {
    for (int n : {1, 2, 3, 4})
        check_size(n);
}
