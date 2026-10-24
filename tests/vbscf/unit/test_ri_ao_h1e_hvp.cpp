#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "parallel/openmp.hpp"
#include "vbscf/integrals/ao/one_electron/backpropagator.hpp"
#include "vbscf/integrals/ao/one_electron/builder.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/ao/ri_operator.hpp"
#include "integrals/ao/graph.hpp"

namespace {

using Matrix = Eigen::MatrixXd;
using xmvb::integrals::RiAoFactorization;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_close(const Eigen::Ref<const Matrix>& actual,
                   const Eigen::Ref<const Matrix>& expected,
                   double tolerance,
                   const std::string& message) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(),
            message + ": shape mismatch");
    const double error = (actual - expected).cwiseAbs().maxCoeff();
    require(error <= tolerance, message + ": max error = " + std::to_string(error));
}

xmvb::integrals::RiAoFactorization make_ri_factorization() {
    xmvb::integrals::RiAoFactorization ri;
    ri.n_basis_functions = 3;
    ri.n_auxiliary_functions = 2;
    ri.n_packed_ao_pairs = 6;
    ri.metric_whitened_ao_pair_factors.resize(2, 6);
    ri.metric_whitened_ao_pair_factors << 0.7, -0.2, 0.5, 0.1, 0.3, -0.4, -0.1, 0.6, 0.2, -0.5, 0.4,
        0.8;
    return ri;
}

Matrix unpack_factor(const xmvb::integrals::RiAoFactorization& ri, int auxiliary_index) {
    Matrix factor = Matrix::Zero(ri.n_basis_functions, ri.n_basis_functions);
    int packed_index = 0;
    for (int column = 0; column < ri.n_basis_functions; ++column) {
        for (int row = 0; row <= column; ++row) {
            const double value =
                ri.metric_whitened_ao_pair_factors(auxiliary_index, packed_index++);
            factor(row, column) = value;
            factor(column, row) = value;
        }
    }
    return factor;
}

Matrix reference_ri_action(const Eigen::Ref<const Matrix>& input,
                           const xmvb::integrals::RiAoFactorization& ri) {
    Matrix result = Matrix::Zero(input.rows(), input.cols());
    for (int auxiliary = 0; auxiliary < ri.n_auxiliary_functions; ++auxiliary) {
        const Matrix factor = unpack_factor(ri, auxiliary);
        const double projection = (factor.array() * input.array()).sum();
        result.noalias() += 2.0 * projection * factor;
        result.noalias() -= factor * input * factor;
    }
    return result;
}

Matrix map_square(const std::vector<double>& storage, int n) {
    require(storage.size() == static_cast<std::size_t>(n * n), "square storage size mismatch");
    return Eigen::Map<const Matrix>(storage.data(), n, n);
}

void check_raw_and_fused_operator() {
    const xmvb::integrals::RiAoFactorization ri = make_ri_factorization();
    Matrix source(3, 3);
    source << 0.9, -0.3, 0.2, -0.3, -0.4, 0.7, 0.2, 0.7, 0.5;
    Matrix adjoint(3, 3);
    adjoint << -0.2, 0.1, 0.6, 0.1, 0.8, -0.5, 0.6, -0.5, 0.3;

    const Matrix reference_source = reference_ri_action(source, ri);
    const Matrix reference_adjoint = reference_ri_action(adjoint, ri);
    const std::vector<double> source_storage(source.data(), source.data() + source.size());
    const Matrix raw =
        map_square(xmvb::integrals::apply_ao_effective_one_electron_ri_operator(
                       source_storage, ri, 3, {.attempt_spectral_factorization = false}),
                   3);
    require_close(
        raw, reference_source, 2.0e-13, "raw RI operator is not the complete physical matrix");

    xmvb::integrals::AoEffectiveOneElectronRiFusedWorkspace workspace;
    Matrix forward;
    Matrix transpose;
    xmvb::integrals::apply_ao_effective_one_electron_ri_operator_fused(
        source, adjoint, ri, &workspace, &forward, &transpose);
    require_close(forward, reference_source, 2.0e-13, "fused RI forward action mismatch");
    require_close(transpose, reference_adjoint, 2.0e-13, "fused RI transpose action mismatch");
    require_close(
        forward, forward.transpose(), 2.0e-13, "fused RI forward result is not symmetric");

    const double adjoint_error = std::abs((adjoint.array() * forward.array()).sum() -
                                          (transpose.array() * source.array()).sum());
    require(adjoint_error <= 5.0e-13, "RI operator is not self-adjoint on symmetric AO matrices");

    const Matrix second_source = -0.4 * source + 0.2 * adjoint;
    const Matrix second_adjoint = source - 0.7 * adjoint;
    xmvb::integrals::apply_ao_effective_one_electron_ri_operator_fused(
        second_source, second_adjoint, ri, &workspace, &forward, &transpose);
    require_close(forward,
                  reference_ri_action(second_source, ri),
                  2.0e-13,
                  "reused fused RI workspace retained forward data");
    require_close(transpose,
                  reference_ri_action(second_adjoint, ri),
                  2.0e-13,
                  "reused fused RI workspace retained adjoint data");
}

void check_ri_builder_and_directional_action() {
    const xmvb::integrals::RiAoFactorization ri = make_ri_factorization();
    Matrix density(3, 3);
    density << 0.8, 0.2, -0.1, 0.2, 0.4, 0.3, -0.1, 0.3, 0.6;
    Matrix direction(3, 3);
    direction << -0.3, 0.5, 0.2, 0.5, 0.1, -0.4, 0.2, -0.4, 0.7;
    Matrix core(3, 3);
    core << -1.1, 0.2, 0.0, 0.2, -0.7, -0.1, 0.0, -0.1, -0.4;

    xmvb::vb::AoEffectiveOneElectronBuilder builder;
    const auto result = builder.build(density, core, ri, 3);
    const Matrix reference = reference_ri_action(density, ri);
    require_close(result.ao_coulomb_exchange_matrix,
                  reference,
                  2.0e-13,
                  "RI builder doubled the complete raw RI matrix");
    require_close(
        result.ao_effective_h1e, core + reference, 2.0e-13, "RI builder effective matrix mismatch");

    constexpr double step = 1.0e-6;
    const Matrix plus = builder.build(density + step * direction, core, ri, 3).ao_effective_h1e;
    const Matrix minus = builder.build(density - step * direction, core, ri, 3).ao_effective_h1e;
    require_close((plus - minus) / (2.0 * step),
                  reference_ri_action(direction, ri),
                  2.0e-9,
                  "RI builder directional action mismatch");
}

/** @brief Checks signed exchange panel ownership against explicit L X L. */
void check_signed_factor_panels(int n_bf = 13,
                                int n_aux = 7,
                                int n_positive = 3,
                                int n_negative = 2) {
    xmvb::integrals::RiAoFactorization ri;
    ri.n_basis_functions = n_bf;
    ri.n_auxiliary_functions = n_aux;
    ri.n_packed_ao_pairs = n_bf * (n_bf + 1) / 2;
    ri.metric_whitened_ao_pair_factors.resize(n_aux, ri.n_packed_ao_pairs);
    for (Eigen::Index i = 0; i < ri.metric_whitened_ao_pair_factors.size(); ++i) {
        ri.metric_whitened_ao_pair_factors.data()[i] = 0.1 * std::sin(0.137 * (i + 1));
    }
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
#endif
    for (const int requested : {1, 4, 32}) {
#ifdef _OPENMP
        omp_set_num_threads(requested);
#endif
        for (const auto counts : {std::pair{n_positive, 0},
                                  std::pair{0, n_negative},
                                  std::pair{n_positive, n_negative},
                                  std::pair{0, 0}}) {
            xmvb::integrals::AoEffectiveOneElectronRiLowRankFactors factors;
            factors.n_positive_components = counts.first;
            factors.n_negative_components = counts.second;
            factors.scaled_factor_matrix.resize(n_bf, counts.first + counts.second);
            for (Eigen::Index i = 0; i < factors.scaled_factor_matrix.size(); ++i) {
                factors.scaled_factor_matrix.data()[i] = 0.2 * std::cos(0.173 * (i + 2));
            }
            Matrix density = Matrix::Zero(n_bf, n_bf);
            for (int k = 0; k < factors.scaled_factor_matrix.cols(); ++k) {
                const double sign = k < counts.first ? 1.0 : -1.0;
                density.noalias() += sign * factors.scaled_factor_matrix.col(k) *
                                     factors.scaled_factor_matrix.col(k).transpose();
            }
            const Matrix reference = reference_ri_action(density, ri);
            require_close(map_square(xmvb::integrals::apply_ao_effective_one_electron_ri_operator(
                                         factors, ri, n_bf),
                                     n_bf),
                          reference,
                          2.0e-12 * std::max(1.0, reference.cwiseAbs().maxCoeff()),
                          "signed RI panels changed a physical matrix element");
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

/** @brief Exercises the aggregate dense worker budget with analytic factors. */
void check_dense_workspace_budget() {
    constexpr int n_bf = 256;
    constexpr int n_aux = 32;
    xmvb::integrals::RiAoFactorization ri;
    ri.n_basis_functions = n_bf;
    ri.n_auxiliary_functions = n_aux;
    ri.n_packed_ao_pairs = n_bf * (n_bf + 1) / 2;
    ri.metric_whitened_ao_pair_factors = Matrix::Zero(n_aux, ri.n_packed_ao_pairs);
    Eigen::VectorXd diagonal(n_bf);
    for (int mu = 0; mu < n_bf; ++mu)
        diagonal[mu] = std::cos(0.1 * (mu + 1));
    double auxiliary_weight = 0.0;
    for (int a = 0; a < n_aux; ++a) {
        const double weight = std::sin(0.2 * (a + 1));
        auxiliary_weight += weight * weight;
        for (int mu = 0; mu < n_bf; ++mu) {
            ri.metric_whitened_ao_pair_factors(a, mu * (mu + 1) / 2 + mu) = weight * diagonal[mu];
        }
    }
    const Matrix source = Matrix::Identity(n_bf, n_bf);
    const Matrix adjoint = -0.5 * source;
    Matrix reference = Matrix::Zero(n_bf, n_bf);
    reference.diagonal() =
        auxiliary_weight *
        (2.0 * diagonal.sum() * diagonal.array() - diagonal.array().square()).matrix();
    xmvb::integrals::AoEffectiveOneElectronRiFusedWorkspace workspace;
    Matrix forward, transpose;
    xmvb::integrals::apply_ao_effective_one_electron_ri_operator_fused(
        source, adjoint, ri, &workspace, &forward, &transpose);
    require_close(forward,
                  reference,
                  2.0e-12 * reference.cwiseAbs().maxCoeff(),
                  "bounded dense RI action changed the analytic operator");
    require_close(transpose,
                  -0.5 * reference,
                  2.0e-12 * reference.cwiseAbs().maxCoeff(),
                  "bounded dense RI transpose changed the analytic operator");
    const std::size_t worker_bytes =
        workspace.partial_forward.size() * 5ULL * n_bf * n_bf * sizeof(double);
    require(worker_bytes <= 64ULL * 1024 * 1024,
            "dense RI workers exceed their aggregate workspace budget");
    std::cout << "dense_ri_workspace_workers = " << workspace.partial_forward.size()
              << "\ndense_ri_workspace_bytes = " << worker_bytes << '\n';
}

void check_ri_backpropagation_adjoint_identity() {
    const xmvb::integrals::RiAoFactorization ri = make_ri_factorization();
    Matrix direction(3, 3);
    direction << 0.4, -0.2, 0.5, -0.2, -0.6, 0.1, 0.5, 0.1, 0.3;
    Matrix gradient(3, 3);
    gradient << 0.7, -0.4, 0.2, 0.6, -0.1, 0.8, -0.3, 0.5, 0.9;

    xmvb::vb::AoEffectiveOneElectronBackpropagator backpropagator;
    const auto backward = backpropagator.backpropagate(gradient, ri, 3);
    const Eigen::Map<const Matrix> density_gradient(
        backward.inactive_density_gradient.data(), 3, 3);
    const double forward_pairing =
        (gradient.array() * reference_ri_action(direction, ri).array()).sum();
    const double reverse_pairing = (density_gradient.array() * direction.array()).sum();
    require(std::abs(forward_pairing - reverse_pairing) <= 5.0e-13,
            "RI backpropagator fails the builder adjoint identity");
}

void check_exact_builder_triangular_regression() {
    xmvb::integrals::AoIntegralInput ao;
    ao.n_basis_functions = 2;
    ao.ao_core_hamiltonian_matrix.resize(2, 2);
    ao.ao_core_hamiltonian_matrix << -1.0, 0.15, 0.15, -0.6;
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(
        {0, 0, 0, 1, 1, 2}, {0, 1, 2, 1, 2, 2}, {0.7, -0.2, 0.1, 0.5, 0.3, 0.9}, 2);
    Matrix density(2, 2);
    density << 0.8, -0.3, -0.3, 0.4;

    const std::vector<double> raw = xmvb::integrals::apply_ao_h1e(density.data(), ao, 1);
    Matrix expected = Eigen::Map<const Matrix>(raw.data(), 2, 2);
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column <= row; ++column) {
            expected(row, column) += expected(column, row);
            expected(column, row) = expected(row, column);
        }
    }

    xmvb::vb::AoEffectiveOneElectronBuilder builder;
    const auto result = builder.build(density, ao);
    require_close(result.ao_coulomb_exchange_matrix,
                  expected,
                  2.0e-13,
                  "exact builder triangular finalization changed");
    require_close(result.ao_effective_h1e,
                  ao.ao_core_hamiltonian_matrix + expected,
                  2.0e-13,
                  "exact builder effective matrix changed");
}

void check_large_work_count_thread_bound() {
    const std::size_t oversized_work_count =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    require(xmvb::bounded_openmp_thread_count(32, oversized_work_count) == 32,
            "large work count overflowed the bounded OpenMP thread count");
    require(xmvb::bounded_openmp_thread_count(32, 7) == 7,
            "small work count did not bound the OpenMP thread count");
    require(xmvb::bounded_openmp_thread_count(32, 0) == 1,
            "empty work count did not select one OpenMP thread");
    require(xmvb::openmp_work_pieces(10000, 16) == 256 && xmvb::openmp_chunk_size(10000, 16) == 40,
            "dynamic chunk granularity is inconsistent");
    require(xmvb::openmp_work_pieces(7, 32) == 7 && xmvb::openmp_chunk_size(0, 32) == 1,
            "empty/small dynamic loops have invalid chunks");
    const auto largest = std::numeric_limits<std::size_t>::max();
    require(xmvb::coarse_openmp_thread_count(32, 0) == 1 &&
                xmvb::coarse_openmp_thread_count(32, 224) == 1 &&
                xmvb::coarse_openmp_thread_count(32, 1024) == 32 &&
                xmvb::coarse_openmp_thread_count(32, largest) == 32,
            "coarse-loop worker bound is inconsistent or overflowed");
    const auto pieces = xmvb::openmp_work_pieces(largest, std::numeric_limits<int>::max());
    const auto chunk = xmvb::openmp_chunk_size(largest, std::numeric_limits<int>::max());
    require(pieces > 0 && chunk > 0 && chunk == largest / pieces + (largest % pieces != 0),
            "dynamic chunk calculation overflowed");
}

/** @brief Checks exact scalar/fused/block actions against a dense stencil. */
void check_exact_parallel_actions(int n_bf = 2) {
    xmvb::integrals::AoIntegralInput ao;
    ao.n_basis_functions = n_bf;
    std::vector<int> left, right;
    std::vector<double> values;
    const int n_pairs = n_bf * (n_bf + 1) / 2;
    for (int a = 0; a < n_pairs; ++a) {
        for (int b = a; b < n_pairs; ++b) {
            // Mixed pair orientations also exercise unordered compact streams.
            left.push_back((a + b) % 2 ? b : a);
            right.push_back((a + b) % 2 ? a : b);
            values.push_back(std::sin(0.7 * (a + 1) * (b + 2)));
        }
    }
    ao.pair_graph = xmvb::integrals::build_ao_pair_graph(left, right, values, n_bf);
    const int size = n_bf * n_bf;
    Matrix stencil = Matrix::Zero(size, size);
    for (std::size_t edge = 0; edge < values.size(); ++edge) {
        const int i = ao.pair_graph.pair_first[left[edge]];
        const int j = ao.pair_graph.pair_second[left[edge]];
        const int k = ao.pair_graph.pair_first[right[edge]];
        const int l = ao.pair_graph.pair_second[right[edge]];
        double v = values[edge];
        if (i == j)
            v *= 0.5;
        if (k == l)
            v *= 0.5;
        if (i == k && j == l)
            v *= 0.5;
        stencil(j * n_bf + i, l * n_bf + k) += 4.0 * v;
        stencil(l * n_bf + k, j * n_bf + i) += 4.0 * v;
        stencil(k * n_bf + i, j * n_bf + l) -= v;
        stencil(l * n_bf + j, i * n_bf + k) -= v;
        stencil(l * n_bf + i, j * n_bf + k) -= v;
        stencil(k * n_bf + j, i * n_bf + l) -= v;
    }
    Matrix sources(size, 3), adjoints(size, 3);
    for (int row = 0; row < size; ++row)
        for (int column = 0; column < 3; ++column)
            sources(row, column) = std::sin(0.3 * (row + 1) * (column + 2));
    adjoints = 0.3 * sources.array().cos().matrix();
    const Matrix expected_forward = stencil * sources;
    const Matrix expected_transpose = stencil.transpose() * adjoints;
    xmvb::integrals::AoH1eSymmetricWorkspace symmetric_workspace;
    xmvb::integrals::AoH1eBatchWorkspace batch_workspace;
    Matrix symmetric_sources(size, 8), symmetric_adjoints(size, 8);
    Matrix expected_block_forward(size, 8), expected_block_transpose(size, 8);
    for (Eigen::Index column = 0; column < 8; ++column) {
        const Eigen::Map<const Matrix> source(sources.col(column % 3).data(), n_bf, n_bf);
        const Eigen::Map<const Matrix> adjoint(adjoints.col(column % 3).data(), n_bf, n_bf);
        Eigen::Map<Matrix>(symmetric_sources.col(column).data(), n_bf, n_bf) =
            (column + 1.0) * (source + source.transpose());
        Eigen::Map<Matrix>(symmetric_adjoints.col(column).data(), n_bf, n_bf) =
            (column + 0.5) * (adjoint + adjoint.transpose());
        const Eigen::VectorXd raw_forward = stencil * symmetric_sources.col(column);
        const Eigen::VectorXd raw_transpose = stencil.transpose() * symmetric_adjoints.col(column);
        const Eigen::Map<const Matrix> forward(raw_forward.data(), n_bf, n_bf);
        const Eigen::Map<const Matrix> transpose(raw_transpose.data(), n_bf, n_bf);
        Eigen::Map<Matrix>(expected_block_forward.col(column).data(), n_bf, n_bf) =
            forward + forward.transpose();
        Eigen::Map<Matrix>(expected_block_transpose.col(column).data(), n_bf, n_bf) =
            0.5 * (transpose + transpose.transpose());
    }
    const auto check = [&]() {
        for (int threads : {1, 2, 4, 32}) {
            for (Eigen::Index column = 0; column < sources.cols(); ++column) {
                const auto forward =
                    xmvb::integrals::apply_ao_h1e(sources.col(column).data(), ao, threads);
                const auto transpose = xmvb::integrals::apply_ao_h1e_transpose(
                    adjoints.col(column).data(), ao, threads);
                require_close(Eigen::Map<const Matrix>(forward.data(), size, 1),
                              expected_forward.col(column),
                              2.0e-13,
                              "parallel exact forward mismatch");
                require_close(Eigen::Map<const Matrix>(transpose.data(), size, 1),
                              expected_transpose.col(column),
                              2.0e-13,
                              "parallel exact transpose mismatch");
                std::vector<double> fused_forward, fused_transpose;
                const Eigen::Map<const Matrix> source_matrix(
                    sources.col(column).data(), n_bf, n_bf);
                const Eigen::Map<const Matrix> adjoint_matrix(
                    adjoints.col(column).data(), n_bf, n_bf);
                const Matrix symmetric_source = source_matrix + source_matrix.transpose();
                const Matrix symmetric_adjoint = adjoint_matrix + adjoint_matrix.transpose();
                const Eigen::Map<const Eigen::VectorXd> source_vector(symmetric_source.data(),
                                                                      size);
                const Eigen::Map<const Eigen::VectorXd> adjoint_vector(symmetric_adjoint.data(),
                                                                       size);
                const Eigen::VectorXd raw_forward = stencil * source_vector;
                const Eigen::VectorXd raw_transpose = stencil.transpose() * adjoint_vector;
                const Eigen::Map<const Matrix> reference_forward(raw_forward.data(), n_bf, n_bf);
                const Eigen::Map<const Matrix> reference_transpose(
                    raw_transpose.data(), n_bf, n_bf);
                xmvb::integrals::apply_symmetric_ao_h1e_fused(symmetric_source.data(),
                                                              symmetric_adjoint.data(),
                                                              ao,
                                                              threads,
                                                              &symmetric_workspace,
                                                              &fused_forward,
                                                              &fused_transpose);
                const Eigen::Map<const Matrix> symmetric_forward(fused_forward.data(), n_bf, n_bf);
                const Eigen::Map<const Matrix> symmetric_transpose(
                    fused_transpose.data(), n_bf, n_bf);
                require_close(symmetric_forward,
                              reference_forward + reference_forward.transpose(),
                              2.0e-13,
                              "two-channel physical forward mismatch");
                require_close(symmetric_transpose + symmetric_transpose.transpose(),
                              reference_transpose + reference_transpose.transpose(),
                              2.0e-13,
                              "two-channel physical transpose mismatch");
            }
            for (Eigen::Index width : {1, 2, 3, 4, 8}) {
                Matrix forward, transpose;
                xmvb::integrals::apply_symmetric_ao_h1e_fused_batch(
                    symmetric_sources.leftCols(width),
                    symmetric_adjoints.leftCols(width),
                    ao,
                    threads,
                    &batch_workspace,
                    &forward,
                    &transpose);
                require_close(forward,
                              expected_block_forward.leftCols(width),
                              2.0e-13,
                              "parallel symmetric block forward mismatch");
                require_close(transpose,
                              expected_block_transpose.leftCols(width),
                              2.0e-13,
                              "parallel symmetric block physical adjoint mismatch");
            }
        }
    };
    check();
    // Exercise the large-ERI representation without allocating a large graph.
    ao.pair_graph.row_offsets.clear();
    ao.pair_graph.columns.clear();
    ao.pair_graph.values.clear();
    ao.pair_graph.integral_edges.clear();
    ao.pair_graph.integral_rows = left;
    ao.pair_graph.integral_columns = right;
    ao.pair_graph.integral_values = values;
    check();
#ifdef _OPENMP
    const int saved_dynamic = omp_get_dynamic();
    omp_set_dynamic(1);
    check();
    omp_set_dynamic(saved_dynamic);
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        check();
    }
#endif
    ao.pair_graph.integral_rows.clear();
    ao.pair_graph.integral_columns.clear();
    ao.pair_graph.integral_values.clear();
    ao.pair_graph.row_offsets.assign(n_pairs + 1, 0);
    std::vector<double> forward, transpose;
    xmvb::integrals::apply_symmetric_ao_h1e_fused(sources.col(0).data(),
                                                  adjoints.col(0).data(),
                                                  ao,
                                                  32,
                                                  &symmetric_workspace,
                                                  &forward,
                                                  &transpose);
    require(Eigen::Map<const Matrix>(forward.data(), size, 1).isZero(0.0) &&
                Eigen::Map<const Matrix>(transpose.data(), size, 1).isZero(0.0),
            "empty exact action retained stale workspace values");
    Matrix block_forward, block_transpose;
    xmvb::integrals::apply_symmetric_ao_h1e_fused_batch(symmetric_sources,
                                                        symmetric_adjoints,
                                                        ao,
                                                        32,
                                                        &batch_workspace,
                                                        &block_forward,
                                                        &block_transpose);
    require(block_forward.isZero(0.0) && block_transpose.isZero(0.0),
            "empty exact block retained stale workspace values");
    xmvb::integrals::apply_symmetric_ao_h1e_fused_batch(symmetric_sources.leftCols(0),
                                                        symmetric_adjoints.leftCols(0),
                                                        ao,
                                                        32,
                                                        &batch_workspace,
                                                        &block_forward,
                                                        &block_transpose);
    require(block_forward.rows() == size && block_forward.cols() == 0 &&
                block_transpose.rows() == size && block_transpose.cols() == 0,
            "zero-column exact block has incorrect shape");
}

} // namespace

int main() {
    try {
        check_raw_and_fused_operator();
        check_signed_factor_panels();
        // At 32 workers the signed full-width transform crosses the 64 MiB
        // auxiliary workspace boundary. Other team widths exercise one tile.
        check_signed_factor_panels(64, 1600, 32, 32);
        check_dense_workspace_budget();
        check_ri_builder_and_directional_action();
        check_ri_backpropagation_adjoint_identity();
        check_exact_builder_triangular_regression();
        check_large_work_count_thread_bound();
        check_exact_parallel_actions();
        check_exact_parallel_actions(5);
        std::cout << "RI AO-H1E fused HVP primitives: passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
