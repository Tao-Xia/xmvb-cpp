#include "integrals/active/two_electron.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/pair_index.hpp"

namespace xmvb::integrals {
namespace {

const std::vector<double>& require_packed_integrals(const ActiveSpaceTwoElectronView& view,
                                                    int n_active) {
    if (!view.packed_active_two_electron_integrals ||
        view.packed_active_two_electron_integrals->size() !=
            packed_active_two_electron_integral_count(n_active))
        throw std::invalid_argument("packed active-space two-electron buffer is unavailable");
    return *view.packed_active_two_electron_integrals;
}

void check_pair(int pair, int pairs) {
    if (pair < 0 || pair >= pairs)
        throw std::invalid_argument("packed active-pair index out of range");
}

} // namespace

ActiveSpaceTwoElectronView make_active_space_two_electron_view(const std::vector<double>& packed) {
    return {&packed};
}

ActiveSpaceTwoElectronView
make_active_space_two_electron_view(const ActiveSpaceTwoElectronResult& result) {
    return {&result.packed_active_two_electron_integrals};
}

int packed_active_pair_count(int n_active_orbitals) {
    if (n_active_orbitals <= 0)
        throw std::invalid_argument("n_active_orbitals must be positive");
    return n_active_orbitals * (n_active_orbitals + 1) / 2;
}

std::size_t packed_active_two_electron_integral_count(int n_active_orbitals) {
    const std::size_t pairs = packed_active_pair_count(n_active_orbitals);
    return pairs * (pairs + 1) / 2;
}

int infer_active_orbital_count_from_packed_pair_count(int pairs) {
    if (pairs <= 0)
        throw std::invalid_argument("packed active-pair count is not triangular");
    const int n = static_cast<int>((std::sqrt(1.0 + 8.0 * pairs) - 1.0) / 2.0 + 0.5);
    if (n * (n + 1) / 2 != pairs)
        throw std::invalid_argument("packed active-pair count is not triangular");
    return n;
}

int infer_active_orbital_count_from_packed_integral_count(std::size_t count) {
    if (!count)
        throw std::invalid_argument("packed_integral_count must be positive");
    const std::size_t pairs =
        static_cast<std::size_t>((std::sqrt(1.0 + 8.0 * count) - 1.0) / 2.0 + 0.5);
    if (pairs * (pairs + 1) / 2 != count)
        throw std::invalid_argument("packed active-space two-electron count is not triangular");
    return infer_active_orbital_count_from_packed_pair_count(static_cast<int>(pairs));
}

double lookup_active_space_two_electron_kernel_value(const ActiveSpaceTwoElectronView& view,
                                                     int row,
                                                     int column,
                                                     int n_active) {
    const int pairs = packed_active_pair_count(n_active);
    check_pair(row, pairs);
    check_pair(column, pairs);
    return require_packed_integrals(
        view, n_active)[core::PairIndex::packed_pair_of_pairs_index(row, column)];
}

std::vector<double>
apply_active_space_two_electron_kernel_to_sparse_projection(const ActiveSpaceTwoElectronView& view,
                                                            int n_active,
                                                            const std::vector<int>& indices,
                                                            const std::vector<double>& values) {
    if (indices.size() != values.size())
        throw std::invalid_argument("packed active-pair projection sizes are inconsistent");
    const int pairs = packed_active_pair_count(n_active);
    const auto& packed = require_packed_integrals(view, n_active);
    std::vector<double> result(pairs, 0.0);
    for (std::size_t k = 0; k < indices.size(); ++k) {
        check_pair(indices[k], pairs);
        for (int row = 0; row < pairs; ++row)
            result[row] +=
                packed[core::PairIndex::packed_pair_of_pairs_index(row, indices[k])] * values[k];
    }
    return result;
}

Eigen::VectorXd apply_active_space_two_electron_kernel_to_sparse_projection_subset(
    const ActiveSpaceTwoElectronView& view,
    int n_active,
    const std::vector<int>& indices,
    const std::vector<double>& values,
    const std::vector<int>& targets) {
    if (indices.size() != values.size())
        throw std::invalid_argument("packed active-pair projection sizes are inconsistent");
    const int pairs = packed_active_pair_count(n_active);
    const auto& packed = require_packed_integrals(view, n_active);
    for (const int pair : indices)
        check_pair(pair, pairs);
    Eigen::VectorXd result = Eigen::VectorXd::Zero(targets.size());
    for (std::size_t row = 0; row < targets.size(); ++row) {
        check_pair(targets[row], pairs);
        for (std::size_t k = 0; k < indices.size(); ++k)
            result[row] +=
                packed[core::PairIndex::packed_pair_of_pairs_index(targets[row], indices[k])] *
                values[k];
    }
    return result;
}

Eigen::MatrixXd
apply_active_space_two_electron_kernel_block(const ActiveSpaceTwoElectronResult& integrals,
                                             int n_active,
                                             const Eigen::Ref<const Eigen::MatrixXd>& vectors) {
    const int pairs = packed_active_pair_count(n_active);
    if (vectors.rows() != pairs || vectors.cols() <= 0)
        throw std::invalid_argument("active pair-kernel block has incompatible dimensions");
    const auto& packed =
        require_packed_integrals(make_active_space_two_electron_view(integrals), n_active);
    Eigen::MatrixXd result(pairs, vectors.cols());
    constexpr int row_tile = 64;
    for (int first = 0; first < pairs; first += row_tile) {
        const int count = std::min(row_tile, pairs - first);
        Eigen::MatrixXd kernel(count, pairs);
        for (int column = 0; column < pairs; ++column)
            for (int row = 0; row < count; ++row)
                kernel(row, column) =
                    packed[core::PairIndex::packed_pair_of_pairs_index(first + row, column)];
        result.middleRows(first, count).noalias() = kernel * vectors;
    }
    return result;
}

std::vector<double>
reconstruct_packed_active_two_electron_integrals(const ActiveSpaceTwoElectronView& view,
                                                 int n_active) {
    return require_packed_integrals(view, n_active);
}

} // namespace xmvb::integrals
