#include "integrals/active/ri_transform.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "parallel/openmp.hpp"

namespace xmvb::integrals {
namespace {

enum class TransformKind { Forward, RawDirection, OrthogonalDirection };

// Each column is one second orbital index. Rows are (first orbital, aux),
// with the auxiliary index contiguous within each orbital block.
void expand_symmetric_tile(const Eigen::Ref<const Eigen::MatrixXd>& factors,
                           int first_auxiliary,
                           int count,
                           int n,
                           Eigen::MatrixXd* expanded) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const int high = std::max(i, j);
            const int low = std::min(i, j);
            const Eigen::Index pair = static_cast<Eigen::Index>(high) * (high + 1) / 2 + low;
            expanded->col(j).segment(static_cast<Eigen::Index>(i) * count, count) =
                factors.col(pair).segment(first_auxiliary, count);
        }
    }
}

// Change (i, aux; j) to (j, aux; i) between the two wide products.
void transpose_tile_pairs(const Eigen::MatrixXd& source,
                          int count,
                          int n,
                          Eigen::MatrixXd* destination) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            destination->col(i).segment(static_cast<Eigen::Index>(j) * count, count) =
                source.col(j).segment(static_cast<Eigen::Index>(i) * count, count);
        }
    }
}

// Partition tall output rows. Each Eigen product runs inside an OpenMP worker,
// where its own nested Eigen parallelism is inactive.
void multiply_tall(const Eigen::MatrixXd& source,
                   const Eigen::Ref<const Eigen::MatrixXd>& right,
                   int count,
                   int n,
                   Eigen::MatrixXd* destination) {
    const Eigen::Index rows = static_cast<Eigen::Index>(count) * n;
    const int workers =
        coarse_openmp_thread_count(effective_openmp_thread_count(), static_cast<std::size_t>(rows));
    if (workers == 1) {
        destination->topRows(rows).noalias() = source.topRows(rows) * right;
        return;
    }
#pragma omp parallel for num_threads(workers) schedule(static)
    for (int worker = 0; worker < workers; ++worker) {
        const Eigen::Index begin = rows * worker / workers;
        const Eigen::Index end = rows * (worker + 1) / workers;
        destination->middleRows(begin, end - begin).noalias() =
            source.middleRows(begin, end - begin) * right;
    }
}

void write_packed_tile(const Eigen::MatrixXd& transformed,
                       int first_auxiliary,
                       int count,
                       int n,
                       Eigen::MatrixXd* result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            const Eigen::Index pair = static_cast<Eigen::Index>(i) * (i + 1) / 2 + j;
            result->col(pair).segment(first_auxiliary, count) =
                transformed.col(j).segment(static_cast<Eigen::Index>(i) * count, count);
        }
    }
}

void add_symmetric_product_tile(const Eigen::MatrixXd& product,
                                int first_auxiliary,
                                int count,
                                int n,
                                double sign,
                                Eigen::MatrixXd* result) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            const Eigen::Index pair = static_cast<Eigen::Index>(i) * (i + 1) / 2 + j;
            result->col(pair).segment(first_auxiliary, count) +=
                sign * (product.col(i).segment(static_cast<Eigen::Index>(j) * count, count) +
                        product.col(j).segment(static_cast<Eigen::Index>(i) * count, count));
        }
    }
}

Eigen::MatrixXd transform_impl(const Eigen::Ref<const Eigen::MatrixXd>& first_factors,
                               const Eigen::Ref<const Eigen::MatrixXd>* second_factors,
                               const Eigen::Ref<const Eigen::MatrixXd>& frame,
                               const Eigen::Ref<const Eigen::MatrixXd>* direction,
                               TransformKind kind,
                               std::size_t workspace_bytes) {
    const Eigen::Index n_index = frame.rows();
    const Eigen::Index n_auxiliary = first_factors.rows();
    if (n_index <= 0 || n_index > std::numeric_limits<int>::max() || n_auxiliary <= 0 ||
        n_auxiliary > std::numeric_limits<int>::max() || frame.cols() != n_index ||
        first_factors.cols() != n_index * (n_index + 1) / 2) {
        throw std::invalid_argument("invalid RI pair-factor transform dimensions");
    }
    const bool directional = kind != TransformKind::Forward;
    if (directional &&
        (second_factors == nullptr || direction == nullptr ||
         second_factors->rows() != n_auxiliary || second_factors->cols() != first_factors.cols() ||
         direction->rows() != n_index || direction->cols() != n_index)) {
        throw std::invalid_argument("invalid RI pair-factor direction dimensions");
    }

    const int n = static_cast<int>(n_index);
    const std::size_t budget = workspace_bytes / sizeof(double);
    const std::size_t dimension = static_cast<std::size_t>(n);
    const std::size_t buffers = kind == TransformKind::RawDirection ? 3 : 2;
    if (dimension > budget / buffers / dimension) {
        throw std::length_error("RI pair-factor numerical workspace is too small");
    }
    const std::size_t elements_per_auxiliary = buffers * dimension * dimension;
    const int tile_width = static_cast<int>(
        std::min(static_cast<std::size_t>(n_auxiliary), budget / elements_per_auxiliary));

    Eigen::MatrixXd result(n_auxiliary, first_factors.cols());
    const Eigen::Index rows = static_cast<Eigen::Index>(tile_width) * n;
    Eigen::MatrixXd expanded(rows, n);
    Eigen::MatrixXd transformed(rows, n);
    Eigen::MatrixXd factor_stage;
    if (kind == TransformKind::RawDirection)
        factor_stage.resize(rows, n);

    for (int first_auxiliary = 0; first_auxiliary < n_auxiliary; first_auxiliary += tile_width) {
        const int count = std::min(tile_width, static_cast<int>(n_auxiliary) - first_auxiliary);
        if (kind == TransformKind::OrthogonalDirection) {
            expand_symmetric_tile(*second_factors, first_auxiliary, count, n, &expanded);
        } else {
            expand_symmetric_tile(first_factors, first_auxiliary, count, n, &expanded);
        }
        multiply_tall(expanded, frame, count, n, &transformed);

        if (kind == TransformKind::RawDirection) {
            // Save B Q, then transform dB. The saved stage supplies the symmetric
            // Q^T B dQ + dQ^T B Q cross term without a separate B dQ product.
            factor_stage.topRows(static_cast<Eigen::Index>(count) * n) =
                transformed.topRows(static_cast<Eigen::Index>(count) * n);
            expand_symmetric_tile(*second_factors, first_auxiliary, count, n, &expanded);
            multiply_tall(expanded, frame, count, n, &transformed);
            transpose_tile_pairs(transformed, count, n, &expanded);
            multiply_tall(expanded, frame, count, n, &transformed);
            write_packed_tile(transformed, first_auxiliary, count, n, &result);

            transpose_tile_pairs(factor_stage, count, n, &expanded);
            multiply_tall(expanded, *direction, count, n, &transformed);
            add_symmetric_product_tile(transformed, first_auxiliary, count, n, 1.0, &result);
        } else {
            transpose_tile_pairs(transformed, count, n, &expanded);
            multiply_tall(expanded, frame, count, n, &transformed);
            write_packed_tile(transformed, first_auxiliary, count, n, &result);

            if (kind == TransformKind::OrthogonalDirection) {
                expand_symmetric_tile(first_factors, first_auxiliary, count, n, &expanded);
                multiply_tall(expanded, *direction, count, n, &transformed);
                add_symmetric_product_tile(transformed, first_auxiliary, count, n, -1.0, &result);
            }
        }
    }
    return result;
}

} // namespace

Eigen::MatrixXd transform_ri_pair_factors(const Eigen::Ref<const Eigen::MatrixXd>& factors,
                                          const Eigen::Ref<const Eigen::MatrixXd>& frame,
                                          std::size_t workspace_bytes) {
    return transform_impl(
        factors, nullptr, frame, nullptr, TransformKind::Forward, workspace_bytes);
}

Eigen::MatrixXd
transform_ri_pair_factor_direction(const Eigen::Ref<const Eigen::MatrixXd>& factors,
                                   const Eigen::Ref<const Eigen::MatrixXd>& factor_direction,
                                   const Eigen::Ref<const Eigen::MatrixXd>& frame,
                                   const Eigen::Ref<const Eigen::MatrixXd>& frame_direction,
                                   std::size_t workspace_bytes) {
    return transform_impl(factors,
                          &factor_direction,
                          frame,
                          &frame_direction,
                          TransformKind::RawDirection,
                          workspace_bytes);
}

Eigen::MatrixXd transform_ri_pair_factor_direction_from_orthogonal(
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& raw_factor_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& frame,
    const Eigen::Ref<const Eigen::MatrixXd>& frame_connection,
    std::size_t workspace_bytes) {
    return transform_impl(orthogonal_factors,
                          &raw_factor_direction,
                          frame,
                          &frame_connection,
                          TransformKind::OrthogonalDirection,
                          workspace_bytes);
}

} // namespace xmvb::integrals
