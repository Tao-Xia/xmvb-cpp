#include "integrals/response/directional.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "parallel/openmp.hpp"
#include "integrals/response/internal.hpp"
#include "integrals/response/adjoint.hpp"
#include "integrals/response/ao_pair_operator.hpp"
#include "core/pair_index.hpp"

namespace xmvb::integrals {
namespace {

using detail::accumulate_pair_gradient_rows;
using detail::apply_exact_ao_pair_kernel;
using detail::apply_generated_pair_rows;
using detail::build_mixed_ao_pair_to_active_pair_coefficients_from_cache;
using detail::build_mixed_pair_rows;
using detail::resize_for_overwrite;

} // namespace

void compute_exact_packed_active_two_electron_integral_directional_derivative(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const AoIntegralInput& ao_integral_input,
    ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace* workspace,
    std::vector<double>* delta_packed_active_two_electron_integrals,
    bool compute_fixed_adjoint) {
    if (workspace == nullptr) {
        throw std::invalid_argument("exact packed delta GGO workspace must not be null");
    }
    if (delta_packed_active_two_electron_integrals == nullptr) {
        throw std::invalid_argument("exact packed delta GGO output must not be null");
    }

    const int n_bf = ao_integral_input.n_basis_functions;
    const int n_ao = accepted_cache.n_active_orbitals;
    if (n_bf <= 0 || n_ao <= 0) {
        throw std::invalid_argument("exact packed delta GGO cache dimensions must be positive");
    }
    if (accepted_cache.n_basis_functions != n_bf) {
        throw std::invalid_argument("exact packed delta GGO cache basis mismatch");
    }
    if (dense_active_direction.rows() != n_bf || dense_active_direction.cols() != n_ao) {
        throw std::invalid_argument("dense active direction shape mismatch in cached delta GGO");
    }

    if (accepted_cache.accepted_active_coefficients == nullptr ||
        accepted_cache.accepted_active_coefficients->rows() != n_bf ||
        accepted_cache.accepted_active_coefficients->cols() != n_ao) {
        throw std::invalid_argument("cached dense active coefficient size mismatch in delta GGO");
    }
    const std::size_t n_active_pairs = accepted_cache.active_pair_first_indices.size();
    if (accepted_cache.active_pair_second_indices.size() != n_active_pairs) {
        throw std::invalid_argument("exact packed delta GGO cache pair-index mismatch");
    }
    const std::size_t n_bf_pairs = n_bf * (n_bf + 1) / 2;
    if (accepted_cache.active_pair_gradient_matrix.rows() !=
            static_cast<Eigen::Index>(n_active_pairs) ||
        accepted_cache.active_pair_gradient_matrix.cols() !=
            static_cast<Eigen::Index>(n_active_pairs)) {
        throw std::invalid_argument("cached active-pair adjoint size mismatch in delta GGO");
    }
    if (accepted_cache.accepted_pair_products == nullptr) {
        if (compute_fixed_adjoint)
            workspace->dense_active_direction = dense_active_direction;
        workspace->active_pair_contraction.setZero(static_cast<Eigen::Index>(n_active_pairs),
                                                   static_cast<Eigen::Index>(n_active_pairs));
        if (compute_fixed_adjoint)
            workspace->dense_fixed_adjoint_direction.setZero(n_bf, n_ao);
        else
            workspace->dense_fixed_adjoint_direction.resize(0, 0);
        const Eigen::Index tile_rows = n_bf;
        for (Eigen::Index row_begin = 0; row_begin < static_cast<Eigen::Index>(n_bf_pairs);
             row_begin += tile_rows) {
            const Eigen::Index row_count =
                std::min(tile_rows, static_cast<Eigen::Index>(n_bf_pairs) - row_begin);
            build_mixed_pair_rows(*accepted_cache.accepted_active_coefficients,
                                  dense_active_direction,
                                  accepted_cache,
                                  row_begin,
                                  row_count,
                                  &workspace->directional_pair_coefficients);
            apply_generated_pair_rows(
                ao_integral_input,
                *accepted_cache.accepted_active_coefficients,
                compute_fixed_adjoint ? &workspace->dense_active_direction : nullptr,
                accepted_cache,
                row_begin,
                row_count,
                &workspace->accepted_pair_products_tile,
                compute_fixed_adjoint ? &workspace->directional_pair_products : nullptr);
            workspace->active_pair_contraction.noalias() +=
                workspace->directional_pair_coefficients.transpose() *
                workspace->accepted_pair_products_tile;

            if (compute_fixed_adjoint) {
                workspace->pair_gradients_tile.noalias() =
                    workspace->accepted_pair_products_tile *
                    accepted_cache.active_pair_gradient_matrix;
                accumulate_pair_gradient_rows(workspace->pair_gradients_tile,
                                              row_begin,
                                              dense_active_direction,
                                              accepted_cache,
                                              &workspace->dense_fixed_adjoint_direction);

                workspace->pair_gradients_tile.noalias() =
                    workspace->directional_pair_products *
                    accepted_cache.active_pair_gradient_matrix;
                accumulate_pair_gradient_rows(workspace->pair_gradients_tile,
                                              row_begin,
                                              *accepted_cache.accepted_active_coefficients,
                                              accepted_cache,
                                              &workspace->dense_fixed_adjoint_direction);
            }
        }
        workspace->directional_pair_products.resize(0, 0);
    } else {
        workspace->dense_fixed_adjoint_direction.resize(0, 0);
        if (accepted_cache.accepted_pair_products->rows() !=
                static_cast<Eigen::Index>(n_bf_pairs) ||
            accepted_cache.accepted_pair_products->cols() !=
                static_cast<Eigen::Index>(n_active_pairs)) {
            throw std::invalid_argument("cached accepted pair buffers size mismatch in delta GGO");
        }

        if (accepted_cache.exchange_integrals.size() == 0 ||
            accepted_cache.orbital_pair_products.size() == 0) {
            build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
                *accepted_cache.accepted_active_coefficients,
                dense_active_direction,
                accepted_cache,
                &workspace->directional_pair_coefficients);
        }
        if (accepted_cache.exchange_integrals.size() != 0) {
            workspace->directional_pair_products.resize(0, 0);
        } else {
            apply_exact_ao_pair_kernel(ao_integral_input,
                                       workspace->directional_pair_coefficients,
                                       n_bf,
                                       n_active_pairs,
                                       &workspace->directional_pair_products);
        }

        workspace->active_pair_contraction.resize(static_cast<Eigen::Index>(n_active_pairs),
                                                  static_cast<Eigen::Index>(n_active_pairs));
        if (accepted_cache.orbital_pair_products.size() != 0) {
            detail::contract_orbital_pair_direction(
                accepted_cache, dense_active_direction, &workspace->active_pair_contraction);
        } else {
            workspace->active_pair_contraction.noalias() =
                workspace->directional_pair_coefficients.transpose() *
                *accepted_cache.accepted_pair_products;
        }
    }

    const std::size_t packed_size = n_active_pairs * (n_active_pairs + 1) / 2;
    resize_for_overwrite(delta_packed_active_two_electron_integrals, packed_size);
    for (std::size_t left_active_pair_index = 0; left_active_pair_index < n_active_pairs;
         ++left_active_pair_index) {
        for (std::size_t right_active_pair_index = 0;
             right_active_pair_index <= left_active_pair_index;
             ++right_active_pair_index) {
            const auto packed_index = xmvb::core::PairIndex::packed_pair_of_pairs_index(
                static_cast<int>(left_active_pair_index),
                static_cast<int>(right_active_pair_index));
            (*delta_packed_active_two_electron_integrals)[packed_index] =
                workspace->active_pair_contraction(
                    static_cast<Eigen::Index>(left_active_pair_index),
                    static_cast<Eigen::Index>(right_active_pair_index)) +
                workspace->active_pair_contraction(
                    static_cast<Eigen::Index>(right_active_pair_index),
                    static_cast<Eigen::Index>(left_active_pair_index));
        }
    }
}

Eigen::MatrixXd compute_exact_packed_active_two_electron_integral_directional_derivative_batch(
    const ExactPackedActiveTwoElectronAdjointCache& accepted_cache,
    const std::vector<Eigen::MatrixXd>& dense_active_directions,
    const AoIntegralInput& ao_integral_input,
    std::vector<ExactCtxPairMatrix>* directional_pair_products,
    std::vector<Eigen::MatrixXd>* fixed_adjoint_directions) {
    const int n_bf = accepted_cache.n_basis_functions;
    const int n_ao = accepted_cache.n_active_orbitals;
    const Eigen::Index n_directions = static_cast<Eigen::Index>(dense_active_directions.size());
    const Eigen::Index n_active_pairs =
        static_cast<Eigen::Index>(accepted_cache.active_pair_first_indices.size());
    const Eigen::Index n_bf_pairs = static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2;
    const Eigen::Index packed_size = n_active_pairs * (n_active_pairs + 1) / 2;
    Eigen::MatrixXd packed_directions(packed_size, n_directions);
    if (n_directions == 0) {
        if (directional_pair_products != nullptr) {
            directional_pair_products->clear();
        }
        if (fixed_adjoint_directions != nullptr) {
            fixed_adjoint_directions->clear();
        }
        return packed_directions;
    }
    if (n_bf <= 0 || n_ao <= 0 || n_bf != ao_integral_input.n_basis_functions ||
        n_active_pairs <= 0 || accepted_cache.accepted_active_coefficients == nullptr ||
        accepted_cache.accepted_active_coefficients->rows() != n_bf ||
        accepted_cache.accepted_active_coefficients->cols() != n_ao ||
        accepted_cache.active_pair_second_indices.size() !=
            static_cast<std::size_t>(n_active_pairs) ||
        (accepted_cache.accepted_pair_products != nullptr &&
         (accepted_cache.accepted_pair_products->rows() != n_bf_pairs ||
          accepted_cache.accepted_pair_products->cols() != n_active_pairs))) {
        throw std::invalid_argument(
            "cached exact delta GGO batch has inconsistent accepted dimensions");
    }

    if (directional_pair_products != nullptr) {
        directional_pair_products->resize(static_cast<std::size_t>(n_directions));
    }
    if (fixed_adjoint_directions != nullptr) {
        fixed_adjoint_directions->resize(static_cast<std::size_t>(n_directions));
    }
    if (accepted_cache.active_pair_gradient_matrix.rows() != n_active_pairs ||
        accepted_cache.active_pair_gradient_matrix.cols() != n_active_pairs) {
        throw std::invalid_argument("exact delta GGO batch adjoint shape mismatch");
    }
    for (const auto& direction : dense_active_directions) {
        if (direction.rows() != n_bf || direction.cols() != n_ao) {
            throw std::invalid_argument("exact delta GGO batch direction shape mismatch");
        }
    }
    const auto pack = [&](const Eigen::MatrixXd& contraction, Eigen::Index direction) {
        for (int left = 0; left < n_active_pairs; ++left) {
            for (int right = 0; right <= left; ++right) {
                packed_directions(xmvb::core::PairIndex::packed_pair_of_pairs_index(left, right),
                                  direction) = contraction(left, right) + contraction(right, left);
            }
        }
    };
    constexpr Eigen::Index workspace_bytes = 64 * 1024 * 1024;
    constexpr Eigen::Index value_bytes = sizeof(double);

    if (fixed_adjoint_directions != nullptr && accepted_cache.exchange_integrals.size() != 0 &&
        accepted_cache.orbital_pair_products.size() != 0) {
        Eigen::MatrixXd contraction;
        ExactPackedActiveTwoElectronApplyWorkspace workspace;
        for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
            detail::contract_orbital_pair_direction(
                accepted_cache, dense_active_directions[direction], &contraction);
            pack(contraction, direction);
            apply_exact_packed_active_two_electron_adjoint_hessian_vector(
                accepted_cache,
                dense_active_directions[direction],
                ao_integral_input,
                &workspace,
                &(*fixed_adjoint_directions)[direction]);
            if (directional_pair_products != nullptr)
                (*directional_pair_products)[direction].resize(0, 0);
        }
        return packed_directions;
    }

    if (accepted_cache.accepted_pair_products == nullptr) {
        if (directional_pair_products == nullptr && fixed_adjoint_directions == nullptr) {
            // Integral-only coupling needs sym(delta D^T K D), not K delta D.
            // Generate each accepted row tile once for every direction in the block.
            std::vector<Eigen::MatrixXd> contractions(
                n_directions, Eigen::MatrixXd::Zero(n_active_pairs, n_active_pairs));
            ExactCtxPairMatrix accepted_tile, mixed_tile;
            const Eigen::Index tile_rows = std::max<Eigen::Index>(
                1,
                std::min<Eigen::Index>(n_bf, workspace_bytes / value_bytes / n_active_pairs / 2));
            for (Eigen::Index begin = 0; begin < n_bf_pairs; begin += tile_rows) {
                const Eigen::Index rows = std::min(tile_rows, n_bf_pairs - begin);
                apply_generated_pair_rows(ao_integral_input,
                                          *accepted_cache.accepted_active_coefficients,
                                          nullptr,
                                          accepted_cache,
                                          begin,
                                          rows,
                                          &accepted_tile,
                                          nullptr);
                for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
                    build_mixed_pair_rows(*accepted_cache.accepted_active_coefficients,
                                          dense_active_directions[direction],
                                          accepted_cache,
                                          begin,
                                          rows,
                                          &mixed_tile);
                    contractions[direction].noalias() += mixed_tile.transpose() * accepted_tile;
                }
            }
            for (Eigen::Index direction = 0; direction < n_directions; ++direction)
                pack(contractions[direction], direction);
        } else {
            // Streamed orbital adjoints consume and release their direction tiles.
            ExactPackedActiveTwoElectronDirectionalDerivativeWorkspace workspace;
            std::vector<double> packed_direction;
            for (Eigen::Index direction = 0; direction < n_directions; ++direction) {
                compute_exact_packed_active_two_electron_integral_directional_derivative(
                    accepted_cache,
                    dense_active_directions[direction],
                    ao_integral_input,
                    &workspace,
                    &packed_direction,
                    fixed_adjoint_directions != nullptr);
                packed_directions.col(direction) = Eigen::Map<const Eigen::VectorXd>(
                    packed_direction.data(), packed_direction.size());
                if (directional_pair_products != nullptr)
                    (*directional_pair_products)[direction].resize(0, 0);
                if (fixed_adjoint_directions != nullptr)
                    (*fixed_adjoint_directions)[direction] =
                        workspace.dense_fixed_adjoint_direction;
            }
        }
        return packed_directions;
    }

    // Bound additional block workspace, including compact-stream private images.
    // A single direction remains the irreducible resident-path working set.
    const Eigen::Index image_count = ao_integral_input.pair_graph.has_compact_integral_stream()
                                         ? 3 + effective_openmp_thread_count()
                                         : 3;
    const Eigen::Index block_width =
        directional_pair_products == nullptr
            ? 1
            : std::max<Eigen::Index>(1,
                                     std::min(n_directions,
                                              workspace_bytes / value_bytes / n_bf_pairs /
                                                  n_active_pairs / image_count));
    ExactCtxPairMatrix mixed, block, products;
    Eigen::MatrixXd contraction;
    for (Eigen::Index begin = 0; begin < n_directions; begin += block_width) {
        const Eigen::Index width = std::min(block_width, n_directions - begin);
        if (directional_pair_products != nullptr && width > 1)
            block.resize(n_bf_pairs, width * n_active_pairs);
        for (Eigen::Index offset = 0; offset < width; ++offset) {
            const Eigen::Index direction = begin + offset;
            if (directional_pair_products != nullptr ||
                accepted_cache.orbital_pair_products.size() == 0) {
                build_mixed_ao_pair_to_active_pair_coefficients_from_cache(
                    *accepted_cache.accepted_active_coefficients,
                    dense_active_directions[direction],
                    accepted_cache,
                    &mixed);
            }
            if (accepted_cache.orbital_pair_products.size() != 0) {
                detail::contract_orbital_pair_direction(
                    accepted_cache, dense_active_directions[direction], &contraction);
            } else {
                contraction.noalias() = mixed.transpose() * *accepted_cache.accepted_pair_products;
            }
            pack(contraction, direction);
            if (directional_pair_products != nullptr && width > 1)
                block.middleCols(offset * n_active_pairs, n_active_pairs) = mixed;
            if (fixed_adjoint_directions != nullptr)
                (*fixed_adjoint_directions)[direction].resize(0, 0);
        }
        if (directional_pair_products != nullptr) {
            apply_exact_ao_pair_kernel(ao_integral_input,
                                       width == 1 ? mixed : block,
                                       n_bf,
                                       width * n_active_pairs,
                                       &products);
            if (width == 1) {
                (*directional_pair_products)[begin] = std::move(products);
            } else {
                for (Eigen::Index offset = 0; offset < width; ++offset)
                    (*directional_pair_products)[begin + offset] =
                        products.middleCols(offset * n_active_pairs, n_active_pairs);
            }
        }
    }
    return packed_directions;
}

} // namespace xmvb::integrals
