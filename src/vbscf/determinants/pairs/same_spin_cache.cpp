#include "core/index.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "parallel/openmp.hpp"
#include "mcscf/ci/strings.hpp"
#include "vbscf/determinants/pairs/storage.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"
#include "vbscf/determinants/pairs/contractions.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "core/pair_index.hpp"
#include "integrals/active/two_electron.hpp"

namespace xmvb::vb {

const AcceptedPairTileProvider& SameSpinPairCacheContext::alpha_provider() const {
    if (!alpha_pair_provider) {
        throw std::logic_error("alpha accepted-pair provider is not initialized");
    }
    return *alpha_pair_provider;
}

const AcceptedPairTileProvider& SameSpinPairCacheContext::beta_provider() const {
    if (!beta_pair_provider) {
        throw std::logic_error("beta accepted-pair provider is not initialized");
    }
    return *beta_pair_provider;
}

void release_same_spin_pair_evaluations(SameSpinPairCacheContext* cache_context) {
    if (cache_context == nullptr) {
        throw std::invalid_argument("same-spin cache context must not be null");
    }
    std::vector<SpinDeterminantPairEvaluation>().swap(cache_context->alpha_pair_cache);
    std::vector<SpinDeterminantPairEvaluation>().swap(cache_context->beta_pair_cache);
    std::vector<double>().swap(cache_context->opposite_spin_cache);
    std::vector<bool>().swap(cache_context->opposite_spin_cache_computed);
    cache_context->cached_same_spin_evaluation_count = 0;
    cache_context->estimated_cache_bytes = 0;
}

bool has_polynomial_same_spin_response_pairs(const SameSpinPairCacheContext& cache_context) {
    const auto contains_polynomial_pair =
        [](const std::vector<SpinDeterminantPairEvaluation>& pairs) {
            return std::any_of(
                pairs.begin(), pairs.end(), [](const SpinDeterminantPairEvaluation& pair) {
                    return pair.same_spin_polynomial_response != nullptr;
                });
        };
    return contains_polynomial_pair(cache_context.alpha_pair_cache_ref()) ||
           (!cache_context.shares_same_spin_pair_cache_between_spins() &&
            contains_polynomial_pair(cache_context.beta_pair_cache_ref()));
}

namespace {

constexpr double kContributionTolerance = 1.0e-15;

int same_spin_pair_cache_thread_count(mcscf::StringIndex n_unique_determinants) {
    // Same-spin cache construction is row-parallel over unique determinants.
    // Keep tiny determinant spaces serial and cap the team at the number of rows
    // so dynamic scheduling does not launch workers with no cache rows to own.
    if (n_unique_determinants <= 10) {
        return 1;
    }
    return std::max<Eigen::Index>(
        1, std::min<Eigen::Index>(xmvb::effective_openmp_thread_count(), n_unique_determinants));
}

bool can_reuse_close_shell_same_spin_pair_cache(SpinStringRows alpha_det, SpinStringRows beta_det) {
    return alpha_det == beta_det;
}

struct SpinWordHasher {
    std::size_t operator()(const std::vector<std::uint64_t>& words) const noexcept {
        std::size_t hash = words.size();
        for (const auto word : words)
            hash = hash * 1315423911u + word;
        return hash;
    }
};
std::vector<std::uint64_t> spin_key(std::span<const std::uint64_t> words) {
    std::vector<std::uint64_t> key(words.begin(), words.end());
    while (key.size() > 1 && key.back() == 0)
        key.pop_back();
    return key;
}
std::vector<mcscf::StringIndex> map_unique_strings(const mcscf::StringTable& source,
                                                   const SpinDeterminantReuseTable& reference) {
    std::vector<mcscf::StringIndex> remap(source.size());
    if (source == reference.unique_determinants) {
        std::iota(remap.begin(), remap.end(), 0);
        return remap;
    }
    if (reference.basis.size()) {
        try {
            for (std::size_t i = 0; i < source.size(); ++i)
                remap[i] = reference.basis.index(source.mask(i));
        } catch (const std::out_of_range&) {
            return {};
        }
        return remap;
    }
    std::unordered_map<std::vector<std::uint64_t>, mcscf::StringIndex, SpinWordHasher> index;
    index.reserve(reference.unique_determinants.size());
    for (int i = 0; i < static_cast<int>(reference.unique_determinants.size()); ++i)
        index.emplace(spin_key(reference.unique_determinants.words(i)), i);
    for (std::size_t i = 0; i < source.size(); ++i) {
        const auto found = index.find(spin_key(source.words(i)));
        if (found == index.end())
            return {};
        remap[i] = found->second;
    }
    return remap;
}

bool try_build_reuse_table_in_reference_basis(
    SpinStringRows spin_determinants,
    const SpinDeterminantReuseTable& reference_reuse_table,
    SpinDeterminantReuseTable* mapped_reuse_table) {
    if (mapped_reuse_table == nullptr) {
        throw std::invalid_argument("mapped_reuse_table must not be null");
    }

    mapped_reuse_table->unique_determinants = reference_reuse_table.unique_determinants;
    mapped_reuse_table->determinant_to_unique_id.assign(spin_determinants.size(), -1);
    std::vector<unsigned char> touched_unique_ids(reference_reuse_table.unique_determinants.size(),
                                                  0u);

    const auto source = build_spin_determinant_reuse_table(spin_determinants);
    const auto remap = map_unique_strings(source.unique_determinants, reference_reuse_table);
    if (remap.size() != source.unique_determinants.size()) {
        mapped_reuse_table->unique_determinants.clear();
        mapped_reuse_table->determinant_to_unique_id.clear();
        return false;
    }
    for (std::size_t i = 0; i < source.determinant_to_unique_id.size(); ++i) {
        const auto id = remap[source.determinant_to_unique_id[i]];
        mapped_reuse_table->determinant_to_unique_id[i] = id;
        touched_unique_ids[id] = 1u;
    }

    for (const unsigned char touched : touched_unique_ids) {
        if (touched == 0u) {
            mapped_reuse_table->unique_determinants.clear();
            mapped_reuse_table->determinant_to_unique_id.clear();
            return false;
        }
    }
    return true;
}

std::size_t
expected_opposite_spin_scalar_cache_size(const SameSpinPairCacheContext& cache_context) {
    return cache_context.n_unique_alpha * cache_context.n_unique_alpha *
           cache_context.n_unique_beta * cache_context.n_unique_beta;
}

bool has_materialized_opposite_spin_scalar_cache(const SameSpinPairCacheContext& cache_context) {
    const std::size_t expected_size = expected_opposite_spin_scalar_cache_size(cache_context);
    if (expected_size == 0) {
        return false;
    }
    if (cache_context.opposite_spin_cache.empty() &&
        cache_context.opposite_spin_cache_computed.empty()) {
        return false;
    }
    if (cache_context.opposite_spin_cache.size() != expected_size ||
        cache_context.opposite_spin_cache_computed.size() != expected_size) {
        throw std::invalid_argument(
            "opposite-spin scalar cache storage does not match unique alpha/beta dimensions");
    }
    return true;
}

std::vector<double> build_dense_active_pair_kernel(const std::vector<double>& eri_act,
                                                   int n_orbitals) {
    const int n_packed_active_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    std::vector<double> dense_pair_kernel(n_packed_active_pairs * n_packed_active_pairs, 0.0);

    for (int column_pair = 0; column_pair < n_packed_active_pairs; ++column_pair) {
        for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
            const auto packed_pair_of_pairs_index =
                xmvb::core::PairIndex::packed_pair_of_pairs_index(row_pair, column_pair);
            dense_pair_kernel[column_pair * n_packed_active_pairs + row_pair] =
                eri_act[packed_pair_of_pairs_index];
        }
    }

    return dense_pair_kernel;
}

OppositeSpinPackedPairProjection build_sparse_packed_pair_projection(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& coefficient_matrix,
    bool coefficient_matrix_is_right_by_left,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronView& two_electron_view,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_projected_pair_values) {
    const int n_packed_active_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    OppositeSpinPackedPairProjection projection;
    if (occ_L.empty()) {
        return projection;
    }

    std::vector<double> dense_pair_values(n_packed_active_pairs, 0.0);
    std::vector<unsigned char> touched_mask(n_packed_active_pairs, 0);
    std::vector<int> touched_indices;
    touched_indices.reserve(occ_L.size() * occ_R.size());

    for (int left_column = 0; left_column < static_cast<int>(occ_L.size()); ++left_column) {
        const int orbital_index_left = occ_L[left_column];
        for (int right_row = 0; right_row < static_cast<int>(occ_R.size()); ++right_row) {
            const int orbital_index_right = occ_R[right_row];
            const auto packed_pair_index =
                xmvb::core::PairIndex::packed_pair_index(orbital_index_right, orbital_index_left);
            if (touched_mask[packed_pair_index] == 0) {
                touched_mask[packed_pair_index] = 1;
                touched_indices.push_back(packed_pair_index);
            }
            const double coefficient = coefficient_matrix_is_right_by_left
                                           ? coefficient_matrix(right_row, left_column)
                                           : coefficient_matrix(left_column, right_row);
            dense_pair_values[packed_pair_index] += coefficient;
        }
    }

    projection.packed_pair_indices.reserve(touched_indices.size());
    projection.packed_pair_values.reserve(touched_indices.size());
    for (const int packed_pair_index : touched_indices) {
        const double packed_pair_value = dense_pair_values[packed_pair_index];
        if (packed_pair_value == 0.0) {
            continue;
        }
        projection.packed_pair_indices.push_back(packed_pair_index);
        projection.packed_pair_values.push_back(packed_pair_value);
    }

    if (projection.packed_pair_indices.empty()) {
        projection.projected_pair_values.clear();
        return projection;
    }

    if (!materialize_projected_pair_values) {
        projection.projected_pair_values.clear();
        return projection;
    }

    if (dense_pair_kernel != nullptr) {
        projection.projected_pair_values.assign(n_packed_active_pairs, 0.0);
        for (std::size_t entry_index = 0; entry_index < projection.packed_pair_indices.size();
             ++entry_index) {
            const int packed_pair_index = projection.packed_pair_indices[entry_index];
            const double packed_pair_value = projection.packed_pair_values[entry_index];
            const std::size_t kernel_column_offset = packed_pair_index * n_packed_active_pairs;
            for (int row_pair = 0; row_pair < n_packed_active_pairs; ++row_pair) {
                projection.projected_pair_values[row_pair] +=
                    (*dense_pair_kernel)[kernel_column_offset + row_pair] * packed_pair_value;
            }
        }
    } else {
        projection.projected_pair_values =
            xmvb::integrals::apply_active_space_two_electron_kernel_to_sparse_projection(
                two_electron_view,
                n_orbitals,
                projection.packed_pair_indices,
                projection.packed_pair_values);
    }
    return projection;
}

void attach_opposite_spin_pair_cache(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronView& two_electron_view,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_first_order_projected_values,
    SpinDeterminantPairEvaluation* pair_evaluation) {
    if (pair_evaluation == nullptr) {
        throw std::invalid_argument("pair_evaluation must not be null");
    }

    pair_evaluation->opposite_spin_pair_cache.n_packed_active_pairs =
        xmvb::integrals::packed_active_pair_count(n_orbitals);

    const auto& overlap_result = pair_evaluation->overlap_result;
    if (occ_L.empty() || overlap_result.nullity >= 2) {
        return;
    }
    pair_evaluation->opposite_spin_pair_cache.first_order_cofactor_projection =
        build_sparse_packed_pair_projection(occ_L,
                                            occ_R,
                                            calc_cofactor_1st(overlap_result),
                                            true,
                                            n_orbitals,
                                            two_electron_view,
                                            dense_pair_kernel,
                                            materialize_first_order_projected_values);

    if (overlap_result.nullity == 0) {
        // Production overlap-response contractions consume the inverse-overlap
        // projection sparsely and apply the active-space kernel only to the
        // channels touched by the partner image.  Retaining a dense G*x image for
        // every ordered unique-string pair duplicates O(P U^2) storage and
        // O(P n_e^2 U^2) work without a production consumer.  The diagnostic
        // scalar helpers retain their exact on-demand fallback.
        pair_evaluation->opposite_spin_pair_cache.inverse_overlap_projection =
            build_sparse_packed_pair_projection(
                occ_L,
                occ_R,
                build_inverse_overlap_submatrix_from_result(overlap_result),
                false,
                n_orbitals,
                two_electron_view,
                dense_pair_kernel,
                false);
    }
}

Eigen::MatrixXd
build_spin_one_electron_block_matrix(const std::vector<int>& occ_L,
                                     const std::vector<int>& occ_R,
                                     const Eigen::Ref<const Eigen::MatrixXd>& h1e_act) {
    const int n_electrons = static_cast<int>(occ_L.size());
    Eigen::MatrixXd one_electron_block(n_electrons, n_electrons);
    for (int left_column = 0; left_column < n_electrons; ++left_column) {
        const int orbital_index_left = occ_L[left_column];
        for (int right_row = 0; right_row < n_electrons; ++right_row) {
            const int orbital_index_right = occ_R[right_row];
            one_electron_block(right_row, left_column) =
                h1e_act(orbital_index_right, orbital_index_left);
        }
    }
    return one_electron_block;
}

Eigen::MatrixXd build_polynomial_same_spin_hamiltonian_overlap_gradient(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    SpinDeterminantPairEvaluation* pair_evaluation) {
    if (pair_evaluation == nullptr)
        throw std::invalid_argument("same-spin polynomial cache output must not be null");
    const CofactorDifferential& cofactor = cached_cofactor_differential(*pair_evaluation);
    auto payload = std::make_shared<SameSpinPolynomialResponsePayload>();
    payload->one_electron_block = build_spin_one_electron_block_matrix(occ_L, occ_R, h1e_act);
    payload->antisymmetrized_interaction = build_spin_antisymmetrized_interaction_matrix(
        occ_L,
        occ_R,
        n_active_orbitals,
        xmvb::integrals::make_active_space_two_electron_view(active_space_two_electron_result));
    const Eigen::MatrixXd gradient =
        cofactor.first(payload->one_electron_block) +
        cofactor.second_contraction_gradient(payload->antisymmetrized_interaction);
    pair_evaluation->same_spin_polynomial_response = std::move(payload);
    return gradient;
}

void populate_same_spin_phi_cache_entry(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    SpinDeterminantPairEvaluation* pair_evaluation) {
    if (pair_evaluation == nullptr) {
        throw std::invalid_argument("same-spin pair evaluation must not be null");
    }
    if (!pair_evaluation->cofactor_differential) {
        pair_evaluation->cofactor_differential =
            std::make_shared<const CofactorDifferential>(pair_evaluation->overlap_result);
    }
    pair_evaluation->has_same_spin_phi_cache = false;
    pair_evaluation->same_spin_one_electron_phi = 0.0;
    pair_evaluation->same_spin_total_phi = 0.0;
    pair_evaluation->same_spin_inverse_overlap_gradient.resize(0, 0);
    pair_evaluation->same_spin_overlap_hamiltonian_gradient.resize(0, 0);

    const CofactorDifferential& cofactor = cached_cofactor_differential(*pair_evaluation);
    if (cofactor.uses_regular_form()) {
        Eigen::MatrixXd inverse_overlap_gradient;
        const SameSpinPhiResult phi_result =
            compute_same_spin_original_phi(occ_L,
                                           occ_R,
                                           h1e_act,
                                           n_orbitals,
                                           active_space_two_electron_result,
                                           pair_evaluation->overlap_result,
                                           &inverse_overlap_gradient);
        pair_evaluation->has_same_spin_phi_cache = true;
        pair_evaluation->same_spin_one_electron_phi = phi_result.one_electron_phi;
        pair_evaluation->same_spin_total_phi = phi_result.total_phi;
        pair_evaluation->same_spin_inverse_overlap_gradient = std::move(inverse_overlap_gradient);
    }

    const bool regular_ri_pair =
        pair_evaluation->has_same_spin_phi_cache &&
        active_space_two_electron_result.representation ==
            xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    if (regular_ri_pair) {
        pair_evaluation->same_spin_overlap_hamiltonian_gradient =
            build_regular_same_spin_overlap_hamiltonian_gradient(
                pair_evaluation->overlap_result,
                pair_evaluation->same_spin_total_phi,
                pair_evaluation->same_spin_inverse_overlap_gradient);
        pair_evaluation->same_spin_polynomial_response.reset();
        return;
    }
    pair_evaluation->same_spin_overlap_hamiltonian_gradient =
        build_polynomial_same_spin_hamiltonian_overlap_gradient(
            occ_L, occ_R, h1e_act, n_orbitals, active_space_two_electron_result, pair_evaluation);
}

std::vector<SpinDeterminantPairEvaluation>
build_same_spin_pair_cache(const xmvb::mcscf::StringTable& unique_spin_determinants,
                           const DeterminantPairEvaluator& pair_evaluator,
                           const std::vector<double>& ovlp_act,
                           const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
                           int n_orbitals,
                           const std::vector<double>& eri_act,
                           const std::vector<double>* dense_pair_kernel,
                           bool materialize_projected_pair_values,
                           bool retain_derivative_payload) {
    const mcscf::StringIndex n_unique_determinants =
        core::checked_index<mcscf::StringIndex>(unique_spin_determinants.size());
    std::vector<SpinDeterminantPairEvaluation> pair_cache(n_unique_determinants *
                                                          n_unique_determinants);

    // Each cached entry is one exact same-spin determinant kernel for a specific
    // left/right orientation. The scalar same-spin matrix elements are symmetric,
    // but the overlap/cofactor payload is directional because rows belong to the
    // right determinant and columns to the left determinant.
    const xmvb::integrals::ActiveSpaceTwoElectronView two_electron_view =
        xmvb::integrals::make_active_space_two_electron_view(eri_act);
    const int n_threads = same_spin_pair_cache_thread_count(n_unique_determinants);

#pragma omp parallel for schedule(dynamic, 1) if (n_threads > 1) num_threads(n_threads)
    for (mcscf::StringIndex left_index = 0; left_index < n_unique_determinants; ++left_index) {
        for (mcscf::StringIndex right_index = 0; right_index < n_unique_determinants;
             ++right_index) {
            pair_cache[ordered_spin_pair_storage_index(
                left_index, right_index, n_unique_determinants)] =
                pair_evaluator.evaluate_same_spin_pair(unique_spin_determinants[left_index],
                                                       unique_spin_determinants[right_index],
                                                       ovlp_act,
                                                       h1e_act,
                                                       n_orbitals,
                                                       eri_act,
                                                       retain_derivative_payload);
            attach_opposite_spin_pair_cache(unique_spin_determinants[left_index],
                                            unique_spin_determinants[right_index],
                                            n_orbitals,
                                            two_electron_view,
                                            dense_pair_kernel,
                                            materialize_projected_pair_values,
                                            &pair_cache[ordered_spin_pair_storage_index(
                                                left_index, right_index, n_unique_determinants)]);
        }
    }

    return pair_cache;
}

std::vector<SpinDeterminantPairEvaluation> build_same_spin_pair_cache(
    const xmvb::mcscf::StringTable& unique_spin_determinants,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    const std::vector<double>* dense_pair_kernel,
    bool materialize_projected_pair_values,
    bool retain_derivative_payload) {
    const mcscf::StringIndex n_unique_determinants =
        core::checked_index<mcscf::StringIndex>(unique_spin_determinants.size());
    std::vector<SpinDeterminantPairEvaluation> pair_cache(n_unique_determinants *
                                                          n_unique_determinants);
    const xmvb::integrals::ActiveSpaceTwoElectronView two_electron_view =
        xmvb::integrals::make_active_space_two_electron_view(active_space_two_electron_result);
    const int n_threads = same_spin_pair_cache_thread_count(n_unique_determinants);

#pragma omp parallel for schedule(dynamic, 1) if (n_threads > 1) num_threads(n_threads)
    for (mcscf::StringIndex left_index = 0; left_index < n_unique_determinants; ++left_index) {
        for (mcscf::StringIndex right_index = 0; right_index < n_unique_determinants;
             ++right_index) {
            pair_cache[ordered_spin_pair_storage_index(
                left_index, right_index, n_unique_determinants)] =
                pair_evaluator.evaluate_same_spin_pair(unique_spin_determinants[left_index],
                                                       unique_spin_determinants[right_index],
                                                       ovlp_act,
                                                       h1e_act,
                                                       n_orbitals,
                                                       active_space_two_electron_result,
                                                       retain_derivative_payload);
            attach_opposite_spin_pair_cache(unique_spin_determinants[left_index],
                                            unique_spin_determinants[right_index],
                                            n_orbitals,
                                            two_electron_view,
                                            dense_pair_kernel,
                                            materialize_projected_pair_values,
                                            &pair_cache[ordered_spin_pair_storage_index(
                                                left_index, right_index, n_unique_determinants)]);
        }
    }

    return pair_cache;
}

void populate_same_spin_phi_cache_entries(
    const xmvb::mcscf::StringTable& unique_spin_determinants,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    std::vector<SpinDeterminantPairEvaluation>* pair_cache) {
    if (pair_cache == nullptr) {
        throw std::invalid_argument("pair_cache must not be null");
    }

    const mcscf::StringIndex n_unique_determinants =
        core::checked_index<mcscf::StringIndex>(unique_spin_determinants.size());
    if (pair_cache->size() != n_unique_determinants * n_unique_determinants) {
        throw std::invalid_argument(
            "same-spin phi cache population size does not match unique-spin dimensions");
    }
    const int n_threads = same_spin_pair_cache_thread_count(n_unique_determinants);

#pragma omp parallel for schedule(dynamic, 1) if (n_threads > 1) num_threads(n_threads)
    for (mcscf::StringIndex left_index = 0; left_index < n_unique_determinants; ++left_index) {
        for (mcscf::StringIndex right_index = 0; right_index < n_unique_determinants;
             ++right_index) {
            auto& pair_evaluation = (*pair_cache)[ordered_spin_pair_storage_index(
                left_index, right_index, n_unique_determinants)];
            populate_same_spin_phi_cache_entry(unique_spin_determinants[left_index],
                                               unique_spin_determinants[right_index],
                                               h1e_act,
                                               n_orbitals,
                                               active_space_two_electron_result,
                                               &pair_evaluation);
        }
    }
}

struct PairProjectionPolicy {
    bool alpha = false;
    bool beta = false;

    bool any() const noexcept {
        return alpha || beta;
    }
};

PairProjectionPolicy resolve_pair_projection_policy(PairProjectionCache requested_cache,
                                                    const SameSpinPairCacheContext& cache_context) {
    switch (requested_cache) {
        case PairProjectionCache::Both:
            return {true, true};
        case PairProjectionCache::SmallerSpin:
            if (cache_context.beta_reuses_alpha_pair_cache) {
                return {true, false};
            }
            if (cache_context.alpha_reuse_table.unique_determinants.size() <=
                cache_context.beta_reuse_table.unique_determinants.size()) {
                return {true, false};
            }
            return {false, true};
    }
    throw std::logic_error("unrecognized pair projection cache policy");
}

} // namespace

void complete_same_spin_pair_evaluation(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_active_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    bool populate_opposite_spin_projection,
    bool materialize_projected_pair_values,
    bool populate_response_payload,
    SpinDeterminantPairEvaluation* pair_evaluation) {
    if (pair_evaluation == nullptr) {
        throw std::invalid_argument("same-spin pair evaluation must not be null");
    }
    if (populate_opposite_spin_projection) {
        attach_opposite_spin_pair_cache(
            occ_L,
            occ_R,
            n_active_orbitals,
            xmvb::integrals::make_active_space_two_electron_view(active_space_two_electron_result),
            nullptr,
            materialize_projected_pair_values,
            pair_evaluation);
    }
    if (populate_response_payload) {
        populate_same_spin_phi_cache_entry(occ_L,
                                           occ_R,
                                           h1e_act,
                                           n_active_orbitals,
                                           active_space_two_electron_result,
                                           pair_evaluation);
    }
}

SpinDeterminantReuseTable build_spin_determinant_reuse_table(SpinStringRows spin_determinants) {
    SpinDeterminantReuseTable reuse_table;
    if (!spin_determinants.indices().empty()) {
        reuse_table.unique_determinants = spin_determinants.strings();
        reuse_table.determinant_to_unique_id.assign(spin_determinants.indices().begin(),
                                                    spin_determinants.indices().end());
        return reuse_table;
    }
    reuse_table.determinant_to_unique_id.resize(spin_determinants.size(), -1);
    if (spin_determinants.strings().word_width() == 1) {
        std::unordered_map<std::uint64_t, mcscf::StringIndex> ids;
        ids.reserve(spin_determinants.size());
        for (std::size_t i = 0; i < spin_determinants.size(); ++i) {
            const auto mask = spin_determinants.mask(i);
            const auto [found, inserted] =
                ids.emplace(mask, reuse_table.unique_determinants.size());
            if (inserted)
                reuse_table.unique_determinants.push_mask(mask);
            reuse_table.determinant_to_unique_id[i] = found->second;
        }
    } else {
        std::unordered_map<std::vector<std::uint64_t>, mcscf::StringIndex, SpinWordHasher> ids;
        ids.reserve(spin_determinants.size());
        for (std::size_t i = 0; i < spin_determinants.size(); ++i) {
            auto key = spin_key(spin_determinants.words(i));
            const auto [found, inserted] = ids.emplace(key, reuse_table.unique_determinants.size());
            if (inserted)
                reuse_table.unique_determinants.push_words(key);
            reuse_table.determinant_to_unique_id[i] = found->second;
        }
    }

    return reuse_table;
}

std::size_t ordered_spin_pair_storage_index(mcscf::StringIndex left_index,
                                            mcscf::StringIndex right_index,
                                            mcscf::StringIndex n_unique_determinants) {
    return core::checked_add(
        core::checked_multiply(core::checked_index<std::size_t>(left_index),
                               core::checked_index<std::size_t>(n_unique_determinants)),
        core::checked_index<std::size_t>(right_index));
}

std::size_t
estimate_same_spin_pair_cache_bytes(const xmvb::mcscf::StringTable& unique_spin_determinants,
                                    int n_orbitals) {
    int max_electron_count = 0;
    for (std::size_t i = 0; i < unique_spin_determinants.size(); ++i)
        max_electron_count =
            std::max<Eigen::Index>(max_electron_count, unique_spin_determinants.electron_count(i));
    const std::size_t n_packed_active_pairs =
        n_orbitals == 0 ? 0 : xmvb::integrals::packed_active_pair_count(n_orbitals);
    const std::size_t max_occupied_pair_count =
        static_cast<std::size_t>(max_electron_count) *
        static_cast<std::size_t>(std::max<Eigen::Index>(0, max_electron_count - 1)) / 2;

    // A cached same-spin pair keeps one `SpinDeterminantPairEvaluation`, plus
    // the dominant dynamic overlap payload:
    // - regular pairs: inverse overlap and first deleted-minor matrices
    // - singular pairs: singular values, full U/V factors, and first cofactors
    // - opposite-spin projected caches: sparse packed-pair regroupings plus the
    //   dense `G u` / `G x` images in active-pair space
    // - gradient reuse: cached same-spin `\partial \phi / \partial X`
    // The bound intentionally overestimates so the cache heuristic prefers the
    // direct path rather than risking a large allocation.
    const std::size_t pair_count =
        unique_spin_determinants.size() * unique_spin_determinants.size();
    const std::size_t per_pair_double_count =
        4ull * max_electron_count * max_electron_count + max_electron_count +
        2ull * n_packed_active_pairs + max_electron_count * max_electron_count +
        4ull * max_electron_count * max_electron_count +
        4ull * max_occupied_pair_count * max_occupied_pair_count;
    const std::size_t per_pair_int_count = 4ull * max_electron_count * max_electron_count;
    return pair_count * (sizeof(SpinDeterminantPairEvaluation) +
                         per_pair_double_count * sizeof(double) + per_pair_int_count * sizeof(int));
}

namespace {

SameSpinPairCacheContext finalize_spin_topology(SameSpinPairCacheContext context, int n_orbitals) {
    if (n_orbitals > 0 && n_orbitals <= 63) {
        auto& alpha = context.alpha_reuse_table;
        auto& beta = context.beta_reuse_table;
        if (!alpha.basis.size())
            alpha.basis = alpha.unique_determinants.basis(n_orbitals);
        if (!beta.basis.size())
            beta.basis = beta.unique_determinants == alpha.unique_determinants
                             ? alpha.basis
                             : beta.unique_determinants.basis(n_orbitals);
        alpha.unique_determinants = mcscf::StringTable::from_basis(alpha.basis);
        beta.unique_determinants = mcscf::StringTable::from_basis(beta.basis);
    }
    context.alpha_pair_provider = std::make_shared<const AcceptedPairTileProvider>(
        context.alpha_reuse_table.unique_determinants, n_orbitals);
    context.beta_pair_provider =
        context.beta_reuses_alpha_pair_cache
            ? context.alpha_pair_provider
            : std::make_shared<const AcceptedPairTileProvider>(
                  context.beta_reuse_table.unique_determinants, n_orbitals);

    const std::size_t n_alpha = context.alpha_reuse_table.unique_determinants.size();
    const std::size_t n_beta = context.beta_reuse_table.unique_determinants.size();
    context.cached_same_spin_evaluation_count = n_alpha * n_alpha;
    context.estimated_cache_bytes = estimate_same_spin_pair_cache_bytes(
        context.alpha_reuse_table.unique_determinants, n_orbitals);
    if (!context.beta_reuses_alpha_pair_cache) {
        context.cached_same_spin_evaluation_count += n_beta * n_beta;
        context.estimated_cache_bytes += estimate_same_spin_pair_cache_bytes(
            context.beta_reuse_table.unique_determinants, n_orbitals);
    }
    context.n_unique_alpha = static_cast<int>(n_alpha);
    context.n_unique_beta = static_cast<int>(n_beta);
    return context;
}

SpinDeterminantReuseTable complete_spin_reuse_table(SpinStringRows determinants, int n_orbitals) {
    if (determinants.empty())
        throw std::invalid_argument("orthogonal spin topology requires determinants");
    const int n_electrons = determinants.electron_count();
    auto table = build_spin_determinant_reuse_table(determinants);
    if (!n_orbitals) {
        if (n_electrons)
            throw std::invalid_argument("nonempty occupation in a zero-orbital sector");
        return table;
    }
    table.basis = table.unique_determinants.basis(n_orbitals);
    if (table.basis.complete()) {
        table.unique_determinants = mcscf::StringTable::from_basis(table.basis);
        return table;
    }
    const mcscf::SpinStrings complete(n_orbitals, n_electrons);
    std::vector<mcscf::StringIndex> remap(table.unique_determinants.size());
    for (std::size_t i = 0; i < remap.size(); ++i)
        remap[i] = complete.index(table.unique_determinants.mask(i));
    for (auto& id : table.determinant_to_unique_id)
        id = remap[id];
    table.basis = complete;
    table.unique_determinants = mcscf::StringTable::from_basis(complete);
    return table;
}

SpinDeterminantReuseTable
map_spin_reuse_table_to_complete_basis(SpinStringRows determinants,
                                       const SpinDeterminantReuseTable& reference) {
    SpinDeterminantReuseTable table;
    table.unique_determinants = reference.unique_determinants;
    table.basis = reference.basis;
    const auto source = build_spin_determinant_reuse_table(determinants);
    const auto remap = map_unique_strings(source.unique_determinants, reference);
    if (remap.size() != source.unique_determinants.size())
        throw std::invalid_argument("spin string is outside complete basis");
    table.determinant_to_unique_id = source.determinant_to_unique_id;
    for (auto& id : table.determinant_to_unique_id)
        id = remap[id];
    return table;
}

} // namespace

SameSpinPairCacheContext
build_same_spin_pair_topology(SpinStringRows alpha_det, SpinStringRows beta_det, int n_orbitals) {
    if (alpha_det.size() != beta_det.size()) {
        throw std::invalid_argument("alpha_det and beta_det must have the same size");
    }

    SameSpinPairCacheContext context;
    context.alpha_reuse_table = build_spin_determinant_reuse_table(alpha_det);
    context.close_shell_diagonal_reuses_same_spin_pair_cache =
        can_reuse_close_shell_same_spin_pair_cache(alpha_det, beta_det);
    context.beta_reuses_alpha_pair_cache =
        context.close_shell_diagonal_reuses_same_spin_pair_cache ||
        try_build_reuse_table_in_reference_basis(
            beta_det, context.alpha_reuse_table, &context.beta_reuse_table);
    if (context.close_shell_diagonal_reuses_same_spin_pair_cache) {
        context.beta_reuse_table = context.alpha_reuse_table;
    } else if (!context.beta_reuses_alpha_pair_cache) {
        context.beta_reuse_table = build_spin_determinant_reuse_table(beta_det);
    }
    return finalize_spin_topology(std::move(context), n_orbitals);
}

SameSpinPairCacheContext
build_orthogonal_spin_topology(SpinStringRows alpha_det, SpinStringRows beta_det, int n_orbitals) {
    if (alpha_det.size() != beta_det.size())
        throw std::invalid_argument("alpha_det and beta_det must have the same size");
    if (n_orbitals < 0 || n_orbitals > 63)
        throw std::invalid_argument("invalid orthogonal spin orbital count");

    SameSpinPairCacheContext context;
    context.alpha_reuse_table = complete_spin_reuse_table(alpha_det, n_orbitals);
    context.close_shell_diagonal_reuses_same_spin_pair_cache =
        can_reuse_close_shell_same_spin_pair_cache(alpha_det, beta_det);
    context.beta_reuses_alpha_pair_cache = alpha_det.electron_count() == beta_det.electron_count();
    context.beta_reuse_table =
        context.beta_reuses_alpha_pair_cache
            ? map_spin_reuse_table_to_complete_basis(beta_det, context.alpha_reuse_table)
            : complete_spin_reuse_table(beta_det, n_orbitals);
    return finalize_spin_topology(std::move(context), n_orbitals);
}

SameSpinPairCacheContext
build_same_spin_pair_cache_context(SpinStringRows alpha_det,
                                   SpinStringRows beta_det,
                                   const DeterminantPairEvaluator& pair_evaluator,
                                   const std::vector<double>& ovlp_act,
                                   const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
                                   int n_orbitals,
                                   const std::vector<double>& eri_act) {
    return build_same_spin_pair_cache_context(alpha_det,
                                              beta_det,
                                              pair_evaluator,
                                              ovlp_act,
                                              h1e_act,
                                              n_orbitals,
                                              eri_act,
                                              SameSpinPairCacheBuildOptions{});
}

SameSpinPairCacheContext
build_same_spin_pair_cache_context(SpinStringRows alpha_det,
                                   SpinStringRows beta_det,
                                   const DeterminantPairEvaluator& pair_evaluator,
                                   const std::vector<double>& ovlp_act,
                                   const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
                                   int n_orbitals,
                                   const std::vector<double>& eri_act,
                                   SameSpinPairCacheBuildOptions build_options) {
    SameSpinPairCacheContext cache_context =
        build_same_spin_pair_topology(alpha_det, beta_det, n_orbitals);
    // Production policy: always materialize the ordered same-spin determinant
    // kernels. Callers may still opt out of the dense opposite-spin projected
    // images when a streamed matrix-form path can rebuild only the touched
    // packed-pair rows on demand.
    cache_context.use_same_spin_pair_cache = !alpha_det.empty();

    const PairProjectionPolicy projection_policy =
        resolve_pair_projection_policy(build_options.pair_projection_cache, cache_context);
    std::vector<double> dense_pair_kernel;
    const std::vector<double>* dense_pair_kernel_ptr = nullptr;
    if (projection_policy.any()) {
        dense_pair_kernel = build_dense_active_pair_kernel(eri_act, n_orbitals);
        dense_pair_kernel_ptr = &dense_pair_kernel;
    }
    cache_context.alpha_pair_cache =
        build_same_spin_pair_cache(cache_context.alpha_reuse_table.unique_determinants,
                                   pair_evaluator,
                                   ovlp_act,
                                   h1e_act,
                                   n_orbitals,
                                   eri_act,
                                   dense_pair_kernel_ptr,
                                   projection_policy.alpha,
                                   build_options.retain_derivative_payload);
    if (!cache_context.beta_reuses_alpha_pair_cache) {
        cache_context.beta_pair_cache =
            build_same_spin_pair_cache(cache_context.beta_reuse_table.unique_determinants,
                                       pair_evaluator,
                                       ovlp_act,
                                       h1e_act,
                                       n_orbitals,
                                       eri_act,
                                       dense_pair_kernel_ptr,
                                       projection_policy.beta,
                                       build_options.retain_derivative_payload);
    } else {
        cache_context.beta_pair_cache.clear();
    }

    return cache_context;
}

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    SpinStringRows alpha_det,
    SpinStringRows beta_det,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
    return build_same_spin_pair_cache_context(alpha_det,
                                              beta_det,
                                              pair_evaluator,
                                              ovlp_act,
                                              h1e_act,
                                              n_orbitals,
                                              active_space_two_electron_result,
                                              SameSpinPairCacheBuildOptions{});
}

SameSpinPairCacheContext build_same_spin_pair_cache_context(
    SpinStringRows alpha_det,
    SpinStringRows beta_det,
    const DeterminantPairEvaluator& pair_evaluator,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    SameSpinPairCacheBuildOptions build_options) {
    SameSpinPairCacheContext cache_context =
        build_same_spin_pair_topology(alpha_det, beta_det, n_orbitals);
    // Production policy: always materialize the ordered same-spin determinant
    // kernels. Callers may still opt out of the dense opposite-spin projected
    // images when a streamed matrix-form path can rebuild only the touched
    // packed-pair rows on demand.
    cache_context.use_same_spin_pair_cache = !alpha_det.empty();
    const PairProjectionPolicy projection_policy =
        resolve_pair_projection_policy(build_options.pair_projection_cache, cache_context);

    const std::size_t expected_packed_size =
        xmvb::integrals::packed_active_two_electron_integral_count(n_orbitals);

    if (!active_space_two_electron_result.packed_active_two_electron_integrals.empty() &&
        active_space_two_electron_result.packed_active_two_electron_integrals.size() ==
            expected_packed_size) {
        // Materialized packed integrals available — use the packed-exact cache
        // builder which constructs a cache-friendly dense_pair_kernel for the
        // opposite-spin projection and avoids view-dispatch indirection.
        const auto& eri_act = active_space_two_electron_result.packed_active_two_electron_integrals;
        std::vector<double> dense_pair_kernel;
        const std::vector<double>* dense_pair_kernel_ptr = nullptr;
        if (projection_policy.any()) {
            dense_pair_kernel = build_dense_active_pair_kernel(eri_act, n_orbitals);
            dense_pair_kernel_ptr = &dense_pair_kernel;
        }

        cache_context.alpha_pair_cache =
            build_same_spin_pair_cache(cache_context.alpha_reuse_table.unique_determinants,
                                       pair_evaluator,
                                       ovlp_act,
                                       h1e_act,
                                       n_orbitals,
                                       eri_act,
                                       dense_pair_kernel_ptr,
                                       projection_policy.alpha,
                                       build_options.retain_derivative_payload);

        if (!cache_context.beta_reuses_alpha_pair_cache) {
            cache_context.beta_pair_cache =
                build_same_spin_pair_cache(cache_context.beta_reuse_table.unique_determinants,
                                           pair_evaluator,
                                           ovlp_act,
                                           h1e_act,
                                           n_orbitals,
                                           eri_act,
                                           dense_pair_kernel_ptr,
                                           projection_policy.beta,
                                           build_options.retain_derivative_payload);
        } else {
            cache_context.beta_pair_cache.clear();
        }
    } else {
        // Direct RI representation — no materialized packed integrals.
        // Streamed callers can skip the dense `G u` / `G x` images entirely and
        // keep only the sparse packed-pair coefficients in the cache.
        cache_context.alpha_pair_cache =
            build_same_spin_pair_cache(cache_context.alpha_reuse_table.unique_determinants,
                                       pair_evaluator,
                                       ovlp_act,
                                       h1e_act,
                                       n_orbitals,
                                       active_space_two_electron_result,
                                       nullptr,
                                       projection_policy.alpha,
                                       build_options.retain_derivative_payload);

        if (!cache_context.beta_reuses_alpha_pair_cache) {
            cache_context.beta_pair_cache =
                build_same_spin_pair_cache(cache_context.beta_reuse_table.unique_determinants,
                                           pair_evaluator,
                                           ovlp_act,
                                           h1e_act,
                                           n_orbitals,
                                           active_space_two_electron_result,
                                           nullptr,
                                           projection_policy.beta,
                                           build_options.retain_derivative_payload);
        } else {
            cache_context.beta_pair_cache.clear();
        }
    }

    return cache_context;
}

void populate_same_spin_phi_cache(
    SameSpinPairCacheContext* same_spin_pair_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result) {
    if (same_spin_pair_cache == nullptr) {
        throw std::invalid_argument("same_spin_pair_cache must not be null");
    }
    if (!same_spin_pair_cache->enabled()) {
        return;
    }

    // The matrix-form same-spin backward reuses the accepted-point same-spin
    // pair kernels many times. Regular pairs cache `phi` and
    // `\partial \phi / \partial X^{-1}`, while singular pairs cache the exact
    // occupied-block overlap gradient of the same-spin Hamiltonian directly.
    // Both cases keep the later matrix-form adjoint sweep on the ordered
    // unique-spin grid and avoid repeated determinant-level integral lookup.
    populate_same_spin_phi_cache_entries(
        same_spin_pair_cache->alpha_reuse_table.unique_determinants,
        h1e_act,
        n_orbitals,
        active_space_two_electron_result,
        same_spin_pair_cache->mutable_alpha_pair_cache_ref());
    if (!same_spin_pair_cache->shares_same_spin_pair_cache_between_spins()) {
        populate_same_spin_phi_cache_entries(
            same_spin_pair_cache->beta_reuse_table.unique_determinants,
            h1e_act,
            n_orbitals,
            active_space_two_electron_result,
            same_spin_pair_cache->mutable_beta_pair_cache_ref());
    }
}

DeterminantPairEvaluation evaluate_full_determinant_pair_with_optional_same_spin_cache(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const DeterminantPairEvaluator& pair_evaluator,
    SpinStringRows alpha_det,
    SpinStringRows beta_det,
    Eigen::Index determinant_index_left,
    Eigen::Index determinant_index_right,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const std::vector<double>& eri_act,
    bool retain_spin_pair_evaluations) {

    if (same_spin_pair_cache != nullptr && same_spin_pair_cache->enabled()) {
        const auto& alpha_occ_L = alpha_det[determinant_index_left];
        const auto& alpha_occ_R = alpha_det[determinant_index_right];
        const auto& beta_occ_L = beta_det[determinant_index_left];
        const auto& beta_occ_R = beta_det[determinant_index_right];
        const int alpha_left_id = same_spin_pair_cache->alpha_reuse_table
                                      .determinant_to_unique_id[determinant_index_left];
        const int alpha_right_id = same_spin_pair_cache->alpha_reuse_table
                                       .determinant_to_unique_id[determinant_index_right];
        const int beta_left_id =
            same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id[determinant_index_left];
        const int beta_right_id = same_spin_pair_cache->beta_reuse_table
                                      .determinant_to_unique_id[determinant_index_right];
        const auto& alpha_pair_result =
            same_spin_pair_cache->alpha_pair_cache[ordered_spin_pair_storage_index(
                alpha_left_id,
                alpha_right_id,
                static_cast<int>(
                    same_spin_pair_cache->alpha_reuse_table.unique_determinants.size()))];
        const auto& beta_pair_result =
            same_spin_pair_cache->beta_pair_cache_ref()[ordered_spin_pair_storage_index(
                beta_left_id,
                beta_right_id,
                static_cast<int>(
                    same_spin_pair_cache->beta_reuse_table.unique_determinants.size()))];

        double opposite_spin_coupling = 0.0;
        if (has_materialized_opposite_spin_scalar_cache(*same_spin_pair_cache)) {
            const std::size_t opp_cache_idx = same_spin_pair_cache->opposite_spin_cache_index(
                alpha_left_id, alpha_right_id, beta_left_id, beta_right_id);
            if (!same_spin_pair_cache->opposite_spin_cache_computed[opp_cache_idx]) {
                const double opp_coupling =
                    evaluate_opposite_spin_coulomb_coupling(alpha_occ_L,
                                                            alpha_occ_R,
                                                            alpha_pair_result,
                                                            beta_occ_L,
                                                            beta_occ_R,
                                                            beta_pair_result,
                                                            eri_act);
                // Note: This is not thread-safe, but redundant writes are acceptable.
                const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)
                    ->opposite_spin_cache[opp_cache_idx] = opp_coupling;
                const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)
                    ->opposite_spin_cache_computed[opp_cache_idx] = true;
            }
            opposite_spin_coupling = same_spin_pair_cache->opposite_spin_cache[opp_cache_idx];
        } else {
            opposite_spin_coupling = evaluate_opposite_spin_coulomb_coupling(alpha_occ_L,
                                                                             alpha_occ_R,
                                                                             alpha_pair_result,
                                                                             beta_occ_L,
                                                                             beta_occ_R,
                                                                             beta_pair_result,
                                                                             eri_act);
        }

        // Combine results using cached opposite-spin coupling
        DeterminantPairEvaluation result;
        if (retain_spin_pair_evaluations) {
            result.alpha = alpha_pair_result;
            result.beta = beta_pair_result;
        }
        result.overlap_determinant = alpha_pair_result.overlap_result.overlap_determinant *
                                     beta_pair_result.overlap_result.overlap_determinant;
        result.one_electron_hamiltonian = alpha_pair_result.one_electron_hamiltonian *
                                              beta_pair_result.overlap_result.overlap_determinant +
                                          beta_pair_result.one_electron_hamiltonian *
                                              alpha_pair_result.overlap_result.overlap_determinant;
        result.total_hamiltonian = alpha_pair_result.total_hamiltonian *
                                       beta_pair_result.overlap_result.overlap_determinant +
                                   beta_pair_result.total_hamiltonian *
                                       alpha_pair_result.overlap_result.overlap_determinant +
                                   opposite_spin_coupling;

        return result;
    }

    return pair_evaluator.evaluate(alpha_det[determinant_index_left],
                                   alpha_det[determinant_index_right],
                                   beta_det[determinant_index_left],
                                   beta_det[determinant_index_right],
                                   ovlp_act,
                                   h1e_act,
                                   n_orbitals,
                                   eri_act,
                                   retain_spin_pair_evaluations);
}

DeterminantPairEvaluation evaluate_full_determinant_pair_with_optional_same_spin_cache(
    const SameSpinPairCacheContext* same_spin_pair_cache,
    const DeterminantPairEvaluator& pair_evaluator,
    SpinStringRows alpha_det,
    SpinStringRows beta_det,
    Eigen::Index determinant_index_left,
    Eigen::Index determinant_index_right,
    const std::vector<double>& ovlp_act,
    const Eigen::Ref<const Eigen::MatrixXd>& h1e_act,
    int n_orbitals,
    const xmvb::integrals::ActiveSpaceTwoElectronResult& active_space_two_electron_result,
    bool retain_spin_pair_evaluations) {
    if (same_spin_pair_cache != nullptr && same_spin_pair_cache->enabled()) {
        const auto& alpha_occ_L = alpha_det[determinant_index_left];
        const auto& alpha_occ_R = alpha_det[determinant_index_right];
        const auto& beta_occ_L = beta_det[determinant_index_left];
        const auto& beta_occ_R = beta_det[determinant_index_right];
        const int alpha_left_id = same_spin_pair_cache->alpha_reuse_table
                                      .determinant_to_unique_id[determinant_index_left];
        const int alpha_right_id = same_spin_pair_cache->alpha_reuse_table
                                       .determinant_to_unique_id[determinant_index_right];
        const int beta_left_id =
            same_spin_pair_cache->beta_reuse_table.determinant_to_unique_id[determinant_index_left];
        const int beta_right_id = same_spin_pair_cache->beta_reuse_table
                                      .determinant_to_unique_id[determinant_index_right];
        const auto& alpha_pair_result =
            same_spin_pair_cache->alpha_pair_cache[ordered_spin_pair_storage_index(
                alpha_left_id,
                alpha_right_id,
                static_cast<int>(
                    same_spin_pair_cache->alpha_reuse_table.unique_determinants.size()))];
        const auto& beta_pair_result =
            same_spin_pair_cache->beta_pair_cache_ref()[ordered_spin_pair_storage_index(
                beta_left_id,
                beta_right_id,
                static_cast<int>(
                    same_spin_pair_cache->beta_reuse_table.unique_determinants.size()))];

        double opposite_spin_coupling = 0.0;
        if (has_materialized_opposite_spin_scalar_cache(*same_spin_pair_cache)) {
            const std::size_t opp_cache_idx = same_spin_pair_cache->opposite_spin_cache_index(
                alpha_left_id, alpha_right_id, beta_left_id, beta_right_id);
            if (!same_spin_pair_cache->opposite_spin_cache_computed[opp_cache_idx]) {
                const double opp_coupling =
                    evaluate_opposite_spin_coulomb_coupling(alpha_occ_L,
                                                            alpha_occ_R,
                                                            alpha_pair_result,
                                                            beta_occ_L,
                                                            beta_occ_R,
                                                            beta_pair_result,
                                                            n_orbitals,
                                                            active_space_two_electron_result);
                const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)
                    ->opposite_spin_cache[opp_cache_idx] = opp_coupling;
                const_cast<SameSpinPairCacheContext*>(same_spin_pair_cache)
                    ->opposite_spin_cache_computed[opp_cache_idx] = true;
            }
            opposite_spin_coupling = same_spin_pair_cache->opposite_spin_cache[opp_cache_idx];
        } else {
            opposite_spin_coupling =
                evaluate_opposite_spin_coulomb_coupling(alpha_occ_L,
                                                        alpha_occ_R,
                                                        alpha_pair_result,
                                                        beta_occ_L,
                                                        beta_occ_R,
                                                        beta_pair_result,
                                                        n_orbitals,
                                                        active_space_two_electron_result);
        }

        // Combine results using cached opposite-spin coupling
        DeterminantPairEvaluation result;
        if (retain_spin_pair_evaluations) {
            result.alpha = alpha_pair_result;
            result.beta = beta_pair_result;
        }
        result.overlap_determinant = alpha_pair_result.overlap_result.overlap_determinant *
                                     beta_pair_result.overlap_result.overlap_determinant;
        result.one_electron_hamiltonian = alpha_pair_result.one_electron_hamiltonian *
                                              beta_pair_result.overlap_result.overlap_determinant +
                                          beta_pair_result.one_electron_hamiltonian *
                                              alpha_pair_result.overlap_result.overlap_determinant;
        result.total_hamiltonian = alpha_pair_result.total_hamiltonian *
                                       beta_pair_result.overlap_result.overlap_determinant +
                                   beta_pair_result.total_hamiltonian *
                                       alpha_pair_result.overlap_result.overlap_determinant +
                                   opposite_spin_coupling;

        return result;
    }

    return pair_evaluator.evaluate(alpha_det[determinant_index_left],
                                   alpha_det[determinant_index_right],
                                   beta_det[determinant_index_left],
                                   beta_det[determinant_index_right],
                                   ovlp_act,
                                   h1e_act,
                                   n_orbitals,
                                   active_space_two_electron_result,
                                   retain_spin_pair_evaluations);
}

} // namespace xmvb::vb
