#include "vbscf/integrals/active/two_electron/response/ri.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/openmp.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"

namespace xmvb::vb {
namespace {

std::size_t packed_pair_count(int dimension) {
  const std::size_t size = static_cast<std::size_t>(dimension);
  return size * (size + 1) / 2;
}

void validate_cache(
    const RiActiveTwoElectronResponseCache& cache,
    const char* operation) {
  const int n_bf = cache.n_basis_functions;
  const int n_active = cache.n_active_orbitals;
  const int n_auxiliary = cache.n_auxiliary_functions;
  if (n_bf <= 0 || n_active <= 0 || n_auxiliary <= 0) {
    throw std::invalid_argument(
        std::string(operation) + " requires positive RI dimensions");
  }
  const std::size_t n_ao_pairs = packed_pair_count(n_bf);
  const std::size_t n_active_pairs = packed_pair_count(n_active);
  if (cache.ao_pair_first_indices.size() != n_ao_pairs ||
      cache.ao_pair_second_indices.size() != n_ao_pairs ||
      cache.active_pair_first_indices.size() != n_active_pairs ||
      cache.active_pair_second_indices.size() != n_active_pairs) {
    throw std::invalid_argument(
        std::string(operation) + " has inconsistent pair-index tables");
  }
  if (cache.metric_whitened_ao_pair_factors == nullptr ||
      cache.metric_whitened_ao_pair_factors->rows() != n_auxiliary ||
      cache.metric_whitened_ao_pair_factors->cols() !=
          static_cast<Eigen::Index>(n_ao_pairs)) {
    throw std::invalid_argument(
        std::string(operation) + " has inconsistent AO RI factors");
  }
  if (cache.accepted_active_coefficients == nullptr ||
      cache.accepted_active_coefficients->rows() != n_bf ||
      cache.accepted_active_coefficients->cols() != n_active) {
    throw std::invalid_argument(
        std::string(operation) + " has inconsistent accepted coefficients");
  }
  if (cache.accepted_active_pair_factors == nullptr ||
      cache.accepted_active_pair_factors->rows() != n_auxiliary ||
      cache.accepted_active_pair_factors->cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument(
        std::string(operation) + " has inconsistent active-pair factors");
  }
}

Eigen::MatrixXd build_active_pair_adjoint(
    const std::vector<double>& packed_gradient,
    const RiActiveTwoElectronResponseCache& cache) {
  const Eigen::Index n_active_pairs = static_cast<Eigen::Index>(
      cache.active_pair_first_indices.size());
  const std::size_t expected_size =
      static_cast<std::size_t>(n_active_pairs) *
      (static_cast<std::size_t>(n_active_pairs) + 1) / 2;
  if (packed_gradient.size() != expected_size) {
    throw std::invalid_argument(
        "packed RI active-2e adjoint has the wrong size");
  }

  Eigen::MatrixXd active_pair_adjoint(n_active_pairs, n_active_pairs);
  for (Eigen::Index row = 0; row < n_active_pairs; ++row) {
    for (Eigen::Index column = 0; column < n_active_pairs; ++column) {
      const int first = static_cast<int>(row > column ? row : column);
      const int second = static_cast<int>(row > column ? column : row);
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(first, second);
      double value = packed_gradient[packed_index];
      if (row == column) {
        value *= 2.0;
      }
      active_pair_adjoint(row, column) = value;
    }
  }
  return active_pair_adjoint;
}

bool direct_pair_factor_direction_has_lower_flop_count(
    int n_basis_functions,
    int n_active_orbitals) {
  // Packed Q' followed by L * Q' costs approximately
  // 2 n_aux n_ao_pair n_active_pair FLOPs.  The direct two-stage
  // three-index transform costs 4 n_aux n_bf n_active (n_bf + n_active).
  // The common auxiliary dimension cancels from the comparison.
  const long double n_bf = n_basis_functions;
  const long double n_active = n_active_orbitals;
  const long double packed_cost =
      0.5L * n_bf * (n_bf + 1.0L) *
      n_active * (n_active + 1.0L);
  const long double direct_cost =
      4.0L * n_bf * n_active * (n_bf + n_active);
  // The packed path is one large GEMM, whereas the direct path is a batch of
  // AO-slice GEMMs plus two final contractions.  Admit the direct algorithm
  // only after its arithmetic count is at least twofold smaller, leaving the
  // near-crossover regime to the more efficient single-GEMM kernel.
  return 2.0L * direct_cost < packed_cost;
}

Eigen::MatrixXd compute_direct_active_pair_factor_direction(
    const RiActiveTwoElectronResponseCache& cache,
    const Eigen::Ref<const Eigen::MatrixXd>& active_direction) {
  const int n_bf = cache.n_basis_functions;
  const int n_active = cache.n_active_orbitals;
  const int n_auxiliary = cache.n_auxiliary_functions;
  const auto& ao_factors = *cache.metric_whitened_ao_pair_factors;
  const auto& active_coefficients = *cache.accepted_active_coefficients;

  Eigen::MatrixXd combined_coefficients(n_bf, 2 * n_active);
  combined_coefficients.leftCols(n_active) = active_coefficients;
  combined_coefficients.rightCols(n_active) = active_direction;

  // Transform one contiguous auxiliary-major AO slice at a time.  Columns of
  // `transformed` contain (L_A C) and (L_A D), grouped first by active orbital
  // and then by auxiliary index.
  Eigen::MatrixXd transformed(2 * n_auxiliary * n_active, n_bf);
  const int n_threads = std::min(
      xmvb::effective_openmp_thread_count(),
      n_bf);
#pragma omp parallel num_threads(n_threads)
  {
    Eigen::MatrixXd ao_slice(n_auxiliary, n_bf);
#pragma omp for schedule(static)
    for (int first_basis = 0; first_basis < n_bf; ++first_basis) {
      for (int second_basis = 0; second_basis < n_bf; ++second_basis) {
        const int larger = std::max(first_basis, second_basis);
        const int smaller = std::min(first_basis, second_basis);
        const Eigen::Index packed_pair =
            static_cast<Eigen::Index>(larger) * (larger + 1) / 2 + smaller;
        ao_slice.col(second_basis) = ao_factors.col(packed_pair);
      }
      Eigen::Map<Eigen::MatrixXd> transformed_slice(
          transformed.col(first_basis).data(),
          n_auxiliary,
          2 * n_active);
      transformed_slice.noalias() = ao_slice * combined_coefficients;
    }
  }

  const Eigen::Index channel_rows =
      static_cast<Eigen::Index>(n_auxiliary) * n_active;
  const auto accepted_transformed = transformed.topRows(channel_rows);
  const auto directional_transformed = transformed.bottomRows(channel_rows);
  const Eigen::MatrixXd accepted_times_direction =
      accepted_transformed * active_direction;
  const Eigen::MatrixXd direction_times_accepted =
      directional_transformed * active_coefficients;

  Eigen::MatrixXd result(
      n_auxiliary,
      static_cast<Eigen::Index>(packed_pair_count(n_active)));
  Eigen::Index active_pair = 0;
  for (int first_active = 0;
       first_active < n_active;
       ++first_active) {
    for (int second_active = 0;
         second_active <= first_active;
         ++second_active, ++active_pair) {
      result.col(active_pair) =
          accepted_times_direction.middleRows(
              static_cast<Eigen::Index>(second_active) * n_auxiliary,
              n_auxiliary).col(first_active) +
          direction_times_accepted.middleRows(
              static_cast<Eigen::Index>(second_active) * n_auxiliary,
              n_auxiliary).col(first_active);
    }
  }
  return result;
}

}  // namespace

RiActiveTwoElectronResponseCache build_ri_active_two_electron_response_cache(
    const RiAoFactorization& ao_ri_factorization,
    const ActiveSpaceTwoElectronResult& accepted_active_space_result,
    int n_active_orbitals) {
  if (ao_ri_factorization.n_basis_functions <= 0 ||
      ao_ri_factorization.n_auxiliary_functions <= 0 ||
      n_active_orbitals <= 0) {
    throw std::invalid_argument(
        "RI active-2e response cache dimensions must be positive");
  }
  if (accepted_active_space_result.representation !=
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    throw std::invalid_argument(
        "RI active-2e response cache requires an RI active-space result");
  }

  const int n_bf = ao_ri_factorization.n_basis_functions;
  const std::size_t n_ao_pairs = packed_pair_count(n_bf);
  const std::size_t n_active_pairs = packed_pair_count(n_active_orbitals);
  if (ao_ri_factorization.n_packed_ao_pairs !=
          static_cast<int>(n_ao_pairs) ||
      ao_ri_factorization.metric_whitened_ao_pair_factors.rows() !=
          ao_ri_factorization.n_auxiliary_functions ||
      ao_ri_factorization.metric_whitened_ao_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_ao_pairs)) {
    throw std::invalid_argument(
        "RI active-2e response cache AO factor dimensions are inconsistent");
  }
  if (accepted_active_space_result.n_auxiliary_functions !=
          ao_ri_factorization.n_auxiliary_functions ||
      accepted_active_space_result.dense_active_coefficients.rows() != n_bf ||
      accepted_active_space_result.dense_active_coefficients.cols() !=
          n_active_orbitals ||
      accepted_active_space_result.ri_active_pair_factors.rows() !=
          ao_ri_factorization.n_auxiliary_functions ||
      accepted_active_space_result.ri_active_pair_factors.cols() !=
          static_cast<Eigen::Index>(n_active_pairs)) {
    throw std::invalid_argument(
        "RI active-2e response cache accepted dimensions are inconsistent");
  }

  RiActiveTwoElectronResponseCache cache;
  cache.n_basis_functions = n_bf;
  cache.n_active_orbitals = n_active_orbitals;
  cache.n_auxiliary_functions =
      ao_ri_factorization.n_auxiliary_functions;
  cache.metric_whitened_ao_pair_factors =
      &ao_ri_factorization.metric_whitened_ao_pair_factors;
  cache.accepted_active_coefficients =
      &accepted_active_space_result.dense_active_coefficients;
  cache.accepted_active_pair_factors =
      &accepted_active_space_result.ri_active_pair_factors;

  cache.ao_pair_first_indices.reserve(n_ao_pairs);
  cache.ao_pair_second_indices.reserve(n_ao_pairs);
  for (int first = 0; first < n_bf; ++first) {
    for (int second = 0; second <= first; ++second) {
      cache.ao_pair_first_indices.push_back(first);
      cache.ao_pair_second_indices.push_back(second);
    }
  }
  cache.active_pair_first_indices.reserve(n_active_pairs);
  cache.active_pair_second_indices.reserve(n_active_pairs);
  for (int first = 0; first < n_active_orbitals; ++first) {
    for (int second = 0; second <= first; ++second) {
      cache.active_pair_first_indices.push_back(first);
      cache.active_pair_second_indices.push_back(second);
    }
  }
  validate_cache(cache, "RI active-2e response cache");
  return cache;
}

Eigen::MatrixXd compute_ri_active_pair_factor_directional_derivative(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction) {
  validate_cache(accepted_cache, "RI active-pair factor direction");
  if (dense_active_direction.rows() != accepted_cache.n_basis_functions ||
      dense_active_direction.cols() != accepted_cache.n_active_orbitals) {
    throw std::invalid_argument(
        "RI active-pair factor direction has the wrong shape");
  }
  if (direct_pair_factor_direction_has_lower_flop_count(
          accepted_cache.n_basis_functions,
          accepted_cache.n_active_orbitals)) {
    return compute_direct_active_pair_factor_direction(
        accepted_cache,
        dense_active_direction);
  }
  PackedOrbitalPairMapMatrix mixed_pair_coefficients;
  build_packed_orbital_pair_map_directional_derivative(
      *accepted_cache.accepted_active_coefficients,
      dense_active_direction,
      &mixed_pair_coefficients);
  return *accepted_cache.metric_whitened_ao_pair_factors *
      mixed_pair_coefficients;
}

std::vector<double>
compute_ri_packed_active_two_electron_integral_directional_derivative(
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>&
        directional_active_pair_factors) {
  validate_cache(accepted_cache, "RI packed active-2e direction");
  const Eigen::MatrixXd& accepted_factors =
      *accepted_cache.accepted_active_pair_factors;
  if (directional_active_pair_factors.rows() != accepted_factors.rows() ||
      directional_active_pair_factors.cols() != accepted_factors.cols()) {
    throw std::invalid_argument(
        "RI active-pair factor direction has the wrong shape");
  }

  const Eigen::MatrixXd one_sided_contraction =
      accepted_factors.transpose() * directional_active_pair_factors;
  const Eigen::Index n_active_pairs = accepted_factors.cols();
  std::vector<double> packed_direction(
      static_cast<std::size_t>(n_active_pairs) *
          (static_cast<std::size_t>(n_active_pairs) + 1) / 2,
      0.0);
  for (Eigen::Index first = 0; first < n_active_pairs; ++first) {
    for (Eigen::Index second = 0; second <= first; ++second) {
      const int packed_index =
          TwoElectronIndexer::packed_pair_of_pairs_index(
              static_cast<int>(first),
              static_cast<int>(second));
      packed_direction[packed_index] =
          one_sided_contraction(first, second) +
          one_sided_contraction(second, first);
    }
  }
  return packed_direction;
}

Eigen::MatrixXd backpropagate_ri_packed_active_two_electron_gradient(
    const std::vector<double>& packed_active_two_electron_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache) {
  validate_cache(accepted_cache, "RI packed active-2e pullback");
  const Eigen::MatrixXd active_pair_adjoint =
      build_active_pair_adjoint(
          packed_active_two_electron_gradient,
          accepted_cache);
  const Eigen::MatrixXd factor_adjoint =
      *accepted_cache.accepted_active_pair_factors * active_pair_adjoint;
  PackedOrbitalPairMapMatrix ao_pair_adjoint;
  ao_pair_adjoint.noalias() =
      accepted_cache.metric_whitened_ao_pair_factors->transpose() *
      factor_adjoint;
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      accepted_cache.n_basis_functions,
      accepted_cache.n_active_orbitals);
  accumulate_packed_orbital_pair_map_adjoint(
      ao_pair_adjoint,
      *accepted_cache.accepted_active_coefficients,
      &result);
  return result;
}

Eigen::MatrixXd
apply_ri_packed_active_two_electron_adjoint_hessian_vector(
    const std::vector<double>& packed_active_two_electron_gradient,
    const RiActiveTwoElectronResponseCache& accepted_cache,
    const Eigen::Ref<const Eigen::MatrixXd>& dense_active_direction,
    const Eigen::MatrixXd* directional_active_pair_factors) {
  validate_cache(accepted_cache, "RI fixed-adjoint active-2e HVP");
  if (dense_active_direction.rows() != accepted_cache.n_basis_functions ||
      dense_active_direction.cols() != accepted_cache.n_active_orbitals) {
    throw std::invalid_argument(
        "RI fixed-adjoint HVP direction has the wrong orbital shape");
  }

  Eigen::MatrixXd owned_factor_direction;
  const Eigen::MatrixXd* factor_direction =
      directional_active_pair_factors;
  if (factor_direction == nullptr) {
    owned_factor_direction =
        compute_ri_active_pair_factor_directional_derivative(
            accepted_cache,
            dense_active_direction);
    factor_direction = &owned_factor_direction;
  }
  if (factor_direction->rows() != accepted_cache.n_auxiliary_functions ||
      factor_direction->cols() !=
          static_cast<Eigen::Index>(
              accepted_cache.active_pair_first_indices.size())) {
    throw std::invalid_argument(
        "precomputed RI active-pair factor direction has the wrong shape");
  }

  const Eigen::MatrixXd active_pair_adjoint =
      build_active_pair_adjoint(
          packed_active_two_electron_gradient,
          accepted_cache);
  const Eigen::MatrixXd accepted_factor_adjoint =
      *accepted_cache.accepted_active_pair_factors * active_pair_adjoint;
  const Eigen::MatrixXd directional_factor_adjoint =
      *factor_direction * active_pair_adjoint;
  PackedOrbitalPairMapMatrix accepted_ao_pair_adjoint;
  accepted_ao_pair_adjoint.noalias() =
      accepted_cache.metric_whitened_ao_pair_factors->transpose() *
      accepted_factor_adjoint;
  PackedOrbitalPairMapMatrix directional_ao_pair_adjoint;
  directional_ao_pair_adjoint.noalias() =
      accepted_cache.metric_whitened_ao_pair_factors->transpose() *
      directional_factor_adjoint;

  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
      accepted_cache.n_basis_functions,
      accepted_cache.n_active_orbitals);
  accumulate_packed_orbital_pair_map_adjoint(
      directional_ao_pair_adjoint,
      *accepted_cache.accepted_active_coefficients,
      &result);
  accumulate_packed_orbital_pair_map_adjoint(
      accepted_ao_pair_adjoint,
      dense_active_direction,
      &result);
  return result;
}

}  // namespace xmvb::vb
