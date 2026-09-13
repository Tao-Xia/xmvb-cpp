#include "vbscf/integrals/ao/one_electron/builder.hpp"

#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "core/openmp.hpp"
#include "vbscf/core/storage/eigen.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"

namespace xmvb::vb {
namespace {

void validate_ao_matrices(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    int n_bf) {
  if (n_bf <= 0 || density.rows() != n_bf || density.cols() != n_bf ||
      core.rows() != n_bf || core.cols() != n_bf) {
    throw std::invalid_argument("invalid AO-H1E matrix dimensions");
  }
}

AoEffectiveOneElectronResult make_result(
    std::vector<double> g_storage,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    int n_bf) {
  Eigen::Map<Eigen::MatrixXd> g(g_storage.data(), n_bf, n_bf);
  for (int row = 0; row < n_bf; ++row) {
    for (int column = 0; column <= row; ++column) {
      g(row, column) += g(column, row);
      g(column, row) = g(row, column);
    }
  }
  AoEffectiveOneElectronResult result;
  result.ao_coulomb_exchange_matrix = g;
  result.ao_effective_h1e = core + g;
  return result;
}

AoEffectiveOneElectronResult build_exact(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const AoIntegralInput& ao) {
  const int n_bf = ao.n_basis_functions;
  validate_ao_matrices(density, core, n_bf);
  return make_result(
      apply_ao_h1e(
          density.data(),
          ao,
          effective_openmp_thread_count()),
      core,
      n_bf);
}

AoEffectiveOneElectronResult build_ri(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const RiAoFactorization& ri,
    int n_bf) {
  validate_ao_matrices(density, core, n_bf);
  if (ri.n_basis_functions != n_bf) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }
  return make_result(
      apply_ao_effective_one_electron_ri_operator(
          flatten_matrix_column_major(density),
          ri,
          n_bf,
          {.attempt_spectral_factorization = true}),
      core,
      n_bf);
}

AoEffectiveOneElectronResult build_ri(
    const AoEffectiveOneElectronRiLowRankFactors& factors,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const RiAoFactorization& ri,
    int n_bf) {
  if (n_bf <= 0 || core.rows() != n_bf || core.cols() != n_bf) {
    throw std::invalid_argument("invalid AO-H1E core matrix dimensions");
  }
  if (ri.n_basis_functions != n_bf) {
    throw std::invalid_argument("RI basis-function count mismatch");
  }
  return make_result(
      apply_ao_effective_one_electron_ri_operator(factors, ri, n_bf),
      core,
      n_bf);
}

Eigen::Map<const Eigen::MatrixXd> map_square(
    const std::vector<double>& values,
    int n_bf) {
  const std::size_t size =
      static_cast<std::size_t>(n_bf) * static_cast<std::size_t>(n_bf);
  if (n_bf <= 0 || values.size() != size) {
    throw std::invalid_argument("invalid AO matrix storage size");
  }
  return {values.data(), n_bf, n_bf};
}

}  // namespace

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const RiAoFactorization& ri,
    int n_bf) const {
  return build_ri(map_square(density, n_bf), core, ri, n_bf);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const RiAoFactorization& ri,
    int n_bf) const {
  return build_ri(density, core, ri, n_bf);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const OrbitalPreparationResult& orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const RiAoFactorization& ri,
    int n_bf,
    int n_inactive) const {
  if (n_inactive < 0 || n_inactive > n_bf ||
      orbitals.inactive_density_low_rank_factors.rows() != n_bf ||
      orbitals.inactive_density_low_rank_factors.cols() != n_inactive) {
    throw std::invalid_argument("invalid inactive-density factor dimensions");
  }
  return build_ri(
      AoEffectiveOneElectronRiLowRankFactors{
          .scaled_factor_matrix = orbitals.inactive_density_low_rank_factors,
          .n_positive_components = n_inactive,
          .n_negative_components = 0,
      },
      core,
      ri,
      n_bf);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const AoIntegralInput& ao) const {
  return build_exact(map_square(density, ao.n_basis_functions), core, ao);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const Eigen::Ref<const Eigen::MatrixXd>& core,
    const AoIntegralInput& ao) const {
  return build_exact(density, core, ao);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const std::vector<double>& density,
    const AoIntegralInput& ao) const {
  return build_exact(
      map_square(density, ao.n_basis_functions),
      ao.ao_core_hamiltonian_matrix,
      ao);
}

AoEffectiveOneElectronResult AoEffectiveOneElectronBuilder::build(
    const Eigen::Ref<const Eigen::MatrixXd>& density,
    const AoIntegralInput& ao) const {
  return build_exact(density, ao.ao_core_hamiltonian_matrix, ao);
}

}  // namespace xmvb::vb
