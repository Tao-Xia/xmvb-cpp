#include "vbscf/integrals/ao/one_electron/backpropagator.hpp"

#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "core/openmp.hpp"
#include "vbscf/core/storage/eigen.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"
#include "vbscf/integrals/ao/one_electron/ri_operator.hpp"

namespace xmvb::vb {
namespace {

std::vector<double> encode_symmetric_gradient(
    const std::vector<double>& gradient,
    int n_bf) {
  const std::size_t size =
      static_cast<std::size_t>(n_bf) * static_cast<std::size_t>(n_bf);
  if (n_bf <= 0 || gradient.size() != size) {
    throw std::invalid_argument("invalid symmetric AO gradient size");
  }
  const Eigen::Map<const Eigen::MatrixXd> matrix(
      gradient.data(), n_bf, n_bf);
  Eigen::MatrixXd lower = matrix.triangularView<Eigen::Lower>();
  lower.diagonal() *= 0.5;
  return {lower.data(), lower.data() + lower.size()};
}

struct SignedFactors {
  Eigen::MatrixXd vectors;
  int n_positive = 0;
  int n_negative = 0;
};

SignedFactors factor_symmetric_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("RI factorization requires a square matrix");
  }
  SignedFactors result;
  if (matrix.rows() == 0) {
    result.vectors.resize(0, 0);
    return result;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("failed to diagonalize active gradient");
  }
  const Eigen::VectorXd eigenvalues = solver.eigenvalues();
  const double scale = eigenvalues.cwiseAbs().maxCoeff();
  const double cutoff = std::numeric_limits<double>::epsilon() *
      static_cast<double>(matrix.rows()) * scale;
  for (const double value : eigenvalues) {
    result.n_positive += value > cutoff;
    result.n_negative += value < -cutoff;
  }
  result.vectors.resize(
      matrix.rows(), result.n_positive + result.n_negative);
  int positive = 0;
  int negative = result.n_positive;
  for (int index = 0; index < eigenvalues.size(); ++index) {
    const double value = eigenvalues(index);
    if (value > cutoff) {
      result.vectors.col(positive++) =
          solver.eigenvectors().col(index) * std::sqrt(value);
    } else if (value < -cutoff) {
      result.vectors.col(negative++) =
          solver.eigenvectors().col(index) * std::sqrt(-value);
    }
  }
  return result;
}

AoEffectiveOneElectronRiLowRankFactors build_ri_factors(
    const std::vector<double>& active_gradient_storage,
    const OrbitalPreparationResult& orbitals,
    int n_bf,
    int n_inactive,
    int n_active) {
  if (n_bf <= 0 || n_active <= 0 || n_inactive < 0 ||
      n_inactive + n_active > n_bf ||
      active_gradient_storage.size() !=
          static_cast<std::size_t>(n_active) * n_active ||
      orbitals.auxiliary_orbital_matrix.size() !=
          static_cast<std::size_t>(n_bf) * n_bf ||
      orbitals.inactive_density_low_rank_factors.rows() != n_bf ||
      orbitals.inactive_density_low_rank_factors.cols() != n_inactive) {
    throw std::invalid_argument("invalid RI AO-H1E factor dimensions");
  }

  const Eigen::Map<const Eigen::MatrixXd> active_gradient(
      active_gradient_storage.data(), n_active, n_active);
  const SignedFactors active_factors =
      factor_symmetric_matrix(active_gradient + active_gradient.transpose());
  const int n_positive = n_inactive + active_factors.n_positive;
  Eigen::MatrixXd factors = Eigen::MatrixXd::Zero(
      n_bf, n_positive + active_factors.n_negative);
  if (n_inactive > 0) {
    factors.leftCols(n_inactive) =
        std::sqrt(2.0) * orbitals.inactive_density_low_rank_factors;
  }
  if (active_factors.vectors.cols() > 0) {
    const Eigen::Map<const Eigen::MatrixXd> auxiliary(
        orbitals.auxiliary_orbital_matrix.data(), n_bf, n_bf);
    factors.middleCols(n_inactive, active_factors.vectors.cols()).noalias() =
        auxiliary.middleCols(n_inactive, n_active) * active_factors.vectors;
  }
  return {
      .scaled_factor_matrix = std::move(factors),
      .n_positive_components = n_positive,
      .n_negative_components = active_factors.n_negative,
  };
}

AoEffectiveOneElectronBackpropagationResult backpropagate_ri(
    const std::vector<double>& gradient_storage,
    const RiAoFactorization& ri,
    int n_bf) {
  const std::size_t size =
      static_cast<std::size_t>(n_bf) * static_cast<std::size_t>(n_bf);
  if (n_bf <= 0 || gradient_storage.size() != size ||
      ri.n_basis_functions != n_bf) {
    throw std::invalid_argument("invalid RI AO-H1E backpropagation dimensions");
  }
  const Eigen::Map<const Eigen::MatrixXd> gradient(
      gradient_storage.data(), n_bf, n_bf);
  const Eigen::MatrixXd symmetric = gradient + gradient.transpose();
  auto density_gradient = apply_ao_effective_one_electron_ri_operator(
      std::vector<double>(symmetric.data(), symmetric.data() + symmetric.size()),
      ri,
      n_bf,
      {.attempt_spectral_factorization = true});
  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient =
      encode_symmetric_gradient(density_gradient, n_bf);
  return result;
}

AoEffectiveOneElectronBackpropagationResult backpropagate_ri(
    const std::vector<double>& active_gradient,
    const OrbitalPreparationResult& orbitals,
    const RiAoFactorization& ri,
    int n_bf,
    int n_inactive,
    int n_active) {
  auto density_gradient = apply_ao_effective_one_electron_ri_operator(
      build_ri_factors(
          active_gradient, orbitals, n_bf, n_inactive, n_active),
      ri,
      n_bf);
  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient =
      encode_symmetric_gradient(density_gradient, n_bf);
  return result;
}

}  // namespace

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& gradient,
    const RiAoFactorization& ri,
    int n_bf) const {
  return backpropagate_ri(gradient, ri, n_bf);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& gradient,
    const RiAoFactorization& ri,
    int n_bf) const {
  return backpropagate_ri(flatten_matrix_column_major(gradient), ri, n_bf);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& active_gradient,
    const OrbitalPreparationResult& orbitals,
    const RiAoFactorization& ri,
    int n_bf,
    int n_inactive,
    int n_active) const {
  return backpropagate_ri(
      active_gradient, orbitals, ri, n_bf, n_inactive, n_active);
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const std::vector<double>& gradient_storage,
    const AoIntegralInput& ao) const {
  const int n_bf = ao.n_basis_functions;
  const std::size_t size =
      static_cast<std::size_t>(n_bf) * static_cast<std::size_t>(n_bf);
  if (n_bf <= 0 || gradient_storage.size() != size) {
    throw std::invalid_argument("invalid AO-H1E gradient dimensions");
  }
  const Eigen::Map<const Eigen::MatrixXd> gradient(
      gradient_storage.data(), n_bf, n_bf);
  const Eigen::MatrixXd symmetric = gradient + gradient.transpose();
  AoEffectiveOneElectronBackpropagationResult result;
  result.inactive_density_gradient = apply_ao_h1e_transpose(
      symmetric.data(), ao, effective_openmp_thread_count());
  return result;
}

AoEffectiveOneElectronBackpropagationResult
AoEffectiveOneElectronBackpropagator::backpropagate(
    const Eigen::Ref<const Eigen::MatrixXd>& gradient,
    const AoIntegralInput& ao) const {
  if (gradient.rows() != ao.n_basis_functions ||
      gradient.cols() != ao.n_basis_functions) {
    throw std::invalid_argument("invalid AO-H1E gradient dimensions");
  }
  return backpropagate(flatten_matrix_column_major(gradient), ao);
}

}  // namespace xmvb::vb
