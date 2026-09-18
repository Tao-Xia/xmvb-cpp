#include "vbscf/derivatives/hessian/exact/ao_one_electron_internal.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "core/openmp.hpp"
#include "vbscf/integrals/ao/contracts/input.hpp"
#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb::detail {

int choose_exact_ao_h1e_thread_count(
    const OrbitalPreparationInput& input) {
  if (input.n_basis_functions == 0) {
    throw std::invalid_argument("AO-H1E basis size must be positive");
  }
  return std::min(
      effective_openmp_thread_count(),
      static_cast<int>(input.n_basis_functions));
}

void symmetrize_exact_ao_h1e_forward(
    Eigen::Ref<Eigen::MatrixXd> delta_h1e) {
  if (delta_h1e.rows() != delta_h1e.cols()) {
    throw std::invalid_argument(
        "AO-H1E forward response must be a square matrix");
  }
  for (Eigen::Index row = 0; row < delta_h1e.rows(); ++row) {
    for (Eigen::Index column = 0; column <= row; ++column) {
      delta_h1e(row, column) += delta_h1e(column, row);
      delta_h1e(column, row) = delta_h1e(row, column);
    }
  }
}

void apply_forward_exact_ao_one_electron_response(
    const Eigen::MatrixXd& density,
    const AoIntegralInput& ao,
    const OrbitalPreparationInput& orbital_input,
    std::vector<double>* delta_h1e) {
  const int n_bf = ao.n_basis_functions;
  if (n_bf <= 0 || density.rows() != n_bf || density.cols() != n_bf ||
      delta_h1e == nullptr) {
    throw std::invalid_argument("invalid forward AO-H1E response arguments");
  }
  *delta_h1e = apply_ao_h1e(
      density.data(),
      ao,
      choose_exact_ao_h1e_thread_count(orbital_input));
  Eigen::Map<Eigen::MatrixXd> delta(delta_h1e->data(), n_bf, n_bf);
  symmetrize_exact_ao_h1e_forward(delta);
}

void apply_fused_exact_ao_one_electron_response(
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& gradient,
    const AoIntegralInput& ao,
    const OrbitalPreparationInput& orbital_input,
    AoH1eFusedWorkspace* workspace,
    Eigen::MatrixXd* symmetric_gradient,
    std::vector<double>* delta_h1e,
    std::vector<double>* density_gradient) {
  const int n_bf = ao.n_basis_functions;
  if (n_bf <= 0 || density.rows() != n_bf || density.cols() != n_bf ||
      gradient.rows() != n_bf || gradient.cols() != n_bf ||
      symmetric_gradient == nullptr || delta_h1e == nullptr ||
      density_gradient == nullptr) {
    throw std::invalid_argument("invalid fused AO-H1E response arguments");
  }

  symmetric_gradient->resizeLike(gradient);
  *symmetric_gradient = gradient + gradient.transpose();
  apply_ao_h1e_fused(
      density.data(),
      symmetric_gradient->data(),
      ao,
      choose_exact_ao_h1e_thread_count(orbital_input),
      workspace,
      delta_h1e,
      density_gradient);

  Eigen::Map<Eigen::MatrixXd> delta(delta_h1e->data(), n_bf, n_bf);
  symmetrize_exact_ao_h1e_forward(delta);
}

}  // namespace xmvb::vb::detail
