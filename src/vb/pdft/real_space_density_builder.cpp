#include "vb/pdft/real_space_density_builder.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb::pdft {

Eigen::VectorXd RealSpaceDensityBuilder::evaluate_electron_density(
    const Eigen::MatrixXd& ao_density_matrix,
    const AoGridValues& ao_values) const {
  const int n_points = ao_values.n_points();
  const int n_basis = ao_values.n_basis_functions();

  if (ao_density_matrix.rows() != n_basis || ao_density_matrix.cols() != n_basis) {
    throw std::invalid_argument("AO density matrix size mismatch");
  }

  // Evaluate rho(r) = sum_{mu,nu} gamma_{mu,nu} chi_mu(r) chi_nu(r)
  //
  // Efficient implementation:
  //   rho = diag(chi * gamma * chi^T)
  //
  // where chi is n_points x n_basis matrix of AO values.

  Eigen::VectorXd rho(n_points);

  // Compute chi * gamma (n_points x n_basis)
  Eigen::MatrixXd chi_gamma = ao_values.values * ao_density_matrix;

  // Compute rho as element-wise product and sum
  // rho(g) = sum_mu (chi * gamma)_{g,mu} * chi_{g,mu}
  #pragma omp parallel for schedule(static) if(n_points > 1000)
  for (int g = 0; g < n_points; ++g) {
    rho(g) = chi_gamma.row(g).dot(ao_values.values.row(g));
  }

  return rho;
}

Eigen::VectorXd RealSpaceDensityBuilder::evaluate_on_top_pair_density(
    const SelectedStateExactPhysicalOnTopPairDensityContext& on_top_context,
    const AoGridValues& ao_values) const {
  if (ao_values.n_basis_functions() != on_top_context.n_basis_functions) {
    throw std::invalid_argument("AO basis dimension mismatch in on-top evaluation");
  }
  // The on-top evaluator expects probe vectors as columns, while `ao_values`
  // stores one grid point per row.  Transposing once lets the block
  // contraction reuse all determinant-pair cofactors across the full grid.
  return evaluate_selected_state_exact_physical_on_top_pair_density_batch(
      on_top_context,
      ao_values.values.transpose());
}

RealSpaceDensities RealSpaceDensityBuilder::build(
    const Eigen::MatrixXd& ao_density_matrix,
    const SelectedStateExactPhysicalOnTopPairDensityContext& on_top_context,
    const AoGridValues& ao_values) const {
  if (ao_values.n_points() <= 0) {
    throw std::invalid_argument("No grid points in AO values");
  }

  RealSpaceDensities densities;

  // Evaluate electron density rho(r)
  densities.rho = evaluate_electron_density(ao_density_matrix, ao_values);

  // Evaluate on-top pair density Pi(r)
  densities.pi = evaluate_on_top_pair_density(on_top_context, ao_values);

  return densities;
}

}  // namespace xmvb::vb::pdft
