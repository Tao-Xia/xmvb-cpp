#include "vb/scf/vb_scf_evaluator.hpp"

#include <stdexcept>
#include <utility>

namespace xmvb::vb {

VbScfEvaluator::VbScfEvaluator(
    OrbitalParameterCodec orbital_parameter_codec,
    HamiltonianOverlapBuilder hamiltonian_overlap_builder,
    xmvb::core::GeneralizedEigensolver generalized_eigensolver)
    : orbital_parameter_codec_(std::move(orbital_parameter_codec)),
      hamiltonian_overlap_builder_(std::move(hamiltonian_overlap_builder)),
      generalized_eigensolver_(std::move(generalized_eigensolver)) {}

VbScfResult VbScfEvaluator::evaluate(
    const OrbitalParameterVector& parameters,
    VbWavefunctionData& wavefunction_data) const {
  if (wavefunction_data.dims.n_structures <= 0) {
    throw std::invalid_argument("dims.n_structures must be positive");
  }

  const OrbitalSpace orbital_space =
      orbital_parameter_codec_.decode(parameters, wavefunction_data);
  const HamiltonianOverlapMatrices matrices =
      hamiltonian_overlap_builder_.build(orbital_space, wavefunction_data);

  const auto eigen_result = generalized_eigensolver_.solve(
      matrices.hamiltonian_matrix,
      matrices.overlap_matrix,
      wavefunction_data.dims.n_structures);

  VbScfResult result;
  result.one_electron_energy = wavefunction_data.one_electron_energy;
  result.valence_bond_structure_energy =
      eigen_result.eigenvalues.empty() ? 0.0 : eigen_result.eigenvalues.front();
  result.total_energy =
      result.one_electron_energy + result.valence_bond_structure_energy;
  result.eigenvector_matrix = eigen_result.eigenvector_matrix;
  result.electronic_state_energies = eigen_result.eigenvalues;

  const int dimension = wavefunction_data.dims.n_structures;
  for (int i = 0; i < dimension; ++i) {
    result.average_structure_overlap +=
        matrices.overlap_matrix[static_cast<std::size_t>(i) * dimension + i];
  }
  result.average_structure_overlap /= static_cast<double>(dimension);

  wavefunction_data.hamiltonian_matrix = matrices.hamiltonian_matrix;
  wavefunction_data.overlap_matrix = matrices.overlap_matrix;
  wavefunction_data.eigenvector_matrix = result.eigenvector_matrix;
  wavefunction_data.electronic_state_energies = result.electronic_state_energies;

  return result;
}

}  // namespace xmvb::vb
