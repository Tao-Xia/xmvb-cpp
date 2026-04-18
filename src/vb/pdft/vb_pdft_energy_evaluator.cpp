#include "vb/pdft/vb_pdft_energy_evaluator.hpp"

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cint.h"

#include "vb/pdft/libcint_ao_grid_evaluator.hpp"
#include "vb/pdft/libxc_functional.hpp"
#include "vb/pdft/molecular_grid.hpp"
#include "vb/pdft/real_space_density_builder.hpp"
#include "vb/pdft/selected_state_exact_physical_on_top_pair_density_builder.hpp"
#include "vb/pdft/selected_state_exact_physical_one_rdm_builder.hpp"
#include "vb/pdft/translated_spin_density.hpp"

namespace xmvb::vb::pdft {

namespace {

constexpr double kUnitWeightTolerance = 1.0e-10;
constexpr double kNormalizationConsistencyTolerance = 1.0e-10;
constexpr double kIntegralSymmetryMultipliers[] = {
    1.0,
    0.5,
    0.25,
    0.125,
};

void extract_atomic_geometry_from_libcint_input(
    const vb::LibcintInput& libcint_input,
    Eigen::MatrixXd* atomic_coords,
    Eigen::VectorXd* atomic_charges);

void validate_single_state_gradient_result(
    const vb::CppActiveSpaceGradientResult& gradient_result) {
  if (gradient_result.scf_result.selected_state_indices.size() != 1 ||
      gradient_result.scf_result.state_average_weights.size() != 1) {
    throw std::invalid_argument(
        "VB-PDFT evaluation requires exactly one selected state");
  }
  if (std::abs(gradient_result.scf_result.state_average_weights.front() - 1.0) >
      kUnitWeightTolerance) {
    throw std::invalid_argument(
        "VB-PDFT evaluation requires a unit selected-state weight");
  }
}

double resolve_nuclear_repulsion_energy(
    double requested_nuclear_repulsion_energy,
    const vb::CppActiveSpaceGradientResult& gradient_result,
    const vb::LibcintInput& libcint_input);

double compute_nuclear_repulsion_energy_from_geometry(
    const Eigen::Ref<const Eigen::MatrixXd>& atomic_coords,
    const Eigen::Ref<const Eigen::VectorXd>& atomic_charges) {
  if (atomic_coords.rows() != atomic_charges.size() || atomic_coords.cols() != 3) {
    throw std::invalid_argument(
        "nuclear repulsion geometry dimensions do not match");
  }

  double nuclear_repulsion_energy = 0.0;
  for (int left_atom = 0; left_atom < atomic_coords.rows(); ++left_atom) {
    const double left_charge = atomic_charges(left_atom);
    const Eigen::RowVector3d left_coord = atomic_coords.row(left_atom);
    for (int right_atom = left_atom + 1;
         right_atom < atomic_coords.rows();
         ++right_atom) {
      const double right_charge = atomic_charges(right_atom);
      const double distance =
          (left_coord - atomic_coords.row(right_atom)).norm();
      if (distance <= 0.0) {
        throw std::invalid_argument(
            "nuclear repulsion energy requires distinct nuclear coordinates");
      }
      nuclear_repulsion_energy += left_charge * right_charge / distance;
    }
  }
  return nuclear_repulsion_energy;
}

double compute_nuclear_repulsion_energy_from_libcint_input(
    const vb::LibcintInput& libcint_input) {
  Eigen::MatrixXd atomic_coords;
  Eigen::VectorXd atomic_charges;
  extract_atomic_geometry_from_libcint_input(
      libcint_input,
      &atomic_coords,
      &atomic_charges);
  return compute_nuclear_repulsion_energy_from_geometry(
      atomic_coords,
      atomic_charges);
}

double resolve_nuclear_repulsion_energy(
    double requested_nuclear_repulsion_energy,
    const vb::CppActiveSpaceGradientResult& gradient_result,
    const vb::LibcintInput& libcint_input) {
  if (std::isfinite(requested_nuclear_repulsion_energy)) {
    return requested_nuclear_repulsion_energy;
  }

  const double geometry_nuclear_repulsion_energy =
      compute_nuclear_repulsion_energy_from_libcint_input(libcint_input);
  const double gradient_nuclear_repulsion_energy =
      gradient_result.scf_result.nuclear_repulsion_energy;
  if (!std::isfinite(gradient_nuclear_repulsion_energy)) {
    return geometry_nuclear_repulsion_energy;
  }

  const double scale =
      std::max(1.0, std::abs(geometry_nuclear_repulsion_energy));
  if (std::abs(
          gradient_nuclear_repulsion_energy -
          geometry_nuclear_repulsion_energy) <=
      1.0e-10 * scale) {
    return gradient_nuclear_repulsion_energy;
  }
  return geometry_nuclear_repulsion_energy;
}

void extract_atomic_geometry_from_libcint_input(
    const vb::LibcintInput& libcint_input,
    Eigen::MatrixXd* atomic_coords,
    Eigen::VectorXd* atomic_charges) {
  if (atomic_coords == nullptr || atomic_charges == nullptr) {
    throw std::invalid_argument("atomic geometry output pointers must not be null");
  }
  if (libcint_input.n_atoms <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one atom");
  }
  if (libcint_input.atm.size() != xmvb::to_size(libcint_input.n_atoms) * ATM_SLOTS) {
    throw std::invalid_argument("LibcintInput atom table size mismatch");
  }
  if (libcint_input.env.empty()) {
    throw std::invalid_argument("LibcintInput env must not be empty");
  }

  *atomic_coords = Eigen::MatrixXd::Zero(libcint_input.n_atoms, 3);
  *atomic_charges = Eigen::VectorXd::Zero(libcint_input.n_atoms);
  for (int atom_index = 0; atom_index < libcint_input.n_atoms; ++atom_index) {
    const int atom_offset = xmvb::to_size(atom_index) * ATM_SLOTS;
    const int coordinate_offset = libcint_input.atm[atom_offset + PTR_COORD];
    if (coordinate_offset < 0 ||
        coordinate_offset + 2 >= static_cast<int>(libcint_input.env.size())) {
      throw std::invalid_argument("LibcintInput coordinate pointer is out of range");
    }
    (*atomic_coords)(atom_index, 0) =
        libcint_input.env[xmvb::to_size(coordinate_offset)];
    (*atomic_coords)(atom_index, 1) =
        libcint_input.env[xmvb::to_size(coordinate_offset + 1)];
    (*atomic_coords)(atom_index, 2) =
        libcint_input.env[xmvb::to_size(coordinate_offset + 2)];
    (*atomic_charges)(atom_index) = libcint_input.atm[atom_offset + CHARGE_OF];
  }
}

double compute_one_electron_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& physical_ao_density_matrix,
    const vb::AoIntegralInput& ao_integral_input) {
  const int n_basis_functions = physical_ao_density_matrix.rows();
  const std::size_t matrix_size =
      xmvb::to_size(n_basis_functions) * n_basis_functions;
  if (physical_ao_density_matrix.cols() != n_basis_functions ||
      ao_integral_input.n_basis_functions != n_basis_functions ||
      ao_integral_input.ao_core_hamiltonian_matrix.size() != matrix_size) {
    throw std::invalid_argument("one-electron energy AO dimensions do not match");
  }
  const Eigen::Map<const Eigen::MatrixXd> core_hamiltonian(
      ao_integral_input.ao_core_hamiltonian_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  return physical_ao_density_matrix.cwiseProduct(core_hamiltonian).sum();
}

double compute_classical_coulomb_energy(
    const Eigen::Ref<const Eigen::MatrixXd>& physical_ao_density_matrix,
    const vb::AoIntegralInput& ao_integral_input) {
  const int n_basis_functions = physical_ao_density_matrix.rows();
  if (physical_ao_density_matrix.cols() != n_basis_functions ||
      ao_integral_input.n_basis_functions != n_basis_functions) {
    throw std::invalid_argument("Coulomb energy AO dimensions do not match");
  }
  if (ao_integral_input.ao_two_electron_integral_values.empty()) {
    throw std::runtime_error(
        "VB-PDFT Coulomb energy currently requires materialized AO two-electron integrals");
  }
  if (ao_integral_input.ao_two_electron_integral_indices.size() !=
      ao_integral_input.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("AO two-electron integral index/value sizes are inconsistent");
  }
  if (!ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty() &&
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.size() !=
          ao_integral_input.ao_two_electron_integral_values.size()) {
    throw std::invalid_argument("AO two-electron symmetry-shift cache size mismatch");
  }

  double coulomb_energy = 0.0;
  const double* density_data = physical_ao_density_matrix.data();
  const int* integral_indices = ao_integral_input.ao_two_electron_integral_indices.data();
  const std::uint8_t* symmetry_shifts =
      ao_integral_input.ao_two_electron_integral_symmetry_shifts.empty()
          ? nullptr
          : ao_integral_input.ao_two_electron_integral_symmetry_shifts.data();

  // The AO ERI storage keeps only one representative `(ij|kl)` per symmetric
  // pair-of-pairs class.  Multiplying by `0.5 ^ shift` recovers the
  // representative's weight inside the full unsymmetrized tensor, and the
  // remaining multiplicity for the classical Coulomb quadratic form is exactly
  // the factor `4`.
#pragma omp parallel for reduction(+ : coulomb_energy) schedule(static)
  for (std::ptrdiff_t integral_offset = 0;
       integral_offset <
           static_cast<std::ptrdiff_t>(ao_integral_input.ao_two_electron_integral_values.size());
       ++integral_offset) {
    const std::size_t integral_index = xmvb::to_size(integral_offset);
    const int i = integral_indices[integral_index * 4];
    const int j = integral_indices[integral_index * 4 + 1];
    const int k = integral_indices[integral_index * 4 + 2];
    const int l = integral_indices[integral_index * 4 + 3];
    if (i < 0 || i >= n_basis_functions ||
        j < 0 || j >= n_basis_functions ||
        k < 0 || k >= n_basis_functions ||
        l < 0 || l >= n_basis_functions) {
      throw std::invalid_argument("AO two-electron integral index is out of range");
    }
    double scaled_integral_value =
        ao_integral_input.ao_two_electron_integral_values[integral_index];
    if (symmetry_shifts != nullptr) {
      const std::uint8_t shift = symmetry_shifts[integral_index];
      if (shift >= 4) {
        throw std::invalid_argument("AO two-electron integral symmetry shift is out of range");
      }
      scaled_integral_value *= kIntegralSymmetryMultipliers[shift];
    } else {
      if (i == j) {
        scaled_integral_value *= 0.5;
      }
      if (k == l) {
        scaled_integral_value *= 0.5;
      }
      if (i == k && j == l) {
        scaled_integral_value *= 0.5;
      }
    }
    const double density_ij =
        density_data[xmvb::to_size(j) * n_basis_functions + i];
    const double density_kl =
        density_data[xmvb::to_size(l) * n_basis_functions + k];
    coulomb_energy += 4.0 * scaled_integral_value * density_ij * density_kl;
  }
  return coulomb_energy;
}

void validate_exact_density_context_consistency(
    const vb::SelectedStateExactPhysicalOneRdmResult& one_rdm_result,
    const vb::SelectedStateExactPhysicalOnTopPairDensityContext& on_top_context) {
  if (one_rdm_result.state_index != on_top_context.state_index) {
    throw std::runtime_error("VB-PDFT density builders returned different state indices");
  }
  if (one_rdm_result.n_basis_functions != on_top_context.n_basis_functions) {
    throw std::runtime_error("VB-PDFT density builders returned different AO dimensions");
  }
  if (std::abs(
          one_rdm_result.selected_state_overlap_normalization -
          on_top_context.selected_state_overlap_normalization) >
      kNormalizationConsistencyTolerance) {
    throw std::runtime_error(
        "VB-PDFT one-RDM and on-top contexts disagree on the selected-state normalization");
  }
}

}  // namespace

VbPdftEnergyEvaluator::VbPdftEnergyEvaluator(
    VbPdftConfig config,
    vb::VBSCFAlgorithm algorithm)
    : config_(std::move(config)),
      gradient_evaluator_(algorithm) {
  if (config_.functional_id <= 0) {
    throw std::invalid_argument("functional_id must be positive");
  }
  if (config_.density_threshold <= 0.0) {
    throw std::invalid_argument("density_threshold must be positive");
  }
}

VbPdftEnergyEvaluator::VbPdftEnergyEvaluator(
    VbPdftConfig config,
    vb::CppActiveSpaceGradientEvaluator gradient_evaluator)
    : config_(std::move(config)),
      gradient_evaluator_(std::move(gradient_evaluator)) {
  if (config_.functional_id <= 0) {
    throw std::invalid_argument("functional_id must be positive");
  }
  if (config_.density_threshold <= 0.0) {
    throw std::invalid_argument("density_threshold must be positive");
  }
}

VbPdftEnergyResult VbPdftEnergyEvaluator::evaluate(
    const vb::CppVbInput& input,
    int state_index,
    double nuclear_repulsion_energy) const {
  const double resolved_nuclear_repulsion_energy =
      std::isfinite(nuclear_repulsion_energy)
          ? nuclear_repulsion_energy
          : compute_nuclear_repulsion_energy_from_libcint_input(
                input.libcint_input);
  const vb::CppActiveSpaceGradientResult gradient_result =
      gradient_evaluator_.evaluate(
          input,
          std::vector<int>{state_index},
          std::vector<double>{1.0},
          resolved_nuclear_repulsion_energy);
  return evaluate_from_state_specific_gradient(
      input,
      gradient_result,
      nuclear_repulsion_energy);
}

VbPdftEnergyResult VbPdftEnergyEvaluator::evaluate_from_state_specific_gradient(
    const vb::CppVbInput& input,
    const vb::CppActiveSpaceGradientResult& gradient_result,
    double nuclear_repulsion_energy) const {
  validate_single_state_gradient_result(gradient_result);

  vb::SelectedStateExactPhysicalOneRdmBuilder one_rdm_builder(gradient_evaluator_);
  const vb::SelectedStateExactPhysicalOneRdmResult one_rdm_result =
      one_rdm_builder.build_from_state_specific_gradient(input, gradient_result);
  vb::SelectedStateExactPhysicalOnTopPairDensityBuilder on_top_builder(
      gradient_evaluator_);
  const vb::SelectedStateExactPhysicalOnTopPairDensityContext on_top_context =
      on_top_builder.build_from_state_specific_gradient(input, gradient_result);
  validate_exact_density_context_consistency(one_rdm_result, on_top_context);

  Eigen::MatrixXd atomic_coords;
  Eigen::VectorXd atomic_charges;
  extract_atomic_geometry_from_libcint_input(
      input.libcint_input,
      &atomic_coords,
      &atomic_charges);
  MolecularGridBuilder grid_builder(config_.grid_config);
  const MolecularGrid grid = grid_builder.build(atomic_coords, atomic_charges);

  LibcintAoGridEvaluator ao_grid_evaluator(input.libcint_input);
  const AoGridValues ao_values = ao_grid_evaluator.evaluate_values(grid.points);

  RealSpaceDensityBuilder density_builder;
  const RealSpaceDensities densities =
      density_builder.build(
          one_rdm_result.physical_ao_total_density_matrix,
          on_top_context,
          ao_values);

  LibxcFunctional functional(config_.functional_id);
  if (config_.use_gga || functional.is_gga()) {
    throw std::runtime_error(
        "VB-PDFT currently supports only LDA on-top functionals");
  }
  const TranslatedSpinDensity spin_density =
      compute_translated_spin_density(
          densities.rho,
          densities.pi,
          config_.density_threshold);
  const Eigen::VectorXd energy_density =
      functional.evaluate_energy_density(
          spin_density.rho_alpha,
          spin_density.rho_beta);

  VbPdftEnergyResult result;
  result.state_index = one_rdm_result.state_index;
  result.nuclear_repulsion_energy =
      resolve_nuclear_repulsion_energy(
          nuclear_repulsion_energy,
          gradient_result,
          input.libcint_input);
  result.n_grid_points = grid.n_points();
  result.on_top_energy =
      grid.weights.dot(densities.rho.cwiseProduct(energy_density));
  result.integrated_electron_count = grid.weights.dot(densities.rho);
  result.one_electron_energy =
      compute_one_electron_energy(
          one_rdm_result.physical_ao_total_density_matrix,
          input.ao_integral_input);
  result.coulomb_energy =
      compute_classical_coulomb_energy(
          one_rdm_result.physical_ao_total_density_matrix,
          input.ao_integral_input);
  result.total_energy =
      result.nuclear_repulsion_energy +
      result.one_electron_energy +
      result.coulomb_energy +
      result.on_top_energy;
  return result;
}

}  // namespace xmvb::vb::pdft
