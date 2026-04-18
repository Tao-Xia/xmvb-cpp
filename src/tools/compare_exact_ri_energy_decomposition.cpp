#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "core/linear_algebra/generalized_eigensolver.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/cpp_vb_input_ri_cache.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/orbital/active_space_one_electron_builder.hpp"
#include "vb/orbital/active_space_orbital_preparer.hpp"
#include "vb/orbital/active_space_two_electron_builder.hpp"
#include "vb/orbital/ao_effective_one_electron_builder.hpp"
#include "vb/orbital/ri_active_space_two_electron_builder.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Matrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct EnergyBreakdown {
  double one_electron_reference_energy = 0.0;
  double active_eigenvalue = 0.0;
  double total_energy = 0.0;
  double active_h1e_inf_norm = 0.0;
  double active_eri_inf_norm = 0.0;
  double g11_diag_energy = 0.0;
  double g11_offdiag_energy = 0.0;
  double g11_diag_inf_norm = 0.0;
  double g11_offdiag_inf_norm = 0.0;
};

double evaluate_active_eigenvalue(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::OrbitalPreparationResult& orbital_result,
    const std::vector<double>& active_h1e,
    const std::vector<double>& packed_active_eri) {
  xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder structure_builder(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto structure_matrices = structure_builder.build(
      input.structure_data.alpha_det,
      input.structure_data.beta_det,
      input.structure_data.determinant_to_structure_terms,
      orbital_result.active_orbital_overlap_matrix,
      active_h1e,
      input.orbital_preparation_input.n_active_orbitals,
      packed_active_eri,
      input.structure_data.n_structures);

  xmvb::core::GeneralizedEigensolver generalized_eigensolver;
  const auto eigen_result = generalized_eigensolver.solve(
      structure_matrices.hamiltonian_matrix,
      structure_matrices.overlap_matrix,
      input.structure_data.n_structures);
  return eigen_result.eigenvalues.front();
}

double compute_inf_norm(const std::vector<double>& values) {
  double inf_norm = 0.0;
  for (const double value : values) {
    inf_norm = std::max(inf_norm, std::abs(value));
  }
  return inf_norm;
}

double compute_one_electron_reference_energy(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& ao_effective_h1e,
    const std::vector<double>& ao_core_hamiltonian_matrix,
    int n_basis_functions) {
  double one_electron_reference_energy = 0.0;
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < n_basis_functions; ++row) {
      const std::size_t index =
          xmvb::to_size(column) * n_basis_functions + row;
      one_electron_reference_energy +=
          inactive_density_matrix[index] *
          (ao_effective_h1e[index] + ao_core_hamiltonian_matrix[index]);
    }
  }
  return one_electron_reference_energy;
}

void accumulate_g11_statistics(
    const std::vector<double>& inactive_density_matrix,
    const std::vector<double>& g11_matrix,
    int n_basis_functions,
    EnergyBreakdown* breakdown) {
  if (breakdown == nullptr) {
    throw std::invalid_argument("breakdown must not be null");
  }
  for (int column = 0; column < n_basis_functions; ++column) {
    for (int row = 0; row < n_basis_functions; ++row) {
      const std::size_t index =
          xmvb::to_size(column) * n_basis_functions + row;
      const double contribution =
          inactive_density_matrix[index] * g11_matrix[index];
      if (row == column) {
        breakdown->g11_diag_energy += contribution;
        breakdown->g11_diag_inf_norm =
            std::max(breakdown->g11_diag_inf_norm, std::abs(g11_matrix[index]));
      } else {
        breakdown->g11_offdiag_energy += contribution;
        breakdown->g11_offdiag_inf_norm =
            std::max(breakdown->g11_offdiag_inf_norm, std::abs(g11_matrix[index]));
      }
    }
  }
}

EnergyBreakdown build_exact_breakdown(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::OrbitalPreparationResult& orbital_result,
    double nuclear_repulsion_energy) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  xmvb::vb::AoEffectiveOneElectronBuilder ao_builder;
  const auto ao_result = ao_builder.build(
      orbital_result.inactive_density_matrix,
      input.ao_integral_input);

  xmvb::vb::ActiveSpaceOneElectronBuilder active_h1e_builder;
  const auto active_h1e_result = active_h1e_builder.build(
      ao_result.ao_effective_h1e,
      orbital_result.auxiliary_orbital_matrix,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);

  xmvb::vb::ActiveSpaceTwoElectronBuilder exact_builder;
  const auto exact_eri_result = exact_builder.build(
      input.ao_integral_input,
      orbital_result,
      n_active_orbitals);

  EnergyBreakdown breakdown;
  breakdown.one_electron_reference_energy =
      compute_one_electron_reference_energy(
          orbital_result.inactive_density_matrix,
          ao_result.ao_effective_h1e,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          n_basis_functions);
  breakdown.active_eigenvalue = evaluate_active_eigenvalue(
      input,
      orbital_result,
      active_h1e_result.h1e_act,
      exact_eri_result.packed_active_two_electron_integrals);
  breakdown.total_energy =
      breakdown.one_electron_reference_energy +
      breakdown.active_eigenvalue +
      nuclear_repulsion_energy;
  breakdown.active_h1e_inf_norm = compute_inf_norm(active_h1e_result.h1e_act);
  breakdown.active_eri_inf_norm =
      compute_inf_norm(exact_eri_result.packed_active_two_electron_integrals);
  accumulate_g11_statistics(
      orbital_result.inactive_density_matrix,
      ao_result.ao_coulomb_exchange_matrix,
      n_basis_functions,
      &breakdown);
  return breakdown;
}

EnergyBreakdown build_ri_breakdown(
    const xmvb::vb::CppVbInput& input,
    const xmvb::vb::OrbitalPreparationResult& orbital_result,
    double nuclear_repulsion_energy) {
  const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
  const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
  const int n_inactive_doubly_occupied_orbitals =
      (input.orbital_preparation_input.n_total_electrons -
       input.orbital_preparation_input.n_active_electrons) / 2;

  const auto& ri_cache = xmvb::vb::ensure_cpp_vb_input_ri_cache(input);

  xmvb::vb::AoEffectiveOneElectronBuilder ao_builder;
  xmvb::vb::AoEffectiveOneElectronResult ao_result;
  if (!input.ao_integral_input.ao_two_electron_integral_values.empty()) {
    // Match the production objective: even under `INT=RI`, keep the AO-side
    // inactive/reference contraction on the validated exact four-center path
    // whenever materialized AO integrals are available. Only the active-space
    // two-electron representation should switch to RI.
    ao_result = ao_builder.build(
        orbital_result.inactive_density_matrix,
        input.ao_integral_input);
  } else {
    ao_result = ao_builder.build(
        orbital_result.inactive_density_matrix,
        input.ao_integral_input.ao_core_hamiltonian_matrix,
        ri_cache,
        n_basis_functions);
  }

  xmvb::vb::ActiveSpaceOneElectronBuilder active_h1e_builder;
  const auto active_h1e_result = active_h1e_builder.build(
      ao_result.ao_effective_h1e,
      orbital_result.auxiliary_orbital_matrix,
      n_basis_functions,
      n_inactive_doubly_occupied_orbitals,
      n_active_orbitals);

  xmvb::vb::RiActiveSpaceTwoElectronBuilder ri_builder;
  const auto ri_eri_result = ri_builder.build(
      ri_cache,
      orbital_result,
      n_basis_functions,
      n_active_orbitals,
      {.reconstruct_packed_integrals = true});

  EnergyBreakdown breakdown;
  breakdown.one_electron_reference_energy =
      compute_one_electron_reference_energy(
          orbital_result.inactive_density_matrix,
          ao_result.ao_effective_h1e,
          input.ao_integral_input.ao_core_hamiltonian_matrix,
          n_basis_functions);
  breakdown.active_eigenvalue = evaluate_active_eigenvalue(
      input,
      orbital_result,
      active_h1e_result.h1e_act,
      ri_eri_result.packed_active_two_electron_integrals);
  breakdown.total_energy =
      breakdown.one_electron_reference_energy +
      breakdown.active_eigenvalue +
      nuclear_repulsion_energy;
  breakdown.active_h1e_inf_norm = compute_inf_norm(active_h1e_result.h1e_act);
  breakdown.active_eri_inf_norm =
      compute_inf_norm(ri_eri_result.packed_active_two_electron_integrals);
  accumulate_g11_statistics(
      orbital_result.inactive_density_matrix,
      ao_result.ao_coulomb_exchange_matrix,
      n_basis_functions,
      &breakdown);
  return breakdown;
}

void print_breakdown(
    const std::string& label,
    const EnergyBreakdown& breakdown) {
  std::cout << label << "_one_electron_reference_energy = "
            << breakdown.one_electron_reference_energy << '\n';
  std::cout << label << "_active_eigenvalue = "
            << breakdown.active_eigenvalue << '\n';
  std::cout << label << "_total_energy = "
            << breakdown.total_energy << '\n';
  std::cout << label << "_active_h1e_inf_norm = "
            << breakdown.active_h1e_inf_norm << '\n';
  std::cout << label << "_active_eri_inf_norm = "
            << breakdown.active_eri_inf_norm << '\n';
  std::cout << label << "_g11_diag_energy = "
            << breakdown.g11_diag_energy << '\n';
  std::cout << label << "_g11_offdiag_energy = "
            << breakdown.g11_offdiag_energy << '\n';
  std::cout << label << "_g11_diag_inf_norm = "
            << breakdown.g11_diag_inf_norm << '\n';
  std::cout << label << "_g11_offdiag_inf_norm = "
            << breakdown.g11_offdiag_inf_norm << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: compare_exact_ri_energy_decomposition <input.xmi>");
    }

    xmvb::vb::CppVbInputLoadOptions load_options;
    load_options.ao_integral_source =
        xmvb::vb::AoIntegralSource::LibcintMaterializedCpp;
    load_options.orbital_guess_source =
        xmvb::vb::OrbitalGuessSource::Cpp;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Auto;
    const auto load_result =
        xmvb::vb::load_cpp_vb_input_with_timings(argv[1], load_options);
    const auto& input = load_result.input;

    xmvb::vb::ActiveSpaceOrbitalPreparer orbital_preparer;
    const auto orbital_result =
        orbital_preparer.prepare(input.orbital_preparation_input);

    const EnergyBreakdown exact_breakdown = build_exact_breakdown(
        input,
        orbital_result,
        load_result.nuclear_repulsion_energy);
    const EnergyBreakdown ri_breakdown = build_ri_breakdown(
        input,
        orbital_result,
        load_result.nuclear_repulsion_energy);

    const int n_basis_functions = input.orbital_preparation_input.n_basis_functions;
    const int n_active_orbitals = input.orbital_preparation_input.n_active_orbitals;
    const int n_inactive_doubly_occupied_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) / 2;

    xmvb::vb::AoEffectiveOneElectronBuilder ao_builder;
    const auto exact_ao_result = ao_builder.build(
        orbital_result.inactive_density_matrix,
        input.ao_integral_input);
    const auto ri_ao_result = ao_builder.build(
        orbital_result.inactive_density_matrix,
        input.ao_integral_input.ao_core_hamiltonian_matrix.vector(),
        xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
        n_basis_functions);
    xmvb::vb::ActiveSpaceOneElectronBuilder active_h1e_builder;
    const auto exact_active_h1e = active_h1e_builder.build(
        exact_ao_result.ao_effective_h1e,
        orbital_result.auxiliary_orbital_matrix,
        n_basis_functions,
        n_inactive_doubly_occupied_orbitals,
        n_active_orbitals);
    const auto ri_active_h1e = active_h1e_builder.build(
        ri_ao_result.ao_effective_h1e,
        orbital_result.auxiliary_orbital_matrix,
        n_basis_functions,
        n_inactive_doubly_occupied_orbitals,
        n_active_orbitals);
    xmvb::vb::ActiveSpaceTwoElectronBuilder exact_eri_builder;
    const auto exact_eri = exact_eri_builder.build(
        input.ao_integral_input,
        orbital_result,
        n_active_orbitals);
    xmvb::vb::RiActiveSpaceTwoElectronBuilder ri_eri_builder;
    const auto ri_eri = ri_eri_builder.build(
        xmvb::vb::ensure_cpp_vb_input_ri_cache(input),
        orbital_result,
        n_basis_functions,
        n_active_orbitals,
        {.reconstruct_packed_integrals = true});

    const double active_eigen_exact_h1e_exact_eri = evaluate_active_eigenvalue(
        input,
        orbital_result,
        exact_active_h1e.h1e_act,
        exact_eri.packed_active_two_electron_integrals);
    const double active_eigen_exact_h1e_ri_eri = evaluate_active_eigenvalue(
        input,
        orbital_result,
        exact_active_h1e.h1e_act,
        ri_eri.packed_active_two_electron_integrals);
    const double active_eigen_ri_h1e_exact_eri = evaluate_active_eigenvalue(
        input,
        orbital_result,
        ri_active_h1e.h1e_act,
        exact_eri.packed_active_two_electron_integrals);
    const double active_eigen_ri_h1e_ri_eri = evaluate_active_eigenvalue(
        input,
        orbital_result,
        ri_active_h1e.h1e_act,
        ri_eri.packed_active_two_electron_integrals);

    std::cout << std::setprecision(15);
    std::cout << "nuclear_repulsion_energy = "
              << load_result.nuclear_repulsion_energy << '\n';
    print_breakdown("exact", exact_breakdown);
    print_breakdown("ri", ri_breakdown);
    std::cout << "diff_one_electron_reference_energy = "
              << (ri_breakdown.one_electron_reference_energy -
                  exact_breakdown.one_electron_reference_energy)
              << '\n';
    std::cout << "diff_active_eigenvalue = "
              << (ri_breakdown.active_eigenvalue -
                  exact_breakdown.active_eigenvalue)
              << '\n';
    std::cout << "diff_total_energy = "
              << (ri_breakdown.total_energy - exact_breakdown.total_energy)
              << '\n';
    std::cout << "active_eigen_exact_h1e_exact_eri = "
              << active_eigen_exact_h1e_exact_eri << '\n';
    std::cout << "active_eigen_exact_h1e_ri_eri = "
              << active_eigen_exact_h1e_ri_eri << '\n';
    std::cout << "active_eigen_ri_h1e_exact_eri = "
              << active_eigen_ri_h1e_exact_eri << '\n';
    std::cout << "active_eigen_ri_h1e_ri_eri = "
              << active_eigen_ri_h1e_ri_eri << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
