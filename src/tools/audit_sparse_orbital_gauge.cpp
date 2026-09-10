#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "runtime/vbscf_input_loader.hpp"
#include "vbscf/orbitals/charts/support_layout_adapter.hpp"
#include "vbscf/orbitals/charts/orbital_chart.hpp"
#include "vbscf/diagnostics/orbital_chart_audit.hpp"
#include "vbscf/orbitals/charts/sparse_parameter_layout.hpp"

namespace {

bool parse_bool(const std::string& value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument("invalid Boolean value: " + value);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 4) {
    std::cerr
        << "usage: audit_orbital_chart <input.xmi> "
        << "[--nonredundant-adapt true|false]\n";
    return EXIT_FAILURE;
  }

  try {
    bool nonredundant_adapt = true;
    if (argc == 4) {
      if (std::string(argv[2]) != "--nonredundant-adapt") {
        throw std::invalid_argument("unknown option: " + std::string(argv[2]));
      }
      nonredundant_adapt = parse_bool(argv[3]);
    }

    xmvb::vb::VbScfInputLoadOptions load_options;
    load_options.standard_two_electron_mode =
        xmvb::vb::StandardTwoElectronMode::Exact;
    const auto loaded =
        xmvb::vb::load_vbscf_input_with_timings(argv[1], load_options);
    const xmvb::vb::VbScfInput input = nonredundant_adapt
        ? xmvb::vb::build_nonredundant_optimizer_input(loaded.input)
        : loaded.input;
    const xmvb::vb::SparseParameterLayout parameter_view(
        input.orbital_preparation_input);
    const auto& orbital = input.orbital_preparation_input;
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(
        orbital.n_basis_functions, orbital.n_orbitals);
    for (int p = 0; p < orbital.n_orbitals; ++p) {
      for (int j = 0; j < xmvb::vb::stored_sparse_orbital_coefficient_count(orbital, p); ++j) {
        const int slot = p * orbital.n_basis_functions + j;
        coefficients(orbital.orbital_basis_index_table[slot] - 1, p) =
            orbital.orbital_value_table[slot];
      }
    }
    const int occupied_count =
        (orbital.n_total_electrons - orbital.n_active_electrons) / 2 +
        orbital.n_active_orbitals;
    const xmvb::vb::OrbitalChart space(
        orbital, parameter_view, coefficients.leftCols(occupied_count),
        coefficients, nullptr, true);
    Eigen::MatrixXd basis(parameter_view.size(), space.reduced_size());
    for (int j = 0; j < basis.cols(); ++j) {
      basis.col(j) = space.expand_step(Eigen::VectorXd::Unit(basis.cols(), j));
    }
    const auto audit = xmvb::vb::audit_orbital_chart(
        orbital, parameter_view, &basis);

    std::cout << std::setprecision(12);
    std::cout << "input = " << argv[1] << '\n';
    std::cout << "nonredundant_adapt = "
              << (nonredundant_adapt ? "true" : "false") << '\n';
    std::cout << "packed_dimension = " << audit.packed_dimension << '\n';
    std::cout << "gauge_parameter_dimension = "
              << audit.gauge_parameter_dimension << '\n';
    std::cout << "support_constraint_rank = "
              << audit.support_constraint_rank << '\n';
    std::cout << "admissible_gauge_parameter_dimension = "
              << audit.admissible_gauge_parameter_dimension << '\n';
    std::cout << "gauge_rank = " << audit.gauge_rank << '\n';
    std::cout << "quotient_dimension = " << audit.quotient_dimension << '\n';
    std::cout << "physical_jacobian_rank = "
              << audit.physical_jacobian_rank << '\n';
    std::cout << "physical_jacobian_nullity = "
              << audit.physical_jacobian_nullity << '\n';
    std::cout << "unmapped_parameter_count = "
              << audit.unmapped_parameter_count << '\n';
    std::cout << "relative_gauge_annihilation_residual = "
              << audit.relative_gauge_annihilation_residual << '\n';
    std::cout << "maximum_principal_angle_sine = "
              << audit.maximum_gauge_kernel_principal_angle_sine << '\n';
    const double gauge_overlap =
        (audit.packed_gauge_basis.transpose() * basis).norm();
    std::cout << "current_reduced_dimension = " << audit.current_reduced_dimension << '\n';
    std::cout << "current_retained_gauge_dimension = "
              << audit.current_retained_gauge_dimension << '\n';
    std::cout << "current_missing_physical_dimension = "
              << audit.current_missing_physical_dimension << '\n';
    std::cout << "current_basis_gauge_overlap = " << gauge_overlap << '\n';
    const double tolerance = 100 * std::numeric_limits<double>::epsilon() *
        std::max(1, audit.packed_dimension);
    if (audit.unmapped_parameter_count != 0 ||
        audit.gauge_rank != audit.physical_jacobian_nullity ||
        audit.current_retained_gauge_dimension != 0 ||
        audit.current_missing_physical_dimension != 0 ||
        gauge_overlap > tolerance ||
        audit.relative_gauge_annihilation_residual > tolerance) {
      throw std::runtime_error("quotient audit did not pass");
    }
  } catch (const std::exception& exception) {
    std::cerr << "audit_orbital_chart failed: "
              << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
#include <algorithm>
