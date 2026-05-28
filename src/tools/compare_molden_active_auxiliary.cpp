#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

#include "runtime/cpp_vb_input_loader.hpp"
#include "runtime/libcint_compat.hpp"
#include "runtime/legacy_shell_utils.hpp"

namespace {

using xmvb::vb::CppVbInput;
using xmvb::vb::CppVbInputLoadOptions;
using xmvb::vb::load_cpp_vb_input_with_timings;

struct MoldenOrbitals {
  Eigen::MatrixXd coefficients_in_molden_order;
  int n_orbitals = 0;
};

void print_usage() {
  std::cerr
      << "usage: compare_molden_active_auxiliary <input.xmi> <current.molden> <reference.molden>\n";
}

std::string trim_ascii_whitespace(const std::string& value) {
  std::size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first])) != 0) {
    ++first;
  }
  std::size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
    --last;
  }
  return value.substr(first, last - first);
}

MoldenOrbitals parse_molden_file(
    const std::string& path,
    int n_basis_functions) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open Molden file: " + path);
  }

  // Each `Sym=` header starts one dense AO column in Molden order. The
  // diagnostic only needs those AO coefficients, so it ignores the remaining
  // Molden metadata lines.
  std::vector<Eigen::VectorXd> orbitals;
  std::string line;
  bool inside_mo_section = false;
  bool have_current_orbital = false;
  int coefficient_count = 0;
  Eigen::VectorXd current_orbital = Eigen::VectorXd::Zero(n_basis_functions);

  while (std::getline(input, line)) {
    const std::string trimmed = trim_ascii_whitespace(line);
    if (trimmed == "[MO]") {
      inside_mo_section = true;
      continue;
    }
    if (!inside_mo_section) {
      continue;
    }
    if (trimmed.rfind("Sym=", 0) == 0) {
      if (have_current_orbital) {
        if (coefficient_count != n_basis_functions) {
          throw std::runtime_error(
              "incomplete Molden orbital in " + path +
              ": expected " + std::to_string(n_basis_functions) +
              " coefficients, got " + std::to_string(coefficient_count));
        }
        orbitals.push_back(current_orbital);
      }
      current_orbital = Eigen::VectorXd::Zero(n_basis_functions);
      have_current_orbital = true;
      coefficient_count = 0;
      continue;
    }
    if (!have_current_orbital || trimmed.empty()) {
      continue;
    }

    std::istringstream line_stream(trimmed);
    int one_based_index = 0;
    double value = 0.0;
    if (!(line_stream >> one_based_index >> value)) {
      continue;
    }
    if (one_based_index <= 0 || one_based_index > n_basis_functions) {
      throw std::runtime_error(
          "Molden AO index is out of range in " + path);
    }
    current_orbital(one_based_index - 1) = value;
    ++coefficient_count;
  }

  if (have_current_orbital) {
    if (coefficient_count != n_basis_functions) {
      throw std::runtime_error(
          "incomplete final Molden orbital in " + path +
          ": expected " + std::to_string(n_basis_functions) +
          " coefficients, got " + std::to_string(coefficient_count));
    }
    orbitals.push_back(current_orbital);
  }
  if (orbitals.empty()) {
    throw std::runtime_error("no [MO] orbitals found in Molden file: " + path);
  }

  Eigen::MatrixXd coefficient_matrix(n_basis_functions, orbitals.size());
  for (Eigen::Index orbital_index = 0;
       orbital_index < static_cast<Eigen::Index>(orbitals.size());
       ++orbital_index) {
    coefficient_matrix.col(orbital_index) = orbitals[orbital_index];
  }

  MoldenOrbitals result;
  result.coefficients_in_molden_order = std::move(coefficient_matrix);
  result.n_orbitals = static_cast<int>(orbitals.size());
  return result;
}

std::vector<int> build_molden_to_internal_ao_permutation(
    const xmvb::vb::LibcintInput& libcint_input,
    int n_basis_functions) {
  constexpr std::array<int, 6> kDOrder = {0, 3, 5, 1, 2, 4};
  constexpr std::array<int, 10> kFOrder = {0, 6, 9, 3, 1, 2, 5, 8, 7, 4};

  std::vector<int> molden_to_internal;
  molden_to_internal.reserve(n_basis_functions);

  for (int shell_index = 0; shell_index < libcint_input.n_shells; ++shell_index) {
    const std::size_t shell_offset =
        shell_index * BAS_SLOTS;
    const int angular_momentum = libcint_input.bas[shell_offset + ANG_OF];
    const int ao_offset =
        libcint_input.basidx[shell_index * 2];
    const int ao_count =
        libcint_input.basidx[shell_index * 2 + 1];
    if (ao_offset < 0 || ao_count != xmvb::vb::cartesian_ao_count(angular_momentum) ||
        ao_offset + ao_count > n_basis_functions) {
      throw std::runtime_error(
          "Libcint shell AO layout is inconsistent while building Molden permutation");
    }

    if (angular_momentum == 2) {
      for (const int local_index : kDOrder) {
        molden_to_internal.push_back(ao_offset + local_index);
      }
      continue;
    }
    if (angular_momentum == 3) {
      for (const int local_index : kFOrder) {
        molden_to_internal.push_back(ao_offset + local_index);
      }
      continue;
    }
    for (int local_index = 0; local_index < ao_count; ++local_index) {
      molden_to_internal.push_back(ao_offset + local_index);
    }
  }

  if (molden_to_internal.size() != n_basis_functions) {
    throw std::runtime_error("Molden/internal AO permutation size mismatch");
  }
  return molden_to_internal;
}

Eigen::MatrixXd build_molden_order_overlap_matrix(
    const CppVbInput& input) {
  const int n_basis_functions =
      input.orbital_preparation_input.n_basis_functions;
  const Eigen::Map<const Eigen::MatrixXd> basis_overlap_internal(
      input.orbital_preparation_input.ao_overlap_matrix.data(),
      n_basis_functions,
      n_basis_functions);
  const std::vector<int> molden_to_internal =
      build_molden_to_internal_ao_permutation(
          input.libcint_input,
          n_basis_functions);

  // The Molden parser returns coefficient columns in Molden AO order. Reorder
  // the internal AO overlap matrix into that same basis so the remaining
  // orbital algebra can be carried out directly on the parsed columns.
  Eigen::MatrixXd basis_overlap_molden =
      Eigen::MatrixXd::Zero(n_basis_functions, n_basis_functions);
  for (int molden_column = 0; molden_column < n_basis_functions; ++molden_column) {
    const int internal_column =
        molden_to_internal[molden_column];
    for (int molden_row = 0; molden_row < n_basis_functions; ++molden_row) {
      const int internal_row =
          molden_to_internal[molden_row];
      basis_overlap_molden(molden_row, molden_column) =
          basis_overlap_internal(internal_row, internal_column);
    }
  }
  return basis_overlap_molden;
}

double s_metric_cosine(
    const Eigen::Ref<const Eigen::VectorXd>& left,
    const Eigen::Ref<const Eigen::VectorXd>& right,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix) {
  const double left_norm = left.transpose() * overlap_matrix * left;
  const double right_norm = right.transpose() * overlap_matrix * right;
  if (!(left_norm > 0.0) || !(right_norm > 0.0)) {
    throw std::runtime_error("encountered non-positive S-metric norm while comparing orbitals");
  }
  const double overlap = left.transpose() * overlap_matrix * right;
  return std::abs(overlap) / std::sqrt(left_norm * right_norm);
}

Eigen::MatrixXd build_spd_inverse_square_root(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    const char* label) {
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument(std::string(label) + " must be square");
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_solver(matrix);
  if (eigen_solver.info() != Eigen::Success) {
    throw std::runtime_error(std::string("failed eigendecomposition for ") + label);
  }
  Eigen::VectorXd inverse_sqrt_eigenvalues(matrix.rows());
  for (Eigen::Index index = 0; index < matrix.rows(); ++index) {
    const double eigenvalue = eigen_solver.eigenvalues()[index];
    if (!(eigenvalue > 0.0)) {
      throw std::runtime_error(std::string(label) + " is not positive definite");
    }
    inverse_sqrt_eigenvalues[index] = 1.0 / std::sqrt(eigenvalue);
  }
  return eigen_solver.eigenvectors() *
      inverse_sqrt_eigenvalues.asDiagonal() *
      eigen_solver.eigenvectors().transpose();
}

Eigen::MatrixXd build_inactive_projector(
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix) {
  if (inactive_orbitals.cols() == 0) {
    return Eigen::MatrixXd::Zero(inactive_orbitals.rows(), inactive_orbitals.rows());
  }

  // `OrbPrep6` and the C++ orbital preparer both use
  // `P_i = C_i (C_i^T S C_i)^{-1} C_i^T` as the inactive occupied projector.
  const Eigen::MatrixXd inactive_metric =
      inactive_orbitals.transpose() * overlap_matrix * inactive_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(inactive_metric);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor inactive occupied overlap");
  }
  const Eigen::MatrixXd inactive_metric_inverse =
      ldlt.solve(
          Eigen::MatrixXd::Identity(
              inactive_metric.rows(),
              inactive_metric.cols()));
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("failed to invert inactive occupied overlap");
  }
  return inactive_orbitals * inactive_metric_inverse * inactive_orbitals.transpose();
}

Eigen::MatrixXd build_active_auxiliary_orbitals(
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    int n_inactive_orbitals,
    int n_active_orbitals) {
  const Eigen::MatrixXd inactive_orbitals =
      occupied_orbitals.leftCols(n_inactive_orbitals);
  const Eigen::MatrixXd active_orbitals =
      occupied_orbitals.middleCols(n_inactive_orbitals, n_active_orbitals);
  const Eigen::MatrixXd inactive_projector =
      build_inactive_projector(inactive_orbitals, overlap_matrix);
  const Eigen::MatrixXd occupied_space_projector =
      Eigen::MatrixXd::Identity(
          occupied_orbitals.rows(),
          occupied_orbitals.rows()) -
      inactive_projector * overlap_matrix;
  return occupied_space_projector * active_orbitals;
}

Eigen::MatrixXd build_inactive_dual_overlap_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    int n_inactive_orbitals,
    int n_active_orbitals) {
  if (n_inactive_orbitals == 0 || n_active_orbitals == 0) {
    return Eigen::MatrixXd::Zero(n_inactive_orbitals, n_active_orbitals);
  }

  // `A2 = C_i^T S C_a` in the inactive-dual metric measures how much of each
  // original active orbital lies in the inactive occupied span before the
  // projector `T_a = (I - P_i S) C_a` removes that component.
  const Eigen::MatrixXd inactive_orbitals =
      occupied_orbitals.leftCols(n_inactive_orbitals);
  const Eigen::MatrixXd active_orbitals =
      occupied_orbitals.middleCols(n_inactive_orbitals, n_active_orbitals);
  const Eigen::MatrixXd inactive_metric =
      inactive_orbitals.transpose() * overlap_matrix * inactive_orbitals;
  Eigen::LDLT<Eigen::MatrixXd> ldlt(inactive_metric);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("failed to factor inactive metric for dual-overlap coefficients");
  }
  const Eigen::MatrixXd inactive_metric_inverse =
      ldlt.solve(Eigen::MatrixXd::Identity(
          inactive_metric.rows(),
          inactive_metric.cols()));
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("failed to invert inactive metric for dual-overlap coefficients");
  }
  return inactive_metric_inverse *
      inactive_orbitals.transpose() *
      overlap_matrix *
      active_orbitals;
}

void print_inactive_dual_overlap_summary(
    const char* label,
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    int n_inactive_orbitals,
    int n_active_orbitals,
    int start_index_one_based) {
  if (n_inactive_orbitals == 0 || n_active_orbitals == 0) {
    return;
  }

  const Eigen::MatrixXd inactive_dual_overlap_coefficients =
      build_inactive_dual_overlap_coefficients(
          occupied_orbitals,
          overlap_matrix,
          n_inactive_orbitals,
          n_active_orbitals);
  std::cout << label << '\n';
  for (int active_index = 0; active_index < n_active_orbitals; ++active_index) {
    const double coefficient_l2_norm =
        inactive_dual_overlap_coefficients.col(active_index).norm();
    std::cout << "  orbital " << std::setw(2)
              << start_index_one_based + active_index
              << "  inactive_dual_l2 = " << coefficient_l2_norm << '\n';
  }
}

Eigen::VectorXd build_s_metric_inactive_repaired_orbital(
    const Eigen::Ref<const Eigen::VectorXd>& current_orbital,
    const Eigen::Ref<const Eigen::VectorXd>& reference_orbital,
    const Eigen::Ref<const Eigen::MatrixXd>& inactive_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix) {
  if (inactive_orbitals.cols() == 0) {
    return current_orbital;
  }
  const Eigen::MatrixXd inactive_metric =
      inactive_orbitals.transpose() * overlap_matrix * inactive_orbitals;
  const Eigen::VectorXd rhs =
      inactive_orbitals.transpose() *
      overlap_matrix *
      (reference_orbital - current_orbital);
  Eigen::LDLT<Eigen::MatrixXd> ldlt(inactive_metric);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to factor inactive metric for representative repair");
  }
  const Eigen::VectorXd coefficients = ldlt.solve(rhs);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error(
        "failed to solve inactive representative repair system");
  }
  return current_orbital + inactive_orbitals * coefficients;
}

void print_orbital_cosines(
    const char* label,
    const Eigen::Ref<const Eigen::MatrixXd>& current_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_orbitals,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    int start_index_one_based) {
  std::cout << label << '\n';
  for (int orbital_index = 0; orbital_index < current_orbitals.cols(); ++orbital_index) {
    const double cosine =
        s_metric_cosine(
            current_orbitals.col(orbital_index),
            reference_orbitals.col(orbital_index),
            overlap_matrix);
    std::cout << "  orbital " << std::setw(2)
              << start_index_one_based + orbital_index
              << "  s_cos = " << std::setprecision(12) << cosine << '\n';
  }
}

void print_active_subspace_singular_values(
    const Eigen::Ref<const Eigen::MatrixXd>& current_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& reference_active_auxiliary,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_matrix,
    int start_index_one_based) {
  if (current_active_auxiliary.cols() == 0) {
    return;
  }

  // The active auxiliaries are generally not orthonormal among themselves, so
  // compare the subspaces through their S-orthonormal representatives. If
  // these singular values are all near one, the remaining discrepancy is a
  // basis choice inside the same active subspace rather than a different
  // physical active span.
  const Eigen::MatrixXd current_metric =
      current_active_auxiliary.transpose() *
      overlap_matrix *
      current_active_auxiliary;
  const Eigen::MatrixXd reference_metric =
      reference_active_auxiliary.transpose() *
      overlap_matrix *
      reference_active_auxiliary;
  const Eigen::MatrixXd current_orthonormal =
      current_active_auxiliary *
      build_spd_inverse_square_root(
          current_metric,
          "current_active_auxiliary_metric");
  const Eigen::MatrixXd reference_orthonormal =
      reference_active_auxiliary *
      build_spd_inverse_square_root(
          reference_metric,
          "reference_active_auxiliary_metric");
  const Eigen::MatrixXd subspace_overlap =
      current_orthonormal.transpose() *
      overlap_matrix *
      reference_orthonormal;
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      subspace_overlap,
      Eigen::ComputeThinU | Eigen::ComputeThinV);
  std::cout << "active_auxiliary_subspace_singular_values";
  for (Eigen::Index index = 0; index < svd.singularValues().size(); ++index) {
    std::cout << ' ' << svd.singularValues()[index];
  }
  std::cout << '\n';

  // Rotate the current active auxiliaries within their own subspace so they
  // best match the reference active-auxiliary basis. If these repaired column
  // cosines become near-perfect, the remaining mismatch is a canonicalization
  // issue inside the same active span rather than a different physical space.
  const Eigen::MatrixXd subspace_rotation =
      svd.matrixU() * svd.matrixV().transpose();
  const Eigen::MatrixXd current_to_reference_basis =
      build_spd_inverse_square_root(
          current_metric,
          "current_active_auxiliary_metric") *
      subspace_rotation *
      build_spd_inverse_square_root(
          reference_metric,
          "reference_active_auxiliary_metric")
          .inverse();
  const Eigen::MatrixXd rotated_current_active_auxiliary =
      current_active_auxiliary * current_to_reference_basis;
  std::cout << "active_auxiliary_with_best_active_subspace_rotation\n";
  for (int active_index = 0;
       active_index < rotated_current_active_auxiliary.cols();
       ++active_index) {
    const double cosine =
        s_metric_cosine(
            rotated_current_active_auxiliary.col(active_index),
            reference_active_auxiliary.col(active_index),
            overlap_matrix);
    std::cout << "  orbital " << std::setw(2)
              << start_index_one_based + active_index
              << "  s_cos = " << cosine << '\n';
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      print_usage();
      return 1;
    }

    const std::string input_path = argv[1];
    const std::string current_molden_path = argv[2];
    const std::string reference_molden_path = argv[3];

    CppVbInputLoadOptions load_options;
    load_options.skip_orbital_guess = true;
    load_options.ao_integral_source =
        xmvb::vb::AoIntegralSource::RuntimeCoreHamiltonianOnly;
    const auto load_result =
        load_cpp_vb_input_with_timings(input_path, load_options);
    const CppVbInput& input = load_result.input;

    const int n_basis_functions =
        input.orbital_preparation_input.n_basis_functions;
    const int n_orbitals =
        input.orbital_preparation_input.n_orbitals;
    const int n_active_orbitals =
        input.orbital_preparation_input.n_active_orbitals;
    const int n_inactive_orbitals =
        (input.orbital_preparation_input.n_total_electrons -
         input.orbital_preparation_input.n_active_electrons) / 2;
    const int n_occupied_orbitals =
        n_inactive_orbitals + n_active_orbitals;
    if (n_inactive_orbitals < 0 ||
        n_occupied_orbitals > n_orbitals) {
      throw std::runtime_error("invalid occupied-space partition in input");
    }

    const MoldenOrbitals current_molden =
        parse_molden_file(current_molden_path, n_basis_functions);
    const MoldenOrbitals reference_molden =
        parse_molden_file(reference_molden_path, n_basis_functions);
    if (current_molden.n_orbitals < n_occupied_orbitals ||
        reference_molden.n_orbitals < n_occupied_orbitals) {
      throw std::runtime_error(
          "Molden file does not contain all occupied orbitals required for comparison");
    }

    const Eigen::MatrixXd basis_overlap_molden =
        build_molden_order_overlap_matrix(input);
    const Eigen::MatrixXd current_occupied =
        current_molden.coefficients_in_molden_order.leftCols(n_occupied_orbitals);
    const Eigen::MatrixXd reference_occupied =
        reference_molden.coefficients_in_molden_order.leftCols(n_occupied_orbitals);
    const Eigen::MatrixXd current_active_auxiliary =
        build_active_auxiliary_orbitals(
            current_occupied,
            basis_overlap_molden,
            n_inactive_orbitals,
            n_active_orbitals);
    const Eigen::MatrixXd reference_active_auxiliary =
        build_active_auxiliary_orbitals(
            reference_occupied,
            basis_overlap_molden,
            n_inactive_orbitals,
            n_active_orbitals);

    std::cout << std::setprecision(12);
    std::cout << "n_basis_functions = " << n_basis_functions << '\n';
    std::cout << "n_orbitals = " << n_orbitals << '\n';
    std::cout << "n_inactive_orbitals = " << n_inactive_orbitals << '\n';
    std::cout << "n_active_orbitals = " << n_active_orbitals << '\n';
    std::cout << "n_occupied_orbitals = " << n_occupied_orbitals << '\n';

    if (n_inactive_orbitals > 0) {
      print_orbital_cosines(
          "inactive_original_orbitals",
          current_occupied.leftCols(n_inactive_orbitals),
          reference_occupied.leftCols(n_inactive_orbitals),
          basis_overlap_molden,
          1);
    }
    print_orbital_cosines(
        "active_original_orbitals",
        current_occupied.middleCols(n_inactive_orbitals, n_active_orbitals),
        reference_occupied.middleCols(n_inactive_orbitals, n_active_orbitals),
        basis_overlap_molden,
        n_inactive_orbitals + 1);
    print_inactive_dual_overlap_summary(
        "current_active_original_inactive_dual_overlap",
        current_occupied,
        basis_overlap_molden,
        n_inactive_orbitals,
        n_active_orbitals,
        n_inactive_orbitals + 1);
    print_inactive_dual_overlap_summary(
        "reference_active_original_inactive_dual_overlap",
        reference_occupied,
        basis_overlap_molden,
        n_inactive_orbitals,
        n_active_orbitals,
        n_inactive_orbitals + 1);
    print_orbital_cosines(
        "active_auxiliary_orbitals",
        current_active_auxiliary,
        reference_active_auxiliary,
        basis_overlap_molden,
        n_inactive_orbitals + 1);
    print_active_subspace_singular_values(
        current_active_auxiliary,
        reference_active_auxiliary,
        basis_overlap_molden,
        n_inactive_orbitals + 1);

    if (n_inactive_orbitals > 0) {
      const Eigen::MatrixXd current_inactive =
          current_occupied.leftCols(n_inactive_orbitals);
      std::cout << "active_original_with_best_inactive_repair\n";
      for (int active_index = 0; active_index < n_active_orbitals; ++active_index) {
        const Eigen::VectorXd repaired_orbital =
            build_s_metric_inactive_repaired_orbital(
                current_occupied.col(n_inactive_orbitals + active_index),
                reference_occupied.col(n_inactive_orbitals + active_index),
                current_inactive,
                basis_overlap_molden);
        const double repaired_cosine =
            s_metric_cosine(
                repaired_orbital,
                reference_occupied.col(n_inactive_orbitals + active_index),
                basis_overlap_molden);
        std::cout << "  orbital " << std::setw(2)
                  << n_inactive_orbitals + active_index + 1
                  << "  s_cos = " << repaired_cosine << '\n';
      }
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "compare_molden_active_auxiliary: " << error.what() << '\n';
    return 1;
  }
}
