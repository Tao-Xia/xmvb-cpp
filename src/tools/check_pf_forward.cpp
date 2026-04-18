#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_pair_kernels.hpp"
#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/full_structure_builder.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

struct Options {
  std::string input_path;
  int k = 0;
  int seed = 20260328;
  int max_dets = 2000;
};

Matrix dense_mat(
    const std::vector<double>& data,
    int dim);

/**
 * @brief Prints tool usage.
 */
void print_usage() {
  std::cerr << "usage: check_pf_forward <input.xmi> "
               "[--k K] [--seed S] [--max-dets N]\n"
               "  --k 0 uses all selected raw VB structures.\n";
}

/**
 * @brief Parses command-line options.
 */
Options parse_args(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options opt;
  opt.input_path = argv[1];
  for (int i = 2; i < argc; i += 2) {
    const std::string name = argv[i];
    const std::string value = argv[i + 1];
    if (name == "--k") {
      opt.k = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      opt.seed = std::stoi(value);
      continue;
    }
    if (name == "--max-dets") {
      opt.max_dets = std::stoi(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (opt.k < 0) {
    throw std::invalid_argument("--k must be non-negative");
  }
  if (opt.max_dets <= 0) {
    throw std::invalid_argument("--max-dets must be positive");
  }
  return opt;
}

/**
 * @brief Evaluates the Pfaffian of a dense antisymmetric matrix.
 */
double skew_pf(Matrix mat) {
  if (mat.rows() != mat.cols()) {
    throw std::invalid_argument("Pfaffian requires a square matrix");
  }
  const int dim = mat.rows();
  if ((dim % 2) != 0) {
    return 0.0;
  }
  if (dim == 0) {
    return 1.0;
  }

  double value = 1.0;
  for (int pivot_row = 0; pivot_row < dim - 1; pivot_row += 2) {
    int pivot_col = pivot_row + 1;
    double pivot_mag = std::abs(mat(pivot_row, pivot_col));
    for (int col = pivot_row + 2; col < dim; ++col) {
      const double cand_mag = std::abs(mat(pivot_row, col));
      if (cand_mag > pivot_mag) {
        pivot_mag = cand_mag;
        pivot_col = col;
      }
    }

    if (pivot_mag <= 1.0e-14) {
      return 0.0;
    }

    if (pivot_col != pivot_row + 1) {
      mat.row(pivot_row + 1).swap(mat.row(pivot_col));
      mat.col(pivot_row + 1).swap(mat.col(pivot_col));
      value = -value;
    }

    const double pivot = mat(pivot_row, pivot_row + 1);
    value *= pivot;
    for (int row = pivot_row + 2; row < dim; ++row) {
      for (int col = row + 1; col < dim; ++col) {
        const double next =
            mat(row, col) -
            (mat(pivot_row, row) * mat(pivot_row + 1, col) -
             mat(pivot_row, col) * mat(pivot_row + 1, row)) /
                pivot;
        mat(row, col) = next;
        mat(col, row) = -next;
      }
    }
  }
  return value;
}

/**
 * @brief Builds one spin-orbital occupation list from alpha/beta occupations.
 */
std::vector<int> spin_occ(
    const std::vector<int>& alpha_occ,
    const std::vector<int>& beta_occ,
    int n_act) {
  std::vector<int> occ;
  occ.reserve(alpha_occ.size() + beta_occ.size());
  for (const int orb : alpha_occ) {
    occ.push_back(orb);
  }
  for (const int orb : beta_occ) {
    occ.push_back(n_act + orb);
  }
  return occ;
}

/**
 * @brief Extracts the occupied spin-orbital pairing submatrix.
 */
Matrix det_pair_mat(
    const Matrix& pair_mat,
    const std::vector<int>& occ) {
  const int n = static_cast<int>(occ.size());
  Matrix sub = Matrix::Zero(n, n);
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      sub(row, col) = pair_mat(occ[xmvb::to_size(row)],
                               occ[xmvb::to_size(col)]);
    }
  }
  return sub;
}

/**
 * @brief Computes one Pfaffian determinant amplitude.
 */
double det_amp(
    const Matrix& pair_mat,
    const std::vector<int>& occ) {
  return skew_pf(det_pair_mat(pair_mat, occ));
}

/**
 * @brief Builds exact determinant-space reference matrices for the current Pf basis.
 */
void build_exact_mats(
    const xmvb::pfaffian_vbscf::PfBasisData& basis,
    const xmvb::vb::CppVbInput& input,
    const xmvb::pfaffian_vbscf::PfActiveSpaceData& act,
    Matrix* s_exact,
    Matrix* h1e_exact,
    Matrix* h_exact) {
  if (s_exact == nullptr || h1e_exact == nullptr || h_exact == nullptr) {
    throw std::invalid_argument("output matrices must not be null");
  }

  xmvb::vb::FullDeterminantStructureHamiltonianOverlapBuilder det_builder(
      xmvb::vb::VBSCFAlgorithm::Original);
  const auto det_build =
      det_builder.build_with_pair_evaluations(
          input.structure_data.alpha_det,
          input.structure_data.beta_det,
          input.structure_data.determinant_to_structure_terms,
          act.sso,
          act.hho,
          act.n_active_orbitals,
          act.ggo,
          input.structure_data.n_structures);
  if (det_build.n_determinants > 2000000000) {
    throw std::runtime_error("unexpected determinant count");
  }

  const int n_dets = det_build.n_determinants;
  const int k = basis.n_states;
  Matrix s_det = Matrix::Zero(n_dets, n_dets);
  Matrix h1e_det = Matrix::Zero(n_dets, n_dets);
  Matrix h_det = Matrix::Zero(n_dets, n_dets);
  for (int row = 0; row < n_dets; ++row) {
    for (int col = 0; col <= row; ++col) {
      const auto& pair = det_build.pair_evaluation(row, col);
      s_det(row, col) = pair.overlap_determinant;
      s_det(col, row) = pair.overlap_determinant;
      h1e_det(row, col) = pair.one_electron_hamiltonian;
      h1e_det(col, row) = pair.one_electron_hamiltonian;
      h_det(row, col) = pair.total_hamiltonian;
      h_det(col, row) = pair.total_hamiltonian;
    }
  }

  std::vector<std::vector<int>> occs;
  occs.reserve(xmvb::to_size(n_dets));
  for (int det = 0; det < n_dets; ++det) {
    occs.push_back(
        spin_occ(
            input.structure_data.alpha_det[xmvb::to_size(det)],
            input.structure_data.beta_det[xmvb::to_size(det)],
            act.n_active_orbitals));
  }

  Matrix q = Matrix::Zero(n_dets, k);
  for (int state = 0; state < k; ++state) {
    const auto& st = basis.states[xmvb::to_size(state)];
    const Matrix pair_mat =
        xmvb::pfaffian_vbscf::decode_antisymm(st.packed_entries, st.n_spin_orbitals);
    for (int det = 0; det < n_dets; ++det) {
      q(det, state) = det_amp(pair_mat, occs[xmvb::to_size(det)]);
    }
  }

  *s_exact = q.transpose() * s_det * q;
  *h1e_exact = q.transpose() * h1e_det * q;
  *h_exact = q.transpose() * h_det * q;
}

/**
 * @brief Rebuilds the candidate one-electron matrix without using main-path storage.
 */
Matrix build_candidate_h1e(
    const xmvb::pfaffian_vbscf::PfBasisData& basis,
    const xmvb::pfaffian_vbscf::PfActiveSpaceData& act) {
  const int k = basis.n_states;
  const Matrix spatial_overlap_matrix = dense_mat(act.sso, act.n_active_orbitals);
  const Matrix one_electron_matrix = dense_mat(act.hho, act.n_active_orbitals);

  std::vector<Matrix> pairing_matrices;
  pairing_matrices.reserve(xmvb::to_size(k));
  for (const auto& state : basis.states) {
    pairing_matrices.push_back(
        xmvb::pfaffian_vbscf::decode_antisymm(
            state.packed_entries,
            state.n_spin_orbitals));
  }

  Matrix h1e_cand = Matrix::Zero(k, k);
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col <= row; ++col) {
      const auto pair =
          xmvb::pfaffian_vbscf::evaluate_pf_pair_kernel(
              pairing_matrices[xmvb::to_size(row)],
              pairing_matrices[xmvb::to_size(col)],
              spatial_overlap_matrix,
              one_electron_matrix,
              act.ggo,
              act.n_alpha,
              act.n_beta,
              basis.n_beta);
      h1e_cand(row, col) = pair.one_electron_hamiltonian;
      h1e_cand(col, row) = pair.one_electron_hamiltonian;
    }
  }
  return h1e_cand;
}

/**
 * @brief Returns the maximum absolute entrywise error.
 */
double max_abs_err(
    const Matrix& lhs,
    const Matrix& rhs) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    throw std::invalid_argument("matrix dimensions must match");
  }
  double err = 0.0;
  for (int row = 0; row < lhs.rows(); ++row) {
    for (int col = 0; col < lhs.cols(); ++col) {
      err = std::max(err, std::abs(lhs(row, col) - rhs(row, col)));
    }
  }
  return err;
}

/**
 * @brief Converts column-major storage to an Eigen matrix.
 */
Matrix dense_mat(
    const std::vector<double>& data,
    int dim) {
  Matrix mat = Matrix::Zero(dim, dim);
  for (int col = 0; col < dim; ++col) {
    for (int row = 0; row < dim; ++row) {
      mat(row, col) = data[xmvb::to_size(col) * dim + row];
    }
  }
  return mat;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    const auto load = xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);
    if (static_cast<int>(load.input.structure_data.alpha_det.size()) > opt.max_dets) {
      throw std::runtime_error("determinant count exceeds --max-dets");
    }

    xmvb::pfaffian_vbscf::PfActBuilder act_builder;
    const xmvb::pfaffian_vbscf::PfActiveSpaceData act =
        act_builder.build(load.input);
    xmvb::pfaffian_vbscf::PfBasisFactoryOptions basis_options;
    basis_options.n_states = opt.k;
    basis_options.seed = opt.seed;
    const auto basis =
        xmvb::pfaffian_vbscf::build_structure_pf_basis(
            load.input,
            load.raw_structure_data,
            basis_options);

    xmvb::pfaffian_vbscf::PfMatrixBuilder mat_builder;
    const auto cand = mat_builder.build(basis, act);

    Matrix s_exact;
    Matrix h1e_exact;
    Matrix h_exact;
    build_exact_mats(basis, load.input, act, &s_exact, &h1e_exact, &h_exact);

    const Matrix s_cand = dense_mat(cand.s, basis.n_states);
    const Matrix h1e_cand = build_candidate_h1e(basis, act);
    const Matrix h_cand = dense_mat(cand.h, basis.n_states);

    std::cout << std::setprecision(12);
    std::cout << "requested_k = ";
    if (opt.k == 0) {
      std::cout << "auto\n";
    } else {
      std::cout << opt.k << '\n';
    }
    std::cout << "k = " << basis.n_states << '\n';
    std::cout << "seed = " << opt.seed << '\n';
    std::cout << "n_active_orbitals = " << act.n_active_orbitals << '\n';
    std::cout << "n_alpha = " << act.n_alpha << '\n';
    std::cout << "n_beta = " << act.n_beta << '\n';
    std::cout << "source_raw_structure_count = "
              << load.source_raw_structure_count << '\n';
    std::cout << "selected_raw_structure_count = "
              << load.raw_structure_data.n_structures << '\n';
    std::cout << "expanded_determinant_count = "
              << load.input.structure_data.alpha_det.size() << '\n';
    std::cout << "overlap_max_abs_error = " << max_abs_err(s_cand, s_exact) << '\n';
    std::cout << "one_electron_max_abs_error = " << max_abs_err(h1e_cand, h1e_exact) << '\n';
    std::cout << "hamiltonian_max_abs_error = " << max_abs_err(h_cand, h_exact) << '\n';
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
