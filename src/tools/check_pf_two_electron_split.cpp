#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "pfaffian_vbscf/kernel/pf_forward_kernel.hpp"
#include "pfaffian_vbscf/kernel/pf_hamiltonian_tensor_terms.hpp"
#include "pfaffian_vbscf/kernel/pf_operand_views.hpp"
#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/math/dense_utils.hpp"
#include "pfaffian_vbscf/math/trace_projector.hpp"
#include "pfaffian_vbscf/matrices/pf_act_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_matrix_builder.hpp"
#include "pfaffian_vbscf/matrices/pf_pair_kernel_common.hpp"
#include "pfaffian_vbscf/scf/pf_basis_factory.hpp"
#include "pfaffian_vbscf/tensor/pf_tensor_contractor.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/matrices/two_electron_indexer.hpp"

namespace {

using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using GenericMatrix = xmvb::pfaffian_vbscf::detail::GenericMatrix<double>;

struct Options {
  std::string input_path;
  int k = 0;
  int seed = 20260328;
  int max_dets = 2000;
  int row = -1;
  int col = -1;
};

struct TwoElectronSplit {
  double total = 0.0;
  double cross = 0.0;
  double cumulant = 0.0;
  double same_spin_total = 0.0;
  double same_spin_cross = 0.0;
  double same_spin_cumulant = 0.0;
  double opposite_spin_total = 0.0;
  double opposite_spin_cross = 0.0;
  double opposite_spin_cumulant = 0.0;
};

struct MaxErrorRecord {
  double abs_error = 0.0;
  int row = 0;
  int col = 0;
};

struct PairDiagnostics {
  bool valid = false;
  int row = 0;
  int col = 0;
  TwoElectronSplit exact;
  double builder_total = 0.0;
  double builder_cross = 0.0;
  double builder_cumulant = 0.0;
  double builder_same_spin_cumulant = 0.0;
  double builder_opposite_spin_cumulant = 0.0;
  double builder_same_spin_alpha = 0.0;
  double builder_same_spin_beta = 0.0;
  double builder_same_spin_alpha_exchange = 0.0;
  double builder_same_spin_alpha_coulomb = 0.0;
  double builder_same_spin_beta_exchange = 0.0;
  double builder_same_spin_beta_coulomb = 0.0;
  double builder_opposite_exchange = 0.0;
  double builder_opposite_coulomb = 0.0;
};

struct CandidateDiagnostics {
  bool valid = false;
  int row = 0;
  int col = 0;
  TwoElectronSplit exact;
  double total = 0.0;
  double cumulant = 0.0;
  double same_spin_cumulant = 0.0;
  double opposite_spin_cumulant = 0.0;
};

void print_usage() {
  std::cerr << "usage: check_pf_two_electron_split <input.xmi> "
               "[--k K] [--seed S] [--max-dets N] [--row I --col J]\n";
}

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
    if (name == "--row") {
      opt.row = std::stoi(value);
      continue;
    }
    if (name == "--col") {
      opt.col = std::stoi(value);
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
  if ((opt.row < 0) != (opt.col < 0)) {
    throw std::invalid_argument("--row and --col must be provided together");
  }
  return opt;
}

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

template <typename Scalar>
std::vector<Scalar> to_vector(const xmvb::pfaffian_vbscf::ScalarBuffer& buffer) {
  std::vector<Scalar> out(buffer.size(), Scalar(0.0));
  for (std::size_t index = 0; index < buffer.size(); ++index) {
    out[index] = static_cast<Scalar>(buffer[index]);
  }
  return out;
}

TwoElectronSplit evaluate_exact_two_electron_split_generic(
    const GenericMatrix& left_pairing_matrix,
    const GenericMatrix& right_pairing_matrix,
    const GenericMatrix& spatial_overlap_matrix,
    const xmvb::pfaffian_vbscf::ScalarBuffer& packed_active_two_electron_integrals,
    int n_pairs) {
  namespace detail = xmvb::pfaffian_vbscf::detail;
  using xmvb::pfaffian_vbscf::TraceProjector;

  const int n_active_orbitals = static_cast<int>(spatial_overlap_matrix.rows());
  const int n_spatial_entries = n_active_orbitals * n_active_orbitals;

  GenericMatrix spin_orbital_overlap_matrix =
      GenericMatrix::Zero(2 * n_active_orbitals, 2 * n_active_orbitals);
  spin_orbital_overlap_matrix.topLeftCorner(
      n_active_orbitals,
      n_active_orbitals) = spatial_overlap_matrix;
  spin_orbital_overlap_matrix.bottomRightCorner(
      n_active_orbitals,
      n_active_orbitals) = spatial_overlap_matrix;
  const GenericMatrix left_pairing_transpose =
      left_pairing_matrix.transpose();
  const GenericMatrix kernel_matrix =
      detail::build_kernel_generic(
          left_pairing_transpose,
          spin_orbital_overlap_matrix,
          right_pairing_matrix);
  const std::vector<GenericMatrix> matrix_powers =
      detail::build_matrix_powers_including_identity_generic(kernel_matrix, n_pairs);
  const std::vector<double> traces =
      detail::build_power_traces_generic(kernel_matrix, n_pairs);
  const auto projection =
      TraceProjector::project(
          xmvb::pfaffian_vbscf::ScalarBuffer(traces.begin(), traces.end()),
          n_pairs);

  std::vector<GenericMatrix> alpha_source_directions(
      xmvb::to_size(n_spatial_entries));
  std::vector<GenericMatrix> beta_source_directions(
      xmvb::to_size(n_spatial_entries));
  std::vector<GenericMatrix> alpha_first_direction_matrices(
      xmvb::to_size(n_spatial_entries));
  std::vector<GenericMatrix> beta_first_direction_matrices(
      xmvb::to_size(n_spatial_entries));
  std::vector<std::vector<double>> alpha_first_derivative_traces(
      xmvb::to_size(n_spatial_entries));
  std::vector<std::vector<double>> beta_first_derivative_traces(
      xmvb::to_size(n_spatial_entries));

  const GenericMatrix left_transpose = left_pairing_matrix.transpose();
  const GenericMatrix right_times_source_transpose =
      right_pairing_matrix * spin_orbital_overlap_matrix.transpose();
  const GenericMatrix source_times_right =
      spin_orbital_overlap_matrix * right_pairing_matrix;

  for (int left_column = 0; left_column < n_active_orbitals; ++left_column) {
    for (int right_row = 0; right_row < n_active_orbitals; ++right_row) {
      const int entry_index =
          detail::spatial_source_entry_index(
              right_row,
              left_column,
              n_active_orbitals);
      alpha_source_directions[xmvb::to_size(entry_index)] =
          detail::build_spin_source_direction_matrix_generic<double>(
              n_active_orbitals,
              true,
              right_row,
              left_column);
      beta_source_directions[xmvb::to_size(entry_index)] =
          detail::build_spin_source_direction_matrix_generic<double>(
              n_active_orbitals,
              false,
              right_row,
              left_column);
      alpha_first_direction_matrices[xmvb::to_size(entry_index)] =
          left_transpose *
              alpha_source_directions[xmvb::to_size(entry_index)] *
              right_times_source_transpose +
          left_transpose *
              source_times_right *
              alpha_source_directions[xmvb::to_size(entry_index)].transpose();
      beta_first_direction_matrices[xmvb::to_size(entry_index)] =
          left_transpose *
              beta_source_directions[xmvb::to_size(entry_index)] *
              right_times_source_transpose +
          left_transpose *
              source_times_right *
              beta_source_directions[xmvb::to_size(entry_index)].transpose();
      alpha_first_derivative_traces[xmvb::to_size(entry_index)] =
          detail::build_first_derivative_power_traces_generic(
              matrix_powers,
              alpha_first_direction_matrices[xmvb::to_size(entry_index)],
              n_pairs);
      beta_first_derivative_traces[xmvb::to_size(entry_index)] =
          detail::build_first_derivative_power_traces_generic(
              matrix_powers,
              beta_first_direction_matrices[xmvb::to_size(entry_index)],
              n_pairs);
    }
  }

  TwoElectronSplit split;
  for (int left_first = 0; left_first < n_active_orbitals - 1; ++left_first) {
    for (int right_first = 0; right_first < n_active_orbitals - 1; ++right_first) {
      const int first_entry_index =
          detail::spatial_source_entry_index(
              right_first,
              left_first,
              n_active_orbitals);
      for (int left_second = left_first + 1;
           left_second < n_active_orbitals;
           ++left_second) {
        for (int right_second = right_first + 1;
             right_second < n_active_orbitals;
             ++right_second) {
          const int second_entry_index =
              detail::spatial_source_entry_index(
                  right_second,
                  left_second,
                  n_active_orbitals);
          const int direct_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_first,
                  right_second,
                  left_second);
          const int exchange_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_second,
                  right_second,
                  left_first);
          const double same_spin_interaction =
              packed_active_two_electron_integrals[xmvb::to_size(direct_index)] -
              packed_active_two_electron_integrals[xmvb::to_size(exchange_index)];

          const GenericMatrix alpha_second_direction_matrix =
              left_transpose *
                  alpha_source_directions[xmvb::to_size(first_entry_index)] *
                  right_pairing_matrix *
                  alpha_source_directions[xmvb::to_size(second_entry_index)].transpose() +
              left_transpose *
                  alpha_source_directions[xmvb::to_size(second_entry_index)] *
                  right_pairing_matrix *
                  alpha_source_directions[xmvb::to_size(first_entry_index)].transpose();
          const std::vector<double> alpha_second_derivative_traces =
              detail::build_second_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrices[xmvb::to_size(first_entry_index)],
                  alpha_first_direction_matrices[xmvb::to_size(second_entry_index)],
                  alpha_second_direction_matrix,
                  n_pairs);
          const double alpha_total =
              detail::projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces[xmvb::to_size(first_entry_index)],
                  alpha_first_derivative_traces[xmvb::to_size(second_entry_index)],
                  alpha_second_derivative_traces,
                  n_pairs);
          double alpha_cumulant = 0.0;
          for (int power = 0; power < n_pairs; ++power) {
            alpha_cumulant +=
                projection.trace_weights[xmvb::to_size(power)] *
                alpha_second_derivative_traces[xmvb::to_size(power)];
          }
          split.total += same_spin_interaction * alpha_total;
          split.cumulant += same_spin_interaction * alpha_cumulant;
          split.cross += same_spin_interaction * (alpha_total - alpha_cumulant);
          split.same_spin_total += same_spin_interaction * alpha_total;
          split.same_spin_cumulant += same_spin_interaction * alpha_cumulant;
          split.same_spin_cross +=
              same_spin_interaction * (alpha_total - alpha_cumulant);

          const GenericMatrix beta_second_direction_matrix =
              left_transpose *
                  beta_source_directions[xmvb::to_size(first_entry_index)] *
                  right_pairing_matrix *
                  beta_source_directions[xmvb::to_size(second_entry_index)].transpose() +
              left_transpose *
                  beta_source_directions[xmvb::to_size(second_entry_index)] *
                  right_pairing_matrix *
                  beta_source_directions[xmvb::to_size(first_entry_index)].transpose();
          const std::vector<double> beta_second_derivative_traces =
              detail::build_second_derivative_power_traces_generic(
                  matrix_powers,
                  beta_first_direction_matrices[xmvb::to_size(first_entry_index)],
                  beta_first_direction_matrices[xmvb::to_size(second_entry_index)],
                  beta_second_direction_matrix,
                  n_pairs);
          const double beta_total =
              detail::projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  beta_first_derivative_traces[xmvb::to_size(first_entry_index)],
                  beta_first_derivative_traces[xmvb::to_size(second_entry_index)],
                  beta_second_derivative_traces,
                  n_pairs);
          double beta_cumulant = 0.0;
          for (int power = 0; power < n_pairs; ++power) {
            beta_cumulant +=
                projection.trace_weights[xmvb::to_size(power)] *
                beta_second_derivative_traces[xmvb::to_size(power)];
          }
          split.total += same_spin_interaction * beta_total;
          split.cumulant += same_spin_interaction * beta_cumulant;
          split.cross += same_spin_interaction * (beta_total - beta_cumulant);
          split.same_spin_total += same_spin_interaction * beta_total;
          split.same_spin_cumulant += same_spin_interaction * beta_cumulant;
          split.same_spin_cross +=
              same_spin_interaction * (beta_total - beta_cumulant);
        }
      }
    }
  }

  for (int alpha_left_column = 0;
       alpha_left_column < n_active_orbitals;
       ++alpha_left_column) {
    for (int alpha_right_row = 0;
         alpha_right_row < n_active_orbitals;
         ++alpha_right_row) {
      const int alpha_entry_index =
          detail::spatial_source_entry_index(
              alpha_right_row,
              alpha_left_column,
              n_active_orbitals);
      for (int beta_left_column = 0;
           beta_left_column < n_active_orbitals;
           ++beta_left_column) {
        for (int beta_right_row = 0;
             beta_right_row < n_active_orbitals;
             ++beta_right_row) {
          const int beta_entry_index =
              detail::spatial_source_entry_index(
                  beta_right_row,
                  beta_left_column,
                  n_active_orbitals);
          const int opposite_spin_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  beta_right_row,
                  beta_left_column,
                  alpha_right_row,
                  alpha_left_column);
          const double interaction =
              packed_active_two_electron_integrals[xmvb::to_size(opposite_spin_index)];
          const GenericMatrix mixed_second_direction_matrix =
              left_transpose *
                  alpha_source_directions[xmvb::to_size(alpha_entry_index)] *
                  right_pairing_matrix *
                  beta_source_directions[xmvb::to_size(beta_entry_index)].transpose() +
              left_transpose *
                  beta_source_directions[xmvb::to_size(beta_entry_index)] *
                  right_pairing_matrix *
                  alpha_source_directions[xmvb::to_size(alpha_entry_index)].transpose();
          const std::vector<double> mixed_second_derivative_traces =
              detail::build_second_derivative_power_traces_generic(
                  matrix_powers,
                  alpha_first_direction_matrices[xmvb::to_size(alpha_entry_index)],
                  beta_first_direction_matrices[xmvb::to_size(beta_entry_index)],
                  mixed_second_direction_matrix,
                  n_pairs);
          const double mixed_total =
              detail::projected_overlap_second_derivative_coefficient_from_traces_generic(
                  traces,
                  alpha_first_derivative_traces[xmvb::to_size(alpha_entry_index)],
                  beta_first_derivative_traces[xmvb::to_size(beta_entry_index)],
                  mixed_second_derivative_traces,
                  n_pairs);
          double mixed_cumulant = 0.0;
          for (int power = 0; power < n_pairs; ++power) {
            mixed_cumulant +=
                projection.trace_weights[xmvb::to_size(power)] *
                mixed_second_derivative_traces[xmvb::to_size(power)];
          }
          split.total += interaction * mixed_total;
          split.cumulant += interaction * mixed_cumulant;
          split.cross += interaction * (mixed_total - mixed_cumulant);
          split.opposite_spin_total += interaction * mixed_total;
          split.opposite_spin_cumulant += interaction * mixed_cumulant;
          split.opposite_spin_cross +=
              interaction * (mixed_total - mixed_cumulant);
        }
      }
    }
  }

  return split;
}

void update_record(
    double candidate,
    int row,
    int col,
    MaxErrorRecord* record) {
  if (record == nullptr) {
    throw std::invalid_argument("record must not be null");
  }
  if (candidate > record->abs_error) {
    record->abs_error = candidate;
    record->row = row;
    record->col = col;
  }
}

double contract_same_spin_bridge(
    const xmvb::pfaffian_vbscf::ScalarBuffer& ggo,
    int n_active_orbitals,
    const xmvb::pfaffian_vbscf::ConstMatrixRef& left,
    const xmvb::pfaffian_vbscf::ConstMatrixRef& right) {
  if (left.rows() != n_active_orbitals || left.cols() != n_active_orbitals ||
      right.rows() != n_active_orbitals || right.cols() != n_active_orbitals) {
    throw std::invalid_argument("same-spin bridge operands must be spatial n x n matrices");
  }

  double value = 0.0;
  for (int left_first = 0; left_first < n_active_orbitals - 1; ++left_first) {
    for (int right_first = 0; right_first < n_active_orbitals - 1; ++right_first) {
      for (int left_second = left_first + 1;
           left_second < n_active_orbitals;
           ++left_second) {
        for (int right_second = right_first + 1;
             right_second < n_active_orbitals;
             ++right_second) {
          const double bridge_left = left(right_first, left_second);
          const double bridge_right = right(right_second, left_first);
          if (std::abs(bridge_left) < 1.0e-15 || std::abs(bridge_right) < 1.0e-15) {
            continue;
          }
          const int direct_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_first,
                  right_second,
                  left_second);
          const int exchange_index =
              xmvb::vb::TwoElectronIndexer::two_electron_storage_index(
                  right_first,
                  left_second,
                  right_second,
                  left_first);
          const double interaction =
              ggo[xmvb::to_size(direct_index)] -
              ggo[xmvb::to_size(exchange_index)];
          value += interaction * bridge_left * bridge_right;
        }
      }
    }
  }
  return value;
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
    if (opt.row >= basis.n_states || opt.col >= basis.n_states) {
      throw std::invalid_argument("selected state pair is out of range");
    }

    const Matrix spatial_overlap_matrix = dense_mat(act.sso, act.n_active_orbitals);
    const Matrix one_electron_matrix = dense_mat(act.hho, act.n_active_orbitals);
    const Matrix spin_metric =
        xmvb::pfaffian_vbscf::build_spin_block_diagonal_metric(spatial_overlap_matrix);

    std::vector<Matrix> pairing_matrices;
    pairing_matrices.reserve(xmvb::to_size(basis.n_states));
    for (const auto& state : basis.states) {
      pairing_matrices.push_back(
          xmvb::pfaffian_vbscf::decode_antisymm(
              state.packed_entries,
              state.n_spin_orbitals));
    }

    MaxErrorRecord total_error;
    MaxErrorRecord cross_error;
    MaxErrorRecord cumulant_error;
    MaxErrorRecord candidate_total_error;
    MaxErrorRecord candidate_cumulant_error;
    PairDiagnostics total_argmax_diag;
    CandidateDiagnostics candidate_argmax_diag;
    const bool restrict_pair = (opt.row >= 0);
    for (int row = 0; row < basis.n_states; ++row) {
      if (restrict_pair && row != opt.row) {
        continue;
      }
      const int col_begin = restrict_pair ? opt.col : 0;
      const int col_end = restrict_pair ? opt.col : row;
      for (int col = col_begin; col <= col_end; ++col) {
        const GenericMatrix left_pairing_matrix =
            pairing_matrices[xmvb::to_size(row)];
        const GenericMatrix right_pairing_matrix =
            pairing_matrices[xmvb::to_size(col)];
        const TwoElectronSplit exact =
            evaluate_exact_two_electron_split_generic(
                left_pairing_matrix,
                right_pairing_matrix,
                spatial_overlap_matrix,
                act.ggo,
                basis.n_beta);

        const xmvb::pfaffian_vbscf::PfKernelCache cache =
            xmvb::pfaffian_vbscf::PfForwardKernel::build_cache(
                pairing_matrices[xmvb::to_size(row)].transpose(),
                spin_metric,
                pairing_matrices[xmvb::to_size(col)],
                basis.n_beta);
        const std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> terms =
            xmvb::pfaffian_vbscf::build_two_electron_hamiltonian_tensor_terms(cache);

        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> cross_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> cumulant_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_cumulant_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> opposite_spin_cumulant_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_alpha_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_beta_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_alpha_exchange_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_alpha_coulomb_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_beta_exchange_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> same_spin_beta_coulomb_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> opposite_exchange_terms;
        std::vector<xmvb::pfaffian_vbscf::PfTensorTerm> opposite_coulomb_terms;
        for (const xmvb::pfaffian_vbscf::PfTensorTerm& term : terms) {
          const bool is_cross =
              term.left_operand.source == xmvb::pfaffian_vbscf::PfMatrixSource::TraceRDM ||
              term.right_operand.source == xmvb::pfaffian_vbscf::PfMatrixSource::TraceRDM;
          if (is_cross) {
            cross_terms.push_back(term);
            continue;
          }
          cumulant_terms.push_back(term);
          if (term.same_spin) {
            same_spin_cumulant_terms.push_back(term);
            if (term.left_operand.block == xmvb::pfaffian_vbscf::PfSpinBlock::AlphaAlpha) {
              same_spin_alpha_terms.push_back(term);
              if (term.contraction == xmvb::pfaffian_vbscf::PfTensorContraction::Exchange) {
                same_spin_alpha_exchange_terms.push_back(term);
              } else if (
                  term.contraction ==
                  xmvb::pfaffian_vbscf::PfTensorContraction::Coulomb) {
                same_spin_alpha_coulomb_terms.push_back(term);
              }
            } else if (
                term.left_operand.block ==
                xmvb::pfaffian_vbscf::PfSpinBlock::BetaBeta) {
              same_spin_beta_terms.push_back(term);
              if (term.contraction == xmvb::pfaffian_vbscf::PfTensorContraction::Exchange) {
                same_spin_beta_exchange_terms.push_back(term);
              } else if (
                  term.contraction ==
                  xmvb::pfaffian_vbscf::PfTensorContraction::Coulomb) {
                same_spin_beta_coulomb_terms.push_back(term);
              }
            }
          } else {
            opposite_spin_cumulant_terms.push_back(term);
            if (term.contraction == xmvb::pfaffian_vbscf::PfTensorContraction::Exchange) {
              opposite_exchange_terms.push_back(term);
            } else if (
                term.contraction == xmvb::pfaffian_vbscf::PfTensorContraction::Coulomb) {
              opposite_coulomb_terms.push_back(term);
            }
          }
        }

        const double builder_cross =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                cross_terms);
        const double builder_cumulant =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                cumulant_terms);
        const double builder_same_spin_cumulant =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_cumulant_terms);
        const double builder_opposite_spin_cumulant =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                opposite_spin_cumulant_terms);
        const double builder_same_spin_alpha =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_alpha_terms);
        const double builder_same_spin_beta =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_beta_terms);
        const double builder_opposite_exchange =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                opposite_exchange_terms);
        const double builder_opposite_coulomb =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                opposite_coulomb_terms);
        const double builder_same_spin_alpha_exchange =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_alpha_exchange_terms);
        const double builder_same_spin_alpha_coulomb =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_alpha_coulomb_terms);
        const double builder_same_spin_beta_exchange =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_beta_exchange_terms);
        const double builder_same_spin_beta_coulomb =
            xmvb::pfaffian_vbscf::PfForwardKernel::evaluate_tensor_terms(
                cache,
                act.ggo,
                same_spin_beta_coulomb_terms);
        const double builder_total = builder_cross + builder_cumulant;

        const Matrix left_sigma_right =
            cache.left * cache.sigma * cache.right;
        const Matrix right_sigma = cache.right * cache.sigma;
        std::vector<Matrix> kernel_power_times_left_sigma_right(
            xmvb::to_size(cache.trace_order));
        std::vector<Matrix> right_sigma_times_kernel_power_times_left_sigma_right(
            xmvb::to_size(cache.trace_order));
        for (int power = 0; power < cache.trace_order; ++power) {
          kernel_power_times_left_sigma_right[xmvb::to_size(power)] =
              cache.kernel_powers[xmvb::to_size(power)] *
              left_sigma_right;
          right_sigma_times_kernel_power_times_left_sigma_right[xmvb::to_size(power)] =
              right_sigma *
              kernel_power_times_left_sigma_right[xmvb::to_size(power)];
        }

        double candidate_same_spin_cumulant = 0.0;
        double candidate_opposite_spin_cumulant = 0.0;
        for (int power = 0; power < cache.trace_order; ++power) {
          const double scale =
              cache.trace_weights[xmvb::to_size(power)] *
              static_cast<double>(power + 1);
          if (scale == 0.0) {
            continue;
          }

          candidate_opposite_spin_cumulant +=
              scale *
              xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
                  act.ggo,
                  cache.n_active_orbitals,
                  xmvb::pfaffian_vbscf::detail::copy_spin_block(
                      cache,
                      cache.right,
                      xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta),
                  xmvb::pfaffian_vbscf::detail::copy_spin_block(
                      cache,
                      cache.kernel_power_times_left[xmvb::to_size(power)],
                      xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha));
          candidate_opposite_spin_cumulant +=
              scale *
              xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
                  act.ggo,
                  cache.n_active_orbitals,
                  xmvb::pfaffian_vbscf::detail::copy_spin_block(
                      cache,
                      cache.right,
                      xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha),
                  xmvb::pfaffian_vbscf::detail::copy_spin_block(
                      cache,
                      cache.kernel_power_times_left[xmvb::to_size(power)],
                      xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta));

          if (power == 0) {
            continue;
          }

          for (int split_power = 0; split_power <= power - 1; ++split_power) {
            const int tail_power = power - 1 - split_power;
            candidate_same_spin_cumulant +=
                (2.0 * scale) *
                contract_same_spin_bridge(
                    act.ggo,
                    cache.n_active_orbitals,
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        kernel_power_times_left_sigma_right[
                            xmvb::to_size(split_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::AlphaAlpha),
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        kernel_power_times_left_sigma_right[
                            xmvb::to_size(tail_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::AlphaAlpha));
            candidate_same_spin_cumulant +=
                (2.0 * scale) *
                contract_same_spin_bridge(
                    act.ggo,
                    cache.n_active_orbitals,
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        kernel_power_times_left_sigma_right[
                            xmvb::to_size(split_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::BetaBeta),
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        kernel_power_times_left_sigma_right[
                            xmvb::to_size(tail_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::BetaBeta));
            candidate_opposite_spin_cumulant +=
                scale *
                xmvb::pfaffian_vbscf::PfTensorContractor::contract_exchange(
                    act.ggo,
                    cache.n_active_orbitals,
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        right_sigma_times_kernel_power_times_left_sigma_right[
                            xmvb::to_size(split_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta),
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        cache.kernel_power_times_left[
                            xmvb::to_size(tail_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha));
            candidate_opposite_spin_cumulant +=
                scale *
                xmvb::pfaffian_vbscf::PfTensorContractor::contract_coulomb(
                    act.ggo,
                    cache.n_active_orbitals,
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        right_sigma_times_kernel_power_times_left_sigma_right[
                            xmvb::to_size(tail_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha),
                    xmvb::pfaffian_vbscf::detail::copy_spin_block(
                        cache,
                        cache.kernel_power_times_left[
                            xmvb::to_size(split_power)],
                        xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta));
          }
        }
        const double candidate_cumulant =
            candidate_same_spin_cumulant + candidate_opposite_spin_cumulant;
        const double candidate_total = builder_cross + candidate_cumulant;

        const double total_abs_error = std::abs(builder_total - exact.total);
        update_record(
            total_abs_error,
            row,
            col,
            &total_error);
        update_record(
            std::abs(builder_cross - exact.cross),
            row,
            col,
            &cross_error);
        update_record(
            std::abs(builder_cumulant - exact.cumulant),
            row,
            col,
            &cumulant_error);
        update_record(
            std::abs(candidate_total - exact.total),
            row,
            col,
            &candidate_total_error);
        update_record(
            std::abs(candidate_cumulant - exact.cumulant),
            row,
            col,
            &candidate_cumulant_error);
        const double candidate_total_abs_error =
            std::abs(candidate_total - exact.total);
        if (candidate_total_abs_error >= candidate_total_error.abs_error) {
          candidate_argmax_diag.valid = true;
          candidate_argmax_diag.row = row;
          candidate_argmax_diag.col = col;
          candidate_argmax_diag.exact = exact;
          candidate_argmax_diag.total = candidate_total;
          candidate_argmax_diag.cumulant = candidate_cumulant;
          candidate_argmax_diag.same_spin_cumulant = candidate_same_spin_cumulant;
          candidate_argmax_diag.opposite_spin_cumulant = candidate_opposite_spin_cumulant;
        }
        if (total_abs_error >= total_error.abs_error) {
          total_argmax_diag.valid = true;
          total_argmax_diag.row = row;
          total_argmax_diag.col = col;
          total_argmax_diag.exact = exact;
          total_argmax_diag.builder_total = builder_total;
          total_argmax_diag.builder_cross = builder_cross;
          total_argmax_diag.builder_cumulant = builder_cumulant;
          total_argmax_diag.builder_same_spin_cumulant = builder_same_spin_cumulant;
          total_argmax_diag.builder_opposite_spin_cumulant = builder_opposite_spin_cumulant;
          total_argmax_diag.builder_same_spin_alpha = builder_same_spin_alpha;
          total_argmax_diag.builder_same_spin_beta = builder_same_spin_beta;
          total_argmax_diag.builder_same_spin_alpha_exchange =
              builder_same_spin_alpha_exchange;
          total_argmax_diag.builder_same_spin_alpha_coulomb =
              builder_same_spin_alpha_coulomb;
          total_argmax_diag.builder_same_spin_beta_exchange =
              builder_same_spin_beta_exchange;
          total_argmax_diag.builder_same_spin_beta_coulomb =
              builder_same_spin_beta_coulomb;
          total_argmax_diag.builder_opposite_exchange = builder_opposite_exchange;
          total_argmax_diag.builder_opposite_coulomb = builder_opposite_coulomb;
        }
      }
    }

    std::cout << std::setprecision(12);
    std::cout << "requested_k = ";
    if (opt.k == 0) {
      std::cout << "auto\n";
    } else {
      std::cout << opt.k << '\n';
    }
    if (restrict_pair) {
      std::cout << "requested_row = " << opt.row << '\n';
      std::cout << "requested_col = " << opt.col << '\n';
    }
    std::cout << "k = " << basis.n_states << '\n';
    std::cout << "seed = " << opt.seed << '\n';
    std::cout << "n_active_orbitals = " << act.n_active_orbitals << '\n';
    std::cout << "n_alpha = " << act.n_alpha << '\n';
    std::cout << "n_beta = " << act.n_beta << '\n';
    std::cout << "split_total_max_abs_error = " << total_error.abs_error << '\n';
    std::cout << "split_total_argmax = (" << total_error.row << ", " << total_error.col << ")\n";
    std::cout << "split_cross_max_abs_error = " << cross_error.abs_error << '\n';
    std::cout << "split_cross_argmax = (" << cross_error.row << ", " << cross_error.col << ")\n";
    std::cout << "split_cumulant_max_abs_error = " << cumulant_error.abs_error << '\n';
    std::cout << "split_cumulant_argmax = (" << cumulant_error.row << ", " << cumulant_error.col << ")\n";
    std::cout << "candidate_total_max_abs_error = " << candidate_total_error.abs_error << '\n';
    std::cout << "candidate_total_argmax = (" << candidate_total_error.row << ", "
              << candidate_total_error.col << ")\n";
    std::cout << "candidate_cumulant_max_abs_error = " << candidate_cumulant_error.abs_error
              << '\n';
    std::cout << "candidate_cumulant_argmax = (" << candidate_cumulant_error.row << ", "
              << candidate_cumulant_error.col << ")\n";
    if (candidate_argmax_diag.valid) {
      std::cout << "candidate_argmax_total = " << candidate_argmax_diag.total << '\n';
      std::cout << "candidate_argmax_exact_total = " << candidate_argmax_diag.exact.total
                << '\n';
      std::cout << "candidate_argmax_cumulant = " << candidate_argmax_diag.cumulant << '\n';
      std::cout << "candidate_argmax_exact_cumulant = "
                << candidate_argmax_diag.exact.cumulant << '\n';
      std::cout << "candidate_argmax_same_spin_cumulant = "
                << candidate_argmax_diag.same_spin_cumulant << '\n';
      std::cout << "candidate_argmax_exact_same_spin_cumulant = "
                << candidate_argmax_diag.exact.same_spin_cumulant << '\n';
      std::cout << "candidate_argmax_opposite_spin_cumulant = "
                << candidate_argmax_diag.opposite_spin_cumulant << '\n';
      std::cout << "candidate_argmax_exact_opposite_spin_cumulant = "
                << candidate_argmax_diag.exact.opposite_spin_cumulant << '\n';
    }
    if (total_argmax_diag.valid) {
      std::cout << "argmax_builder_total = " << total_argmax_diag.builder_total << '\n';
      std::cout << "argmax_exact_total = " << total_argmax_diag.exact.total << '\n';
      std::cout << "argmax_builder_cross = " << total_argmax_diag.builder_cross << '\n';
      std::cout << "argmax_exact_cross = " << total_argmax_diag.exact.cross << '\n';
      std::cout << "argmax_builder_cumulant = " << total_argmax_diag.builder_cumulant << '\n';
      std::cout << "argmax_exact_cumulant = " << total_argmax_diag.exact.cumulant << '\n';
      std::cout << "argmax_builder_same_spin_cumulant = "
                << total_argmax_diag.builder_same_spin_cumulant << '\n';
      std::cout << "argmax_exact_same_spin_cumulant = "
                << total_argmax_diag.exact.same_spin_cumulant << '\n';
      std::cout << "argmax_builder_opposite_spin_cumulant = "
                << total_argmax_diag.builder_opposite_spin_cumulant << '\n';
      std::cout << "argmax_exact_opposite_spin_cumulant = "
                << total_argmax_diag.exact.opposite_spin_cumulant << '\n';
      std::cout << "argmax_builder_same_spin_alpha = "
                << total_argmax_diag.builder_same_spin_alpha << '\n';
      std::cout << "argmax_builder_same_spin_beta = "
                << total_argmax_diag.builder_same_spin_beta << '\n';
      std::cout << "argmax_builder_same_spin_alpha_exchange = "
                << total_argmax_diag.builder_same_spin_alpha_exchange << '\n';
      std::cout << "argmax_builder_same_spin_alpha_coulomb = "
                << total_argmax_diag.builder_same_spin_alpha_coulomb << '\n';
      std::cout << "argmax_builder_same_spin_beta_exchange = "
                << total_argmax_diag.builder_same_spin_beta_exchange << '\n';
      std::cout << "argmax_builder_same_spin_beta_coulomb = "
                << total_argmax_diag.builder_same_spin_beta_coulomb << '\n';
      std::cout << "argmax_builder_opposite_exchange = "
                << total_argmax_diag.builder_opposite_exchange << '\n';
      std::cout << "argmax_builder_opposite_coulomb = "
                << total_argmax_diag.builder_opposite_coulomb << '\n';
      std::cout << "argmax_exact_same_spin_cross = "
                << total_argmax_diag.exact.same_spin_cross << '\n';
      std::cout << "argmax_exact_opposite_spin_cross = "
                << total_argmax_diag.exact.opposite_spin_cross << '\n';

      const xmvb::pfaffian_vbscf::PfKernelCache argmax_cache =
          xmvb::pfaffian_vbscf::PfForwardKernel::build_cache(
              pairing_matrices[xmvb::to_size(total_argmax_diag.row)].transpose(),
              spin_metric,
              pairing_matrices[xmvb::to_size(total_argmax_diag.col)],
              basis.n_beta);
      std::cout << "argmax_right_aa_norm = "
                << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                       argmax_cache,
                       argmax_cache.right,
                       xmvb::pfaffian_vbscf::PfSpinBlock::AlphaAlpha)
                       .norm() << '\n';
      std::cout << "argmax_right_ab_norm = "
                << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                       argmax_cache,
                       argmax_cache.right,
                       xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta)
                       .norm() << '\n';
      std::cout << "argmax_right_ba_norm = "
                << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                       argmax_cache,
                       argmax_cache.right,
                       xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha)
                       .norm() << '\n';
      std::cout << "argmax_right_bb_norm = "
                << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                       argmax_cache,
                       argmax_cache.right,
                       xmvb::pfaffian_vbscf::PfSpinBlock::BetaBeta)
                       .norm() << '\n';
      for (int power = 0; power < basis.n_beta; ++power) {
        const Matrix& kpl =
            argmax_cache.kernel_power_times_left[xmvb::to_size(power)];
        std::cout << "argmax_kpl_power_" << power << "_aa_norm = "
                  << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                         argmax_cache,
                         kpl,
                         xmvb::pfaffian_vbscf::PfSpinBlock::AlphaAlpha)
                         .norm() << '\n';
        std::cout << "argmax_kpl_power_" << power << "_ab_norm = "
                  << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                         argmax_cache,
                         kpl,
                         xmvb::pfaffian_vbscf::PfSpinBlock::AlphaBeta)
                         .norm() << '\n';
        std::cout << "argmax_kpl_power_" << power << "_ba_norm = "
                  << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                         argmax_cache,
                         kpl,
                         xmvb::pfaffian_vbscf::PfSpinBlock::BetaAlpha)
                         .norm() << '\n';
        std::cout << "argmax_kpl_power_" << power << "_bb_norm = "
                  << xmvb::pfaffian_vbscf::detail::copy_spin_block(
                         argmax_cache,
                         kpl,
                         xmvb::pfaffian_vbscf::PfSpinBlock::BetaBeta)
                         .norm() << '\n';
      }
    }
    (void) one_electron_matrix;
    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
