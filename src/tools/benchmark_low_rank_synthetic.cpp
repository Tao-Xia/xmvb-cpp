#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

namespace {

struct Options {
  int min_m = 4;
  int max_m = 10;
  int n_auxiliary_functions = 512;
  int n_steps = 500;
  int seed = 20260427;
  double singular_threshold = 1.0e-12;
};

struct BenchmarkResult {
  int n_active_orbitals = 0;
  int n_electrons = 0;
  int n_auxiliary_functions = 0;
  int n_steps = 0;
  int n_resets = 0;
  double direct_overlap_seconds = 0.0;
  double rank1_overlap_seconds = 0.0;
  double direct_phi2_seconds = 0.0;
  double update_phi2_seconds = 0.0;
  double max_inverse_abs_diff = 0.0;
  double max_det_abs_diff = 0.0;
  double max_phi2_abs_diff = 0.0;
  double direct_phi2_checksum = 0.0;
  double update_phi2_checksum = 0.0;
};

void print_usage() {
  std::cerr
      << "usage: benchmark_low_rank_synthetic "
      << "[--min-m M] [--max-m M] [--naux N] [--steps N] [--seed S]\n";
}

int parse_positive_int(const std::string& text, const char* option_name) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(option_name) + " must be positive");
  }
  return value;
}

Options parse_arguments(int argc, char** argv) {
  Options options;
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string name = argv[argument_index];
    if (name == "--help") {
      print_usage();
      std::exit(0);
    }
    if (argument_index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + name);
    }
    const std::string value = argv[++argument_index];
    if (name == "--min-m") {
      options.min_m = parse_positive_int(value, "--min-m");
    } else if (name == "--max-m") {
      options.max_m = parse_positive_int(value, "--max-m");
    } else if (name == "--naux") {
      options.n_auxiliary_functions = parse_positive_int(value, "--naux");
    } else if (name == "--steps") {
      options.n_steps = parse_positive_int(value, "--steps");
    } else if (name == "--seed") {
      options.seed = std::stoi(value);
    } else {
      throw std::invalid_argument("unknown argument: " + name);
    }
  }
  if (options.min_m > options.max_m) {
    throw std::invalid_argument("--min-m must be <= --max-m");
  }
  return options;
}

int packed_pair_index(int first, int second) {
  const int high = std::max(first, second);
  const int low = std::min(first, second);
  return high * (high + 1) / 2 + low;
}

Eigen::MatrixXd make_random_overlap(
    int n_active_orbitals,
    std::mt19937* generator) {
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::MatrixXd coefficient(n_active_orbitals, n_active_orbitals);
  for (int column = 0; column < n_active_orbitals; ++column) {
    for (int row = 0; row < n_active_orbitals; ++row) {
      coefficient(row, column) = normal(*generator);
    }
  }

  Eigen::MatrixXd overlap =
      coefficient * coefficient.transpose() /
      static_cast<double>(n_active_orbitals);
  overlap.diagonal().array() += 0.5;
  for (int index = 0; index < n_active_orbitals; ++index) {
    const double scale = 1.0 / std::sqrt(overlap(index, index));
    overlap.row(index) *= scale;
    overlap.col(index) *= scale;
  }
  return overlap;
}

Eigen::MatrixXd make_random_ri_factors(
    int n_auxiliary_functions,
    int n_active_orbitals,
    std::mt19937* generator) {
  std::normal_distribution<double> normal(0.0, 1.0);
  const int n_packed_pairs = n_active_orbitals * (n_active_orbitals + 1) / 2;
  Eigen::MatrixXd factors(n_auxiliary_functions, n_packed_pairs);
  const double scale = 1.0 / std::sqrt(static_cast<double>(n_auxiliary_functions));
  for (int pair = 0; pair < n_packed_pairs; ++pair) {
    for (int auxiliary = 0; auxiliary < n_auxiliary_functions; ++auxiliary) {
      factors(auxiliary, pair) = scale * normal(*generator);
    }
  }
  return factors;
}

Eigen::MatrixXd build_overlap_block(
    const Eigen::MatrixXd& active_overlap,
    const std::vector<int>& left,
    const std::vector<int>& right) {
  const int m = static_cast<int>(left.size());
  Eigen::MatrixXd block(m, m);
  for (int column = 0; column < m; ++column) {
    for (int row = 0; row < m; ++row) {
      block(row, column) = active_overlap(right[row], left[column]);
    }
  }
  return block;
}

bool resolve_regular_overlap(
    const Eigen::MatrixXd& overlap,
    double singular_threshold,
    Eigen::MatrixXd* inverse,
    double* determinant) {
  Eigen::FullPivLU<Eigen::MatrixXd> lu(overlap);
  lu.setThreshold(singular_threshold);
  if (lu.rank() != overlap.rows()) {
    return false;
  }
  *determinant = lu.determinant();
  if (*determinant == 0.0) {
    return false;
  }
  *inverse = lu.inverse();
  return true;
}

std::vector<int> make_initial_occupation(int m) {
  std::vector<int> occupation(m);
  std::iota(occupation.begin(), occupation.end(), 0);
  return occupation;
}

int choose_new_orbital(
    const std::vector<int>& occupation,
    int n_active_orbitals,
    std::mt19937* generator) {
  std::vector<int> unoccupied;
  for (int orbital = 0; orbital < n_active_orbitals; ++orbital) {
    if (std::find(occupation.begin(), occupation.end(), orbital) ==
        occupation.end()) {
      unoccupied.push_back(orbital);
    }
  }
  std::uniform_int_distribution<int> distribution(
      0,
      static_cast<int>(unoccupied.size()) - 1);
  return unoccupied[distribution(*generator)];
}

double compute_phi2_direct(
    const std::vector<int>& left,
    const std::vector<int>& right,
    const Eigen::MatrixXd& inverse_overlap,
    const Eigen::MatrixXd& ri_factors) {
  const int m = static_cast<int>(left.size());
  double phi2 = 0.0;
  Eigen::MatrixXd block(m, m);
  Eigen::MatrixXd channel(m, m);
  for (int auxiliary = 0; auxiliary < ri_factors.rows(); ++auxiliary) {
    for (int column = 0; column < m; ++column) {
      for (int row = 0; row < m; ++row) {
        block(row, column) =
            ri_factors(auxiliary, packed_pair_index(right[row], left[column]));
      }
    }
    channel.noalias() = block * inverse_overlap;
    const double trace = channel.trace();
    phi2 += trace * trace - (channel * channel).trace();
  }
  return 0.5 * phi2;
}

std::vector<Eigen::MatrixXd> build_channel_matrices(
    const std::vector<int>& left,
    const std::vector<int>& right,
    const Eigen::MatrixXd& inverse_overlap,
    const Eigen::MatrixXd& ri_factors) {
  const int m = static_cast<int>(left.size());
  std::vector<Eigen::MatrixXd> channels;
  channels.reserve(static_cast<std::size_t>(ri_factors.rows()));
  Eigen::MatrixXd block(m, m);
  for (int auxiliary = 0; auxiliary < ri_factors.rows(); ++auxiliary) {
    for (int column = 0; column < m; ++column) {
      for (int row = 0; row < m; ++row) {
        block(row, column) =
            ri_factors(auxiliary, packed_pair_index(right[row], left[column]));
      }
    }
    channels.push_back(block * inverse_overlap);
  }
  return channels;
}

double compute_phi2_from_channels(
    const std::vector<Eigen::MatrixXd>& channels) {
  double phi2 = 0.0;
  for (const Eigen::MatrixXd& channel : channels) {
    const double trace = channel.trace();
    phi2 += trace * trace - (channel * channel).trace();
  }
  return 0.5 * phi2;
}

double compute_phi2_from_channel_stats(
    const std::vector<double>& traces,
    const std::vector<double>& square_traces) {
  double phi2 = 0.0;
  for (std::size_t channel_index = 0; channel_index < traces.size();
       ++channel_index) {
    phi2 += traces[channel_index] * traces[channel_index] -
            square_traces[channel_index];
  }
  return 0.5 * phi2;
}

void update_channel_matrices_rank1_row(
    const std::vector<int>& left,
    int slot,
    int old_orbital,
    int new_orbital,
    const Eigen::MatrixXd& inverse_old,
    const Eigen::RowVectorXd& row_delta_times_inverse,
    double eta,
    const Eigen::MatrixXd& ri_factors,
    std::vector<Eigen::MatrixXd>* channels,
    std::vector<double>* traces,
    std::vector<double>* square_traces) {
  const int m = static_cast<int>(left.size());
  const double inv_eta = 1.0 / eta;
  Eigen::RowVectorXd delta_b(m);
  Eigen::RowVectorXd delta_row(m);
  for (int auxiliary = 0; auxiliary < ri_factors.rows(); ++auxiliary) {
    for (int column = 0; column < m; ++column) {
      delta_b(column) =
          ri_factors(auxiliary, packed_pair_index(new_orbital, left[column])) -
          ri_factors(auxiliary, packed_pair_index(old_orbital, left[column]));
    }
    delta_row.noalias() = delta_b * inverse_old;
    Eigen::MatrixXd& channel = (*channels)[auxiliary];
    const Eigen::VectorXd channel_column = channel.col(slot);

    // The row replacement changes the RI channel by two rank-1 terms:
    // dM = e_slot * delta_row - eta^{-1} * v * row_delta_times_inverse,
    // where v is the old selected column plus the direct row-change correction.
    Eigen::VectorXd correction_column = channel_column;
    correction_column(slot) += delta_row(slot);
    const double trace_rank1_row = delta_row(slot);
    const double trace_rank1_inverse =
        -inv_eta * row_delta_times_inverse.dot(correction_column);
    const double delta_trace = trace_rank1_row + trace_rank1_inverse;

    const double trace_channel_delta =
        delta_row.dot(channel_column) -
        inv_eta *
            (row_delta_times_inverse * channel * correction_column)(0);
    const double delta_square_trace =
        trace_rank1_row * trace_rank1_row +
        2.0 * delta_row.dot(-inv_eta * correction_column) *
            row_delta_times_inverse(slot) +
        trace_rank1_inverse * trace_rank1_inverse;
    (*traces)[auxiliary] += delta_trace;
    (*square_traces)[auxiliary] +=
        2.0 * trace_channel_delta + delta_square_trace;

    channel.noalias() -=
        inv_eta * channel_column * row_delta_times_inverse;
    channel.row(slot).noalias() +=
        delta_row -
        inv_eta * delta_row(slot) * row_delta_times_inverse;
  }
}

BenchmarkResult run_case(
    int m,
    const Options& options,
    std::mt19937* generator) {
  BenchmarkResult result;
  result.n_electrons = m;
  result.n_active_orbitals = 2 * m;
  result.n_auxiliary_functions = options.n_auxiliary_functions;

  const Eigen::MatrixXd active_overlap =
      make_random_overlap(result.n_active_orbitals, generator);
  const Eigen::MatrixXd ri_factors =
      make_random_ri_factors(
          result.n_auxiliary_functions,
          result.n_active_orbitals,
          generator);

  const std::vector<int> left = make_initial_occupation(m);
  std::vector<int> right = make_initial_occupation(m);
  Eigen::MatrixXd overlap = build_overlap_block(active_overlap, left, right);
  Eigen::MatrixXd inverse_overlap;
  double determinant = 0.0;
  if (!resolve_regular_overlap(
          overlap,
          options.singular_threshold,
          &inverse_overlap,
          &determinant)) {
    throw std::runtime_error("initial synthetic overlap is singular");
  }

  std::vector<Eigen::MatrixXd> channels =
      build_channel_matrices(left, right, inverse_overlap, ri_factors);
  std::vector<double> channel_traces(channels.size());
  std::vector<double> channel_square_traces(channels.size());
  for (std::size_t channel_index = 0; channel_index < channels.size();
       ++channel_index) {
    channel_traces[channel_index] = channels[channel_index].trace();
    channel_square_traces[channel_index] =
        (channels[channel_index] * channels[channel_index]).trace();
  }

  std::uniform_int_distribution<int> slot_distribution(0, m - 1);
  for (int step = 0; step < options.n_steps; ++step) {
    const int slot = slot_distribution(*generator);
    const int old_orbital = right[slot];
    const int new_orbital =
        choose_new_orbital(right, result.n_active_orbitals, generator);
    std::vector<int> right_new = right;
    right_new[slot] = new_orbital;

    Eigen::MatrixXd overlap_new = overlap;
    for (int column = 0; column < m; ++column) {
      overlap_new(slot, column) =
          active_overlap(new_orbital, left[column]);
    }

    const auto direct_overlap_start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd inverse_direct;
    double determinant_direct = 0.0;
    const bool direct_regular = resolve_regular_overlap(
        overlap_new,
        options.singular_threshold,
        &inverse_direct,
        &determinant_direct);
    const auto direct_overlap_end = std::chrono::high_resolution_clock::now();
    result.direct_overlap_seconds +=
        std::chrono::duration<double>(
            direct_overlap_end - direct_overlap_start)
            .count();
    if (!direct_regular) {
      ++result.n_resets;
      continue;
    }

    const auto direct_phi2_start = std::chrono::high_resolution_clock::now();
    const double phi2_direct =
        compute_phi2_direct(left, right_new, inverse_direct, ri_factors);
    const auto direct_phi2_end = std::chrono::high_resolution_clock::now();
    result.direct_phi2_seconds +=
        std::chrono::duration<double>(direct_phi2_end - direct_phi2_start).count();

    const auto rank1_overlap_start = std::chrono::high_resolution_clock::now();
    Eigen::RowVectorXd row_delta(m);
    for (int column = 0; column < m; ++column) {
      row_delta(column) =
          active_overlap(new_orbital, left[column]) -
          active_overlap(old_orbital, left[column]);
    }
    const Eigen::VectorXd selected_inverse_column =
        inverse_overlap.col(slot);
    const Eigen::RowVectorXd row_delta_times_inverse =
        row_delta * inverse_overlap;
    const double eta = 1.0 + row_delta_times_inverse(slot);
    if (std::abs(eta) <= options.singular_threshold) {
      inverse_overlap = inverse_direct;
      determinant = determinant_direct;
      overlap = overlap_new;
      right = std::move(right_new);
      channels = build_channel_matrices(left, right, inverse_overlap, ri_factors);
      for (std::size_t channel_index = 0; channel_index < channels.size();
           ++channel_index) {
        channel_traces[channel_index] = channels[channel_index].trace();
        channel_square_traces[channel_index] =
            (channels[channel_index] * channels[channel_index]).trace();
      }
      ++result.n_resets;
      continue;
    }
    Eigen::MatrixXd inverse_updated =
        inverse_overlap -
        (selected_inverse_column * row_delta_times_inverse) / eta;
    const double determinant_updated = determinant * eta;
    const auto rank1_overlap_end = std::chrono::high_resolution_clock::now();
    result.rank1_overlap_seconds +=
        std::chrono::duration<double>(
            rank1_overlap_end - rank1_overlap_start)
            .count();

    const auto update_phi2_start = std::chrono::high_resolution_clock::now();
    update_channel_matrices_rank1_row(
        left,
        slot,
        old_orbital,
        new_orbital,
        inverse_overlap,
        row_delta_times_inverse,
        eta,
        ri_factors,
        &channels,
        &channel_traces,
        &channel_square_traces);
    const double phi2_updated =
        compute_phi2_from_channel_stats(
            channel_traces,
            channel_square_traces);
    const auto update_phi2_end = std::chrono::high_resolution_clock::now();
    result.update_phi2_seconds +=
        std::chrono::duration<double>(update_phi2_end - update_phi2_start).count();

    result.max_inverse_abs_diff = std::max(
        result.max_inverse_abs_diff,
        (inverse_updated - inverse_direct).cwiseAbs().maxCoeff());
    result.max_det_abs_diff = std::max(
        result.max_det_abs_diff,
        std::abs(determinant_updated - determinant_direct));
    result.max_phi2_abs_diff = std::max(
        result.max_phi2_abs_diff,
        std::abs(phi2_updated - phi2_direct));
    result.direct_phi2_checksum += phi2_direct;
    result.update_phi2_checksum += phi2_updated;

    inverse_overlap = std::move(inverse_updated);
    determinant = determinant_updated;
    overlap = std::move(overlap_new);
    right = std::move(right_new);
    ++result.n_steps;
  }

  return result;
}

void print_result(const BenchmarkResult& result) {
  const double overlap_speedup =
      result.rank1_overlap_seconds > 0.0
          ? result.direct_overlap_seconds / result.rank1_overlap_seconds
          : 0.0;
  const double phi2_speedup =
      result.update_phi2_seconds > 0.0
          ? result.direct_phi2_seconds / result.update_phi2_seconds
          : 0.0;
  std::cout << result.n_electrons
            << ' ' << result.n_active_orbitals
            << ' ' << result.n_auxiliary_functions
            << ' ' << result.n_steps
            << ' ' << result.n_resets
            << ' ' << result.direct_overlap_seconds
            << ' ' << result.rank1_overlap_seconds
            << ' ' << overlap_speedup
            << ' ' << result.direct_phi2_seconds
            << ' ' << result.update_phi2_seconds
            << ' ' << phi2_speedup
            << ' ' << result.max_inverse_abs_diff
            << ' ' << result.max_det_abs_diff
            << ' ' << result.max_phi2_abs_diff
            << ' ' << std::abs(
                    result.direct_phi2_checksum - result.update_phi2_checksum)
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);
    std::mt19937 generator(options.seed);
    std::cout << std::setprecision(15);
    std::cout
        << "m n_active n_aux steps resets "
        << "direct_overlap_s rank1_overlap_s overlap_speedup "
        << "direct_phi2_s update_phi2_s phi2_speedup "
        << "max_inv_abs_diff max_det_abs_diff max_phi2_abs_diff "
        << "phi2_checksum_abs_diff\n";
    for (int m = options.min_m; m <= options.max_m; ++m) {
      print_result(run_case(m, options, &generator));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
