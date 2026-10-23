#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/selected_coefficients.hpp"
#include "vbscf/structures/expansion/types.hpp"

namespace {

using Strings = std::vector<std::vector<int>>;
using Pairing = std::vector<std::pair<int, int>>;
using Term = xmvb::vb::StructureExpansionTerm;
using Terms = std::vector<std::vector<Term>>;
using Clock = std::chrono::steady_clock;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void append_combinations(int n, int k, int first, std::vector<int>* occupied,
                         Strings* strings) {
  if (static_cast<int>(occupied->size()) == k) {
    strings->push_back(*occupied);
    return;
  }
  const int remaining = k - static_cast<int>(occupied->size());
  for (int orbital = first; orbital <= n - remaining; ++orbital) {
    occupied->push_back(orbital);
    append_combinations(n, k, orbital + 1, occupied, strings);
    occupied->pop_back();
  }
}

Strings complete_strings(int n, int k) {
  Strings result;
  std::vector<int> occupied;
  append_combinations(n, k, 0, &occupied, &result);
  return result;
}

std::vector<Pairing> rumer_pairings(int first, int past_last) {
  if (first == past_last) return {Pairing{}};
  std::vector<Pairing> result;
  for (int mate = first + 1; mate < past_last; mate += 2) {
    const auto inside = rumer_pairings(first + 1, mate);
    const auto outside = rumer_pairings(mate + 1, past_last);
    for (const auto& left : inside) {
      for (const auto& right : outside) {
        Pairing pairing{{first, mate}};
        pairing.insert(pairing.end(), left.begin(), left.end());
        pairing.insert(pairing.end(), right.begin(), right.end());
        result.push_back(std::move(pairing));
      }
    }
  }
  return result;
}

std::uint64_t occupation_mask(const std::vector<int>& occupied) {
  std::uint64_t mask = 0;
  for (int orbital : occupied) mask |= std::uint64_t{1} << orbital;
  return mask;
}

struct Expansion {
  Terms terms;
  Strings product_alpha;
  Strings product_beta;
};

Expansion covalent_rumer_terms(int n, const Strings& alpha,
                               const Strings& beta,
                               const std::vector<Pairing>& pairings) {
  const int n_beta = static_cast<int>(beta.size());
  std::map<std::uint64_t, int> alpha_index, beta_index;
  for (int i = 0; i < static_cast<int>(alpha.size()); ++i)
    alpha_index.emplace(occupation_mask(alpha[i]), i);
  for (int i = 0; i < n_beta; ++i)
    beta_index.emplace(occupation_mask(beta[i]), i);

  std::map<int, std::vector<Term>> sparse_terms;
  const double scale = std::pow(2.0, -0.5 * (n / 2));
  for (int structure = 0; structure < static_cast<int>(pairings.size());
       ++structure) {
    const auto& pairing = pairings[structure];
    require(static_cast<int>(pairing.size()) == n / 2,
            "Rumer pairing has the wrong number of bonds");
    for (int choice = 0; choice < (1 << (n / 2)); ++choice) {
      std::vector<int> creators;
      std::uint64_t alpha_mask = 0, beta_mask = 0;
      int sign = 1;
      for (int bond = 0; bond < n / 2; ++bond) {
        const auto [a, b] = pairing[bond];
        const bool reverse = (choice & (1 << bond)) != 0;
        const int alpha_site = reverse ? b : a;
        const int beta_site = reverse ? a : b;
        alpha_mask |= std::uint64_t{1} << alpha_site;
        beta_mask |= std::uint64_t{1} << beta_site;
        creators.push_back(a == alpha_site ? a : n + a);
        creators.push_back(b == alpha_site ? b : n + b);
        if (reverse) sign = -sign;
      }
      for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
          if (creators[i] > creators[j]) sign = -sign;
      const int product = alpha_index.at(alpha_mask) * n_beta +
                          beta_index.at(beta_mask);
      sparse_terms[product].push_back({structure, sign * scale});
    }
  }

  Expansion expansion;
  expansion.terms.reserve(sparse_terms.size());
  expansion.product_alpha.reserve(sparse_terms.size());
  expansion.product_beta.reserve(sparse_terms.size());
  for (auto& [product, terms] : sparse_terms) {
    expansion.product_alpha.push_back(alpha[product / n_beta]);
    expansion.product_beta.push_back(beta[product % n_beta]);
    expansion.terms.push_back(std::move(terms));
  }
  return expansion;
}

struct Fixture {
  int n = 12;
  Strings alpha, beta;
  std::vector<double> overlap;
  Eigen::MatrixXd one_electron;
  xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
};

Fixture make_fixture() {
  Fixture f;
  f.alpha = complete_strings(f.n, f.n / 2);
  f.beta = complete_strings(f.n, f.n / 2);
  Eigen::MatrixXd upper = Eigen::MatrixXd::Identity(f.n, f.n);
  for (int j = 0; j < f.n; ++j)
    for (int i = 0; i < j; ++i)
      upper(i, j) = 0.004 * std::cos(0.31 * (i + 2) * (j + 1));
  const Eigen::MatrixXd metric = upper.transpose() * upper;
  f.overlap.assign(metric.data(), metric.data() + metric.size());
  f.one_electron.resize(f.n, f.n);
  for (int j = 0; j < f.n; ++j) {
    for (int i = 0; i <= j; ++i) {
      const double value = i == j ? -0.8 + 0.06 * i
                                  : 0.015 * std::sin(0.19 * (i + 1) * (j + 3));
      f.one_electron(i, j) = value;
      f.one_electron(j, i) = value;
    }
  }
  const int n_pairs = xmvb::integrals::packed_active_pair_count(f.n);
  Eigen::MatrixXd factor(4, n_pairs);
  for (int j = 0; j < n_pairs; ++j)
    for (int i = 0; i < factor.rows(); ++i)
      factor(i, j) = 0.018 * std::cos(0.13 * (i + 2) * (j + 1));
  const Eigen::MatrixXd kernel = factor.transpose() * factor;
  auto& packed = f.two_electron.packed_active_two_electron_integrals;
  packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(f.n));
  for (int j = 0; j < n_pairs; ++j)
    for (int i = 0; i <= j; ++i)
      packed[j * (j + 1) / 2 + i] =
          kernel(i, j);
  return f;
}

// Fixture-only emulation of complete orthogonal spin tables on the baseline.
// The physical determinant list in the expansion remains sparse in both builds.
xmvb::vb::SameSpinPairCacheContext full_carrier_topology(
    const Fixture& f, const Expansion& expansion) {
  auto cache = xmvb::vb::build_same_spin_pair_topology(
      f.alpha, f.beta, f.n);
  std::map<std::vector<int>, int> alpha_ids, beta_ids;
  for (int i = 0; i < static_cast<int>(f.alpha.size()); ++i)
    alpha_ids.emplace(f.alpha[i], i);
  for (int i = 0; i < static_cast<int>(f.beta.size()); ++i)
    beta_ids.emplace(f.beta[i], i);
  cache.alpha_reuse_table.determinant_to_unique_id.clear();
  cache.beta_reuse_table.determinant_to_unique_id.clear();
  // The complete-table seed pairs each string with itself; the physical
  // covalent rows below instead pair complementary occupations.
  cache.close_shell_diagonal_reuses_same_spin_pair_cache=false;
  for (std::size_t i = 0; i < expansion.terms.size(); ++i) {
    cache.alpha_reuse_table.determinant_to_unique_id.push_back(
        alpha_ids.at(expansion.product_alpha[i]));
    cache.beta_reuse_table.determinant_to_unique_id.push_back(
        beta_ids.at(expansion.product_beta[i]));
  }
  return cache;
}

Eigen::MatrixXd test_vectors(int n) {
  Eigen::MatrixXd c(n, 1);
  for (int row = 0; row < n; ++row)
    c(row, 0) = std::cos(0.17 * (row + 1) * 2) +
                0.2 * std::sin(0.11 * (row + 3));
  return c / std::sqrt(n);
}

struct OutputBlock {
  const char* name;
  std::vector<double> values;
  int rows;
  int columns;
  double checksum;
  double norm;
};

template <class Derived>
OutputBlock block(const char* name, const Eigen::MatrixBase<Derived>& x) {
  const int rows = static_cast<int>(x.rows());
  const int columns = static_cast<int>(x.cols());
  OutputBlock result{name, {}, rows, columns, 0.0, 0.0};
  for (int j = 0; j < columns; ++j) {
    for (int i = 0; i < rows; ++i) {
      const double value = x(i, j);
      result.values.push_back(value);
      result.checksum += value;
      result.norm += value * value;
    }
  }
  result.norm = std::sqrt(result.norm);
  return result;
}

double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void run(int count, const std::string& binary_path) {
  require(count == 1 || count == 3 || count == 20 || count == 132,
          "selected structure count must be 1, 3, 20, or 132");
  const Fixture fixture = make_fixture();
  const auto pairings = rumer_pairings(0, fixture.n);
  const Expansion expansion = covalent_rumer_terms(
      fixture.n, fixture.alpha, fixture.beta,
      std::vector<Pairing>(pairings.begin(), pairings.begin() + count));
  const auto cache = full_carrier_topology(fixture, expansion);
  const Eigen::MatrixXd c = test_vectors(count);
  constexpr double energy = -5.3;

  xmvb::vb::StructureExpansion determinant_data;
  determinant_data.n_structures = count;
  determinant_data.alpha_strings = fixture.alpha;
  determinant_data.beta_strings = fixture.beta;
  std::map<std::vector<int>, int> alpha_ids, beta_ids;
  for (int i = 0; i < static_cast<int>(fixture.alpha.size()); ++i)
    alpha_ids.emplace(fixture.alpha[i], i);
  for (int i = 0; i < static_cast<int>(fixture.beta.size()); ++i)
    beta_ids.emplace(fixture.beta[i], i);
  determinant_data.offsets.push_back(0);
  for (std::size_t pair = 0; pair < expansion.terms.size(); ++pair) {
    determinant_data.pair_alpha.push_back(alpha_ids.at(expansion.product_alpha[pair]));
    determinant_data.pair_beta.push_back(beta_ids.at(expansion.product_beta[pair]));
    determinant_data.terms.insert(determinant_data.terms.end(),
                                  expansion.terms[pair].begin(),
                                  expansion.terms[pair].end());
    determinant_data.offsets.push_back(determinant_data.terms.size());
  }

  const auto total_start = Clock::now();
  const auto construct_start = Clock::now();
  const xmvb::vb::StructureAction action(
      expansion.terms, count, cache, fixture.overlap, fixture.one_electron,
      fixture.two_electron, fixture.n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto construct_end = Clock::now();
  const auto action_images = action.apply(c);
  const auto action_end = Clock::now();

  xmvb::vb::StructureAdjointState accepted;
  const auto accepted_start = Clock::now();
#ifdef XMVB_GITHUB_BASELINE
  const auto selected =
      xmvb::vb::build_selected_state_determinant_matrices_from_selected_columns(
          determinant_data, c, {0}, {1.0}, cache);
  accepted = action.prepare_active_adjoint(selected, {energy});
#else
  accepted = action.prepare_active_adjoint(c, {1.0}, {energy});
#endif
  const auto accepted_gradient = action.active_integral_adjoint(accepted);
  const auto accepted_end = Clock::now();

  Eigen::MatrixXd ds(fixture.n, fixture.n), dh(fixture.n, fixture.n);
  for (int j = 0; j < fixture.n; ++j)
    for (int i = 0; i <= j; ++i) {
      ds(i, j) = ds(j, i) = 0.0004 * std::cos(0.13 * (i + 2) * (j + 1));
      dh(i, j) = dh(j, i) = 0.0007 * std::sin(0.17 * (i + 1) * (j + 3));
    }
  const int pairs = xmvb::integrals::packed_active_pair_count(fixture.n);
  std::vector<double> dg(
      fixture.two_electron.packed_active_two_electron_integrals.size());
  for (int i = 0; i < static_cast<int>(dg.size()); ++i)
    dg[i] = 0.00001 * std::cos(0.19 * (i + 1));
  const std::vector<double> overlap_direction(ds.data(),
                                               ds.data() + ds.size());
  const std::vector<double> one_direction(dh.data(), dh.data() + dh.size());

  xmvb::vb::StructureIntegralDirection direct_ci_direction;
  const auto direction_start = Clock::now();
  const auto direction_images = action.apply_integral_direction(
      accepted, overlap_direction, one_direction, dg, &direct_ci_direction);
  const auto direction_end = Clock::now();
  const auto reverse_start = Clock::now();
  const auto reverse = action.active_integral_adjoint_direction(
      accepted, nullptr, nullptr, overlap_direction, one_direction, dg,
      &direct_ci_direction, true);
  const auto reverse_end = Clock::now();
  const auto total_end = Clock::now();

  const std::vector<OutputBlock> blocks = {
      block("action_hamiltonian", action_images.hamiltonian),
      block("action_overlap", action_images.overlap),
      block("accepted_gradient_overlap", accepted_gradient.overlap),
      block("accepted_gradient_one_electron", accepted_gradient.one_electron),
      block("accepted_gradient_pair_kernel", accepted_gradient.pair_kernel),
      block("direction_hamiltonian", direction_images.hamiltonian),
      block("direction_overlap", direction_images.overlap),
      block("reverse_overlap", reverse.overlap),
      block("reverse_one_electron", reverse.one_electron),
      block("reverse_pair_kernel", reverse.pair_kernel)};

  std::size_t value_count = 0;
  double aggregate_checksum = 0.0;
  for (const auto& b : blocks) {
    value_count += b.values.size();
    aggregate_checksum += b.checksum;
  }
  if (!binary_path.empty()) {
    std::ofstream output(binary_path, std::ios::binary | std::ios::trunc);
    require(output.good(), "could not open numerical dump");
    for (const auto& b : blocks)
      output.write(reinterpret_cast<const char*>(b.values.data()),
                   static_cast<std::streamsize>(b.values.size() * sizeof(double)));
    require(output.good(), "could not write numerical dump");
  }

  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  std::cout << std::setprecision(17)
            << "variant="
#ifdef XMVB_GITHUB_BASELINE
            << "github_baseline\n"
#else
            << "current\n"
#endif
            << "n_orbitals=12\nselected_structures=" << count
            << "\nalpha_strings=" << fixture.alpha.size()
            << "\nbeta_strings=" << fixture.beta.size()
            << "\nphysical_products=" << expansion.terms.size()
            << "\nconstruction_seconds="
            << seconds(construct_start, construct_end)
            << "\naction_seconds=" << seconds(construct_end, action_end)
            << "\naccepted_prepare_gradient_seconds="
            << seconds(accepted_start, accepted_end)
            << "\norbital_direction_seconds="
            << seconds(direction_start, direction_end)
            << "\ncached_reverse_seconds=" << seconds(reverse_start, reverse_end)
            << "\ntotal_seconds=" << seconds(total_start, total_end)
            << "\nmax_rss_kib=" << usage.ru_maxrss
            << "\narray_value_count=" << value_count
            << "\narray_checksum_sum=" << aggregate_checksum << '\n';
  for (const auto& b : blocks)
    std::cout << "array." << b.name << ".count=" << b.values.size()
              << "\narray." << b.name << ".logical_rows=" << b.rows
              << "\narray." << b.name << ".logical_columns=" << b.columns
              << "\narray." << b.name << ".layout=column_major"
              << "\narray." << b.name << ".sum=" << b.checksum
              << "\narray." << b.name << ".norm=" << b.norm << '\n';
  std::cout << "binary_path=" << binary_path << '\n'
            << "fixture_complete_spin_carrier=1\n"
            << "fixture_only_carrier_completion=1\n"
            << "baseline_default_admission_claim=0\n"
            << "dg_full_pairs=" << pairs << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2 || argc == 3,
            "usage: benchmark_product12 STRUCTURE_COUNT [NUMERICAL_DUMP.bin]");
    run(std::stoi(argv[1]), argc == 3 ? argv[2] : "");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark_product12: " << error.what() << '\n';
    return 1;
  }
}
