#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/resource.h>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include "core/pair_index.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/factors.hpp"
#include "mcscf/ci/product_support.hpp"
#include "mcscf/ci/sigma_plan.hpp"
#include "mcscf/ci/channel_product.hpp"
#include "mcscf/ci/sigma_gather.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/assembly/input_columns.hpp"
#include "vbscf/structures/assembly/states.hpp"
#include "vbscf/structures/assembly/column_tiles.hpp"
#include "vbscf/structures/assembly/sources.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/structures/overlap/representation.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "mcscf/ci/product_paths.hpp"

namespace {

using Strings = std::vector<std::vector<int>>;
using Pairing = std::vector<std::pair<int, int>>;
using Terms = std::vector<std::vector<xmvb::vb::StructureExpansionTerm>>;
using Clock = std::chrono::steady_clock;

struct CovalentExpansion {
  Terms terms;
  Strings product_alpha;
  Strings product_beta;
};

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

// A first site's mate splits a noncrossing matching into two even intervals.
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

int occupation_mask(const std::vector<int>& occupied) {
  int mask = 0;
  for (int orbital : occupied) mask |= 1 << orbital;
  return mask;
}

// Each bond is (a_alpha^+ b_beta^+ - a_beta^+ b_alpha^+)/sqrt(2).
// Count the creator swaps needed to reach the alpha-then-beta determinant
// convention. This matters even when every site is singly occupied.
CovalentExpansion covalent_rumer_terms(
    int n, const Strings& alpha, const Strings& beta,
    const std::vector<Pairing>& pairings) {
  const int n_beta = static_cast<int>(beta.size());
  std::map<int, std::vector<xmvb::vb::StructureExpansionTerm>> sparse_terms;
  std::map<int, int> alpha_index;
  std::map<int, int> beta_index;
  for (int i = 0; i < static_cast<int>(alpha.size()); ++i)
    alpha_index.emplace(occupation_mask(alpha[i]), i);
  for (int i = 0; i < n_beta; ++i)
    beta_index.emplace(occupation_mask(beta[i]), i);
  const double scale = std::pow(2.0, -0.5 * (n / 2));
  for (int structure = 0; structure < static_cast<int>(pairings.size());
       ++structure) {
    const auto& pairing = pairings[structure];
    require(static_cast<int>(pairing.size()) == n / 2,
            "Rumer pairing has the wrong number of bonds");
    for (int choice = 0; choice < (1 << (n / 2)); ++choice) {
      std::vector<int> creators;
      int alpha_mask = 0;
      int beta_mask = 0;
      int sign = 1;
      for (int bond = 0; bond < n / 2; ++bond) {
        const auto [a, b] = pairing[bond];
        const bool reverse = (choice & (1 << bond)) != 0;
        const int alpha_site = reverse ? b : a;
        const int beta_site = reverse ? a : b;
        alpha_mask |= 1 << alpha_site;
        beta_mask |= 1 << beta_site;
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
  CovalentExpansion expansion;
  expansion.terms.reserve(sparse_terms.size());
  expansion.product_alpha.reserve(sparse_terms.size());
  expansion.product_beta.reserve(sparse_terms.size());
  for (auto& [product, product_terms] : sparse_terms) {
    expansion.product_alpha.push_back(alpha[product / n_beta]);
    expansion.product_beta.push_back(beta[product % n_beta]);
    expansion.terms.push_back(std::move(product_terms));
  }
  return expansion;
}

struct Fixture {
  int n = 0;
  Strings alpha;
  Strings beta;
  std::vector<double> overlap;
  Eigen::MatrixXd one_electron;
  xmvb::integrals::ActiveSpaceTwoElectronResult two_electron;
};

Fixture make_fixture(int n, int n_alpha, int n_beta) {
  Fixture fixture;
  fixture.n = n;
  fixture.alpha = complete_strings(n, n_alpha);
  fixture.beta = complete_strings(n, n_beta);
  Eigen::MatrixXd upper = Eigen::MatrixXd::Identity(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < j; ++i)
      upper(i, j) = 0.004 * std::cos(0.31 * (i + 2) * (j + 1));
  const Eigen::MatrixXd metric = upper.transpose() * upper;
  fixture.overlap.assign(metric.data(), metric.data() + metric.size());
  fixture.one_electron.resize(n, n);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i <= j; ++i) {
      const double value = i == j ? -0.8 + 0.06 * i
                                  : 0.015 * std::sin(0.19 * (i + 1) * (j + 3));
      fixture.one_electron(i, j) = value;
      fixture.one_electron(j, i) = value;
    }
  }
  const int n_pairs = xmvb::integrals::packed_active_pair_count(n);
  Eigen::MatrixXd factor(4, n_pairs);
  for (int j = 0; j < n_pairs; ++j)
    for (int i = 0; i < factor.rows(); ++i)
      factor(i, j) = 0.018 * std::cos(0.13 * (i + 2) * (j + 1));
  const Eigen::MatrixXd kernel = factor.transpose() * factor;
  auto& packed = fixture.two_electron.packed_active_two_electron_integrals;
  packed.resize(xmvb::integrals::packed_active_two_electron_integral_count(n));
  for (int j = 0; j < n_pairs; ++j)
    for (int i = 0; i <= j; ++i)
      packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(i, j)] =
          kernel(i, j);
  return fixture;
}

xmvb::vb::SameSpinPairCacheContext topology(
    const Fixture& fixture, const Strings& product_alpha,
    const Strings& product_beta) {
  return xmvb::vb::build_orthogonal_spin_topology(
      product_alpha, product_beta, fixture.n);
}

std::pair<Strings, Strings> full_products(const Fixture& fixture) {
  Strings product_alpha;
  Strings product_beta;
  const std::size_t products = fixture.alpha.size() * fixture.beta.size();
  product_alpha.reserve(products);
  product_beta.reserve(products);
  for (const auto& a : fixture.alpha)
    for (const auto& b : fixture.beta) {
      product_alpha.push_back(a);
      product_beta.push_back(b);
    }
  return {std::move(product_alpha), std::move(product_beta)};
}

Eigen::MatrixXd test_vectors(int n_structures, int width) {
  Eigen::MatrixXd vectors(n_structures, width);
  for (int column = 0; column < width; ++column)
    for (int row = 0; row < n_structures; ++row)
      vectors(row, column) =
          std::cos(0.17 * (row + 1) * (column + 2)) +
          0.2 * std::sin(0.11 * (row + 3) * (column + 1));
  return vectors;
}

// Reference independently expands C into the complete orthonormal carrier,
// uses public direct-CI sigma, then applies exterior adjoints and C^T.
xmvb::vb::StructureActionResult reference_images(
    const Fixture& fixture, const Terms& terms,
    const Strings& product_alpha, const Strings& product_beta,
    const Eigen::MatrixXd& vectors) {
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, fixture.n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(
      fixture.alpha, fixture.beta, integrals);
  const xmvb::vb::ExteriorOrbitalTransform alpha_transform(
      fixture.alpha, integrals.orbital_transform);
  const xmvb::vb::ExteriorOrbitalTransform beta_transform(
      fixture.beta, integrals.orbital_transform);
  std::map<int, int> alpha_index;
  std::map<int, int> beta_index;
  for (int i = 0; i < static_cast<int>(fixture.alpha.size()); ++i)
    alpha_index.emplace(occupation_mask(fixture.alpha[i]), i);
  for (int i = 0; i < static_cast<int>(fixture.beta.size()); ++i)
    beta_index.emplace(occupation_mask(fixture.beta[i]), i);
  const int n_beta = static_cast<int>(fixture.beta.size());
  Eigen::MatrixXd carrier = Eigen::MatrixXd::Zero(
      fixture.alpha.size(), vectors.cols() * n_beta);
  for (int product = 0; product < static_cast<int>(terms.size()); ++product) {
    const int a = alpha_index.at(occupation_mask(product_alpha[product]));
    const int b = beta_index.at(occupation_mask(product_beta[product]));
    for (const auto& term : terms[product])
      for (int column = 0; column < vectors.cols(); ++column)
        carrier(a, column * n_beta + b) +=
            term.coefficient * vectors(term.structure_index, column);
  }
  alpha_transform.apply_left(&carrier);
  for (int column = 0; column < vectors.cols(); ++column) {
    auto block = carrier.middleCols(column * n_beta, n_beta);
    beta_transform.apply_right_block(block);
  }
  Eigen::MatrixXd hamiltonian = sigma.apply(carrier);
  for (int column = 0; column < vectors.cols(); ++column) {
    auto h = hamiltonian.middleCols(column * n_beta, n_beta);
    auto s = carrier.middleCols(column * n_beta, n_beta);
    beta_transform.apply_adjoint_right_block(h);
    beta_transform.apply_adjoint_right_block(s);
  }
  alpha_transform.apply_adjoint_left(&hamiltonian);
  alpha_transform.apply_adjoint_left(&carrier);
  xmvb::vb::StructureActionResult result{
      Eigen::MatrixXd::Zero(vectors.rows(), vectors.cols()),
      Eigen::MatrixXd::Zero(vectors.rows(), vectors.cols())};
  for (int product = 0; product < static_cast<int>(terms.size()); ++product) {
    const int a = alpha_index.at(occupation_mask(product_alpha[product]));
    const int b = beta_index.at(occupation_mask(product_beta[product]));
    for (const auto& term : terms[product])
      for (int column = 0; column < vectors.cols(); ++column) {
        result.hamiltonian(term.structure_index, column) +=
            term.coefficient * hamiltonian(a, column * n_beta + b);
        result.overlap(term.structure_index, column) +=
            term.coefficient * carrier(a, column * n_beta + b);
      }
  }
  return result;
}

void check_images(const xmvb::vb::StructureActionResult& actual,
                  const xmvb::vb::StructureActionResult& expected) {
  const double h_error = (actual.hamiltonian - expected.hamiltonian).norm();
  const double s_error = (actual.overlap - expected.overlap).norm();
  require(h_error <= 2e-10 * std::max(1.0, expected.hamiltonian.norm()),
          "structure Hamiltonian differs from direct-CI reference");
  require(s_error <= 2e-11 * std::max(1.0, expected.overlap.norm()),
          "structure overlap differs from carrier reference");
}

/** @brief Compare fused fixed states against independent action/adjoint calls. */
void check_fixed_states(const xmvb::vb::StructureAction& action) {
  const int count=std::min(2,action.n_structures());
  const Eigen::MatrixXd c=test_vectors(action.n_structures(),count);
  const std::vector<double> weights(count,1.0/count);
  const auto actual=xmvb::vb::evaluate_fixed_states(action,c,weights);
  const auto images=action.apply(c);
  const xmvb::vb::SelectedMetric shared_metric(action.apply_overlap(
      Eigen::MatrixXd::Identity(action.n_structures(),action.n_structures())));
  const auto reused=xmvb::vb::evaluate_fixed_states(action,c,weights,&shared_metric);
  const auto& reused_adjoint=reused.integral_adjoint.value();
  require((reused.coefficient_gradient-actual.coefficient_gradient).norm()<3e-11 &&
      (reused.residuals-actual.residuals).norm()<3e-11 &&
      (reused_adjoint.overlap-actual.integral_adjoint->overlap).norm()<3e-11 &&
      (reused_adjoint.one_electron-actual.integral_adjoint->one_electron).norm()<3e-11 &&
      (reused_adjoint.pair_kernel-actual.integral_adjoint->pair_kernel).norm()<3e-11,
      "trial-chart Gram reuse changed fixed-state gradients or integral adjoints");
  for (int s=0;s<count;++s)
    require(std::abs(reused.energies[s]-actual.energies[s])<3e-11,
        "trial-chart Gram reuse changed a fixed-state energy");
  const Eigen::MatrixXd gram=c.transpose()*images.overlap;
  const Eigen::LLT<Eigen::MatrixXd> metric(gram);
  const Eigen::MatrixXd white=metric.matrixU().solve(Eigen::MatrixXd::Identity(count,count));
  const Eigen::MatrixXd projected=white.transpose()*(c.transpose()*images.hamiltonian)*white;
  const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(projected);
  const Eigen::MatrixXd transform=white*spectrum.eigenvectors();
  const Eigen::MatrixXd normalized=c*transform;
  std::vector<double> energies(count);
  for (int s=0;s<count;++s)
    energies[s]=normalized.col(s).dot(images.hamiltonian*transform.col(s));
  const auto prepared=action.prepare_active_adjoint(normalized,weights,energies);
  const auto expected=action.active_integral_adjoint(prepared);
  const auto& adjoint=actual.integral_adjoint.value();
  require((adjoint.overlap-expected.overlap).norm()<3e-11 &&
      (adjoint.one_electron-expected.one_electron).norm()<3e-11 &&
      (adjoint.pair_kernel-expected.pair_kernel).norm()<3e-11,
      "fused fixed-state adjoint lost normalization or off-diagonal state multipliers");
  const Eigen::MatrixXd residual=(images.hamiltonian*transform)-
      (images.overlap*transform)*Eigen::Map<const Eigen::VectorXd>(energies.data(),count).asDiagonal();
  require((actual.coefficient_gradient-(2.0/count)*residual*transform.transpose()).norm()<3e-11,
      "fused fixed-state coefficient gradient uses the wrong input frame");
  require((actual.residuals.colwise().norm()-residual.colwise().norm()).norm()<3e-11,
      "fused fixed-state residual differs from separate H/M action");
  // Exercise the optional image consumer independently of Rayleigh normalization.
  xmvb::vb::StructureActionResult fused_images;
  const auto fused=action.prepare_active_adjoint(c,weights,energies,&fused_images);
  xmvb::vb::StructureActionResult known_images;
  const auto known=action.prepare_active_adjoint(c,weights,energies,
      &known_images,&images.overlap);
  check_images(known_images,fused_images);
  require((known.one_electron_gradient-fused.one_electron_gradient).norm()<3e-11 &&
      (known.pair_kernel_gradient-fused.pair_kernel_gradient).norm()<3e-11 &&
      (known.generator_gradient-fused.generator_gradient).norm()<3e-11,
      "known-overlap consumption removed a physical adjoint term");
  check_images(fused_images,images);
  const auto separate=action.prepare_active_adjoint(c,weights,energies);
  require((fused.generator_gradient-separate.generator_gradient).norm()<3e-11 &&
      (fused.one_electron_gradient-separate.one_electron_gradient).norm()<3e-11 &&
      (fused.pair_kernel_gradient-separate.pair_kernel_gradient).norm()<3e-11,
      "requesting shared sigma images changed the accepted adjoint");
}

void check_rumer_case(int n) {
  const Fixture fixture = make_fixture(n, n / 2, n / 2);
  {
    const auto integrals = xmvb::vb::orthogonalize_active_integrals(
        fixture.overlap, fixture.one_electron, fixture.two_electron, n);
    const xmvb::vb::ExteriorOrbitalTransform transform(fixture.alpha, integrals.orbital_transform);
    const std::vector<int> rows{0, transform.dimension() / 3, transform.dimension() - 1};
    auto selected = test_vectors(transform.dimension(), 3), full = selected;
    transform.apply_adjoint_left(&full);
    const auto plan = transform.projection(rows);
    transform.apply_adjoint_left(&selected, plan);
    for (int row : rows)
      require((selected.row(row) - full.row(row)).norm() < 1e-13,
          "causal exterior projection lost a required input path");
  }
  const auto pairings = rumer_pairings(0, n);
  const CovalentExpansion expansion = covalent_rumer_terms(
      n, fixture.alpha, fixture.beta, pairings);
  const int expected = n == 4 ? 2 : 5;
  require(static_cast<int>(pairings.size()) == expected,
          "Rumer pairing count is incorrect");
  int term_count = 0;
  for (const auto& product : expansion.terms) term_count += product.size();
  require(term_count == expected * (1 << (n / 2)),
          "Rumer expansion is not complete");
  require(expansion.terms.size() == fixture.alpha.size(),
          "singly occupied Rumer products were not compacted");
  const auto cache = topology(
      fixture, expansion.product_alpha, expansion.product_beta);
  const xmvb::vb::StructureAction action(
      expansion.terms, expected, cache, fixture.overlap, fixture.one_electron,
      fixture.two_electron, n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  require(action.storage().orthogonal_direct_ci,
          "complete spin strings failed to select direct CI");
  const Eigen::MatrixXd vectors = test_vectors(expected, 3);
  const auto actual = action.apply(vectors);
  require((action.apply_hamiltonian(vectors)-actual.hamiltonian).norm()<
      2e-11*std::max(1.0,actual.hamiltonian.norm()),
      "Hamiltonian-only projection differs from the combined H/M action");
  // A shared budget must also hold one sigma target and its halo, unlike
  // the former column-only budget (which had an independent 64 MiB sigma).
  // Direct LRU eviction is separately exercised by check_input_columns().
  check_images(action.apply_streamed(vectors,16*1024),actual);
  check_images(actual, reference_images(
      fixture, expansion.terms, expansion.product_alpha,
      expansion.product_beta, vectors));
  check_images(action.apply_sources(vectors), actual);
  // Force simultaneous source and channel streaming, including final partial
  // channel blocks. This is an independent order, not the grid's panel budget.
  const std::size_t source_budget = 12 * sizeof(double) *
      (fixture.alpha.size() + fixture.beta.size());
  check_images(action.apply_sources(vectors, source_budget), actual);
  require((actual.overlap.array() * vectors.array()).colwise().sum().minCoeff()
              > 0.0,
          "Rumer structure metric is not positive on test vectors");
}

void check_selected_unequal_spin_case() {
  const Fixture fixture = make_fixture(5, 2, 1);
  Strings product_alpha;
  Strings product_beta;
  for (int a = 0; a < 7; ++a)
    for (int b = 0; b < 4; ++b) {
      product_alpha.push_back(fixture.alpha[a]);
      product_beta.push_back(fixture.beta[b]);
    }
  const int products = static_cast<int>(product_alpha.size());
  constexpr int n_structures = 7;
  Terms terms(products);
  for (int product = 0; product < products; ++product)
    for (int structure = 0; structure < n_structures; ++structure)
      if ((product + 3 * structure) % 11 < 3)
        terms[product].push_back({structure,
            std::sin(0.23 * (product + 1) * (structure + 2))});
  const auto cache = topology(fixture, product_alpha, product_beta);
  const xmvb::vb::StructureAction action(
      terms, n_structures, cache, fixture.overlap, fixture.one_electron,
      fixture.two_electron, fixture.n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  require(action.storage().orthogonal_direct_ci,
          "selected spin strings did not select completed direct CI");
  const Eigen::MatrixXd vectors = test_vectors(n_structures, 4);
  require((action.apply_hamiltonian(vectors)-action.apply(vectors).hamiltonian).norm()<2e-11,
      "unequal-spin Hamiltonian-only projection lost selected products");
  check_images(action.apply(vectors), reference_images(
      fixture, terms, product_alpha, product_beta, vectors));
  check_images(action.apply_streamed(vectors,16*1024), reference_images(
      fixture, terms, product_alpha, product_beta, vectors));
  check_images(action.apply_sources(vectors), action.apply(vectors));
  check_images(action.apply_sources(vectors,
      12 * sizeof(double) * (fixture.alpha.size() + fixture.beta.size())), action.apply(vectors));
}

/** @brief The source quotient keeps even-parity diagonals and both output halves. */
void check_ionic_sources() {
  const auto fixture = make_fixture(4, 2, 2);
  const auto pairings = rumer_pairings(0, 4);
  auto expansion = covalent_rumer_terms(4, fixture.alpha, fixture.beta, {pairings.front()});
  expansion.product_alpha.push_back({2, 3});
  expansion.product_beta.push_back({2, 3});
  expansion.terms.push_back({{1, 0.7}});
  const auto cache = topology(fixture, expansion.product_alpha, expansion.product_beta);
  const xmvb::vb::StructureAction action(expansion.terms, 2, cache,
      fixture.overlap, fixture.one_electron, fixture.two_electron, 4, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto vectors = test_vectors(2, 3);
  const auto reference = reference_images(fixture, expansion.terms,
      expansion.product_alpha, expansion.product_beta, vectors);
  check_images(action.apply_sources(vectors), reference);
  check_images(action.apply(vectors), reference);
}

/** @brief Exhaustive small graphs certify cover size and unique edge ownership. */
void check_source_cover() {
  for (int mask = 1; mask < (1 << 9); ++mask) {
    std::vector<xmvb::mcscf::ProductIndex> products;
    for (int edge = 0; edge < 9; ++edge)
      if (mask & (1 << edge)) products.push_back(edge);
    const xmvb::vb::SpinSources sources(products, 3, 3);
    int minimum = 6;
    for (int cover = 0; cover < (1 << 6); ++cover) {
      bool valid = true;
      for (const int edge : products)
        valid = valid && ((cover & (1 << (edge / 3))) || (cover & (1 << (3 + edge % 3))));
      if (!valid) continue;
      int count = 0;
      for (int bits = cover; bits; bits &= bits - 1) ++count;
      minimum = std::min(minimum, count);
    }
    require(sources.nodes.size() == static_cast<std::size_t>(minimum),
            "source count differs from exhaustive minimum vertex cover");
    std::vector<int> visits(products.size(), 0);
    for (int node = 0; node < static_cast<int>(sources.nodes.size()); ++node)
      for (int offset = sources.offsets[node]; offset < sources.offsets[node + 1]; ++offset) {
        const int group = sources.groups[offset]; const auto edge = products[group];
        const auto& source = sources.nodes[node];
        require((source.beta ? edge % 3 : edge / 3) == source.string,
                "source factor owns an unrelated product");
        ++visits[group];
      }
    require(std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; }),
            "support factorization lost or duplicated an input product");
  }
}

/** @brief The exact merged tangent matches independent full-carrier generators. */
void check_source_tangents() {
  for (const bool unequal : {false,true}) {
    const auto fixture = unequal ? make_fixture(5,2,1) : make_fixture(4,2,2);
    const int na = fixture.alpha.size(), nb = fixture.beta.size();
    const auto integrals = xmvb::vb::orthogonalize_active_integrals(
        fixture.overlap,fixture.one_electron,fixture.two_electron,fixture.n);
    const xmvb::vb::ExteriorOrbitalTransform ta(fixture.alpha,integrals.orbital_transform);
    const xmvb::vb::ExteriorOrbitalTransform tb(fixture.beta,integrals.orbital_transform);
    const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha,fixture.beta,integrals);
    Eigen::MatrixXd generator(fixture.n,fixture.n);
    for (int j = 0; j < generator.cols(); ++j)
      for (int i = 0; i < generator.rows(); ++i)
        generator(i,j) = 0.03*std::sin(0.7*(i+1)*(j+2));
    for (const int parity : {0,1,-1}) {
      if (unequal && parity) continue;
      std::map<int,std::pair<double,double>> entries;
      const auto add = [&](int a,int b) {
        const double c = 0.3+0.02*(a+1)*(b+2), q = std::sin(0.41*(a+2)*(b+1));
        entries[a*nb+b] = {c,q};
        if (parity && a != b) entries[b*nb+a] = {parity*c,parity*q};
      };
      if (parity) {
        for (int b = 2; b < nb; ++b) add(0,b);
        for (int a = 1; a < na-1; ++a) add(a,nb-1);
        if (parity > 0) add(1,1);
      } else {
        for (int b = 1; b < nb; ++b) add(0,b);
        for (int a = 1; a < na; ++a) add(a,0);
      }
      std::vector<xmvb::mcscf::ProductIndex> products;
      Eigen::VectorXd c(entries.size()),q(entries.size());
      int offset = 0;
      for (const auto& [edge,values] : entries) {
        products.push_back(edge); c[offset] = values.first; q[offset++] = values.second;
      }
      const xmvb::vb::SpinSources plan(products,na,nb,parity);
      require(std::any_of(plan.nodes.begin(),plan.nodes.end(),[](auto node) { return node.beta; }) &&
              std::any_of(plan.nodes.begin(),plan.nodes.end(),[](auto node) { return !node.beta; }),
              "tangent test did not exercise both coordinate-anchor orientations");
      const auto reconstruct = [&](const xmvb::mcscf::SpinFactorSource& source,int width) {
        Eigen::MatrixXd value = Eigen::MatrixXd::Zero(na,nb);
        for (int first = 0; first < source.count; first += width) {
          const int count = std::min(width,source.count-first);
          Eigen::MatrixXd a(na,count),b(nb,count);
          source.load(first,count,a,b);
          value.noalias() += a*b.transpose();
        }
        return value;
      };
      for (const bool physical : {false,true}) {
        const auto base = plan.orthogonal_source(products,c,ta,tb,physical);
        const auto response = plan.orthogonal_source(products,q,ta,tb,physical);
        const Eigen::MatrixXd full = reconstruct(base,3);
        for (const bool transpose : {false,true}) {
          const int in_rows = transpose ? na:nb,out_rows = transpose ? nb:na;
          const Eigen::MatrixXd input = test_vectors(in_rows,3);
          Eigen::MatrixXd output = Eigen::MatrixXd::Constant(out_rows,3,0.17);
          Eigen::MatrixXd expected = output;
          if (transpose) expected.noalias() -= 0.63*full.transpose()*input;
          else expected.noalias() -= 0.63*full*input;
          base.contract(transpose,input,output,-0.63);
          require((output-expected).norm() < 4e-12*std::max(1.0,expected.norm()),
                  "support amplitude map lost a transpose, parity weight or unequal spin space");
        }
        const Eigen::MatrixXd orbital = sigma.apply_one_body_generator(reconstruct(base,3),generator);
        const Eigen::MatrixXd coefficient = reconstruct(response,3);
        for (const int mode : {0,1,2}) {
          const auto tangent = plan.tangent_source(products,c,mode == 0 ? nullptr : &q,
              ta,tb,sigma,mode == 1 ? nullptr : &generator,physical);
          const Eigen::MatrixXd expected = mode == 0 ? orbital : mode == 1 ? coefficient :
              Eigen::MatrixXd(orbital+coefficient);
          require(tangent.count == (mode == 1 ? base.count : 2*base.count),
                  "merged source tangent retained a third factor family");
          require(tangent.workspace_rows <= static_cast<std::size_t>(std::max(na,nb)),
                  "merged source tangent needs more than one spin scratch panel");
          for (const int width : {1,3,7,tangent.count})
            require((reconstruct(tangent,width)-expected).norm() <
                    3e-12*std::max(1.0,expected.norm()),
                    "merged tangent changed an oriented generator, coefficient response or parity sign");
        }
      }
      const auto y = plan.orthogonal_source(products,c,ta,tb);
      const auto physical_y = plan.orthogonal_source(products,c,ta,tb,true);
      for (const int mode : {0,1,2}) {
        const auto dy = plan.tangent_source(products,c,mode == 0 ? nullptr : &q,
            ta,tb,sigma,mode == 1 ? nullptr : &generator);
        const auto physical_dy = plan.tangent_source(products,c,mode == 0 ? nullptr : &q,
            ta,tb,sigma,mode == 1 ? nullptr : &generator,true);
        const auto expected = sigma.integral_adjoint(reconstruct(physical_dy,3),reconstruct(y,3));
        const auto check = [&](const xmvb::mcscf::DirectCiIntegralAdjoint& actual) {
          require((actual.one_electron-expected.one_electron).norm() < 3e-11 &&
                  (actual.pair_kernel-expected.pair_kernel).norm() < 3e-11,
                  "integral tangent projector placement changed the functional");
        };
        check(plan.integral_tangent(sigma,y,physical_y,dy,physical_dy));
        if (parity) {
          check(xmvb::mcscf::factor_integral_adjoint(sigma,physical_y,dy));
          // Mock an expensive physical directional loader to exercise the
          // alternative exact schedule without a large test allocation.
          int loads = 0;
          auto expensive = physical_dy;
          expensive.load_work = 1e18L;
          expensive.load = [physical_dy,&loads](int first,int count,
              Eigen::Ref<Eigen::MatrixXd> a,Eigen::Ref<Eigen::MatrixXd> b) {
            ++loads; physical_dy.load(first,count,a,b);
          };
          check(plan.integral_tangent(sigma,y,physical_y,dy,expensive));
          require(loads == 0,"integral planner regenerated an explicitly expensive equivalent bra");
        }
      }
    }
  }
}

/** @brief Row, mixed row/column and parity-representative factors match full CI. */
void check_mixed_source_images() {
  for (const int parity : {0, 1, -1}) {
    const auto fixture = make_fixture(4, 2, 2);
    const int nb = fixture.beta.size();
    std::map<int, std::vector<xmvb::vb::StructureExpansionTerm>> products;
    for (int b = 1; b < nb; ++b) {
      products[b].push_back({0, 0.1 * b});
      if (parity) products[b * nb].push_back({0, parity * 0.1 * b});
      else products[b * nb].push_back({1, -0.07 * (b + 1)});
    }
    Terms terms;
    Strings a, b;
    for (const auto& [product, coefficients] : products) {
      terms.push_back(coefficients);
      a.push_back(fixture.alpha[product / nb]); b.push_back(fixture.beta[product % nb]);
    }
    const auto cache = topology(fixture, a, b);
    const int structures = parity ? 1 : 2;
    const xmvb::vb::StructureAction action(terms, structures, cache,
        fixture.overlap, fixture.one_electron, fixture.two_electron, fixture.n,
        nullptr, xmvb::vb::StructureActionMode::ActionOnly);
    const auto vectors = test_vectors(structures, 3);
    const auto expected = reference_images(fixture, terms, a, b, vectors);
    const auto actual = action.apply_sources(vectors);
    check_images(actual, expected);
    check_images(action.apply(vectors), expected);
    require(actual.tiles.source_columns == vectors.cols() * (parity ? 1 : 2),
            "star support did not reduce to the minimal source cover");
    check_images(action.apply_sources(vectors, 9 * sizeof(double) *
        (fixture.alpha.size() + fixture.beta.size())), expected);
    require((action.apply_overlap(vectors) - expected.overlap).norm() < 1e-12,
            "mixed-source metric differs from full CI");
    std::vector<xmvb::mcscf::ProductIndex> support;
    Eigen::VectorXd support_values(products.size());
    int group = 0;
    for (const auto& [product, coefficients] : products) {
      support.push_back(product); support_values[group] = 0;
      for (const auto& term : coefficients)
        support_values[group] += term.coefficient * vectors(term.structure_index, 0);
      ++group;
    }
    const auto integrals = xmvb::vb::orthogonalize_active_integrals(
        fixture.overlap, fixture.one_electron, fixture.two_electron, fixture.n);
    const xmvb::vb::ExteriorOrbitalTransform ta(fixture.alpha, integrals.orbital_transform);
    const xmvb::vb::ExteriorOrbitalTransform tb(fixture.beta, integrals.orbital_transform);
    const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha, fixture.beta, integrals);
    const xmvb::vb::SpinSources plan(support, nb, nb, parity);
    const auto full = plan.orthogonal_source(support, support_values, ta, tb, true);
    xmvb::mcscf::FactorActionWorkspace workspace(full, sigma.channel_kernel().rows(), 65536, true);
    Eigen::MatrixXd reconstructed = Eigen::MatrixXd::Zero(nb, nb);
    workspace.apply(sigma, full, {}, [&](xmvb::mcscf::FactorImageBlock block) {
      reconstructed.noalias() += block.alpha * block.beta.transpose();
    });
    require((reconstructed - action.orthogonalize_structure_block(vectors.col(0))).norm() < 1e-12,
            "physical factor provider lost an exchange half or a mixed source");
  }
}

/** @brief Full carriers belong only to independent small-system test oracles. */
struct ProductOracle : xmvb::vb::StructureAdjointState {
  std::vector<Eigen::MatrixXd> orthogonal_coefficients,residuals;
};

ProductOracle explicit_product_state(const Fixture& fixture,
    const xmvb::vb::SameSpinPairCacheContext& cache,
    const xmvb::vb::StructureAction& action, xmvb::vb::StructureAdjointState accepted) {
  ProductOracle state;
  static_cast<xmvb::vb::StructureAdjointState&>(state)=std::move(accepted);
  state.product_images.reset();
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap,fixture.one_electron,fixture.two_electron,fixture.n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(cache.alpha_reuse_table.unique_determinants,
      cache.beta_reuse_table.unique_determinants,integrals);
  const auto y = action.orthogonalize_structure_block(state.structure_coefficients);
  state.source_factors = false; state.orthogonal_coefficients.clear(); state.residuals.clear();
  const int nb = fixture.beta.size();
  for (int i = 0; i < state.structure_coefficients.cols(); ++i) {
    state.orthogonal_coefficients.emplace_back(y.middleCols(i*nb,nb));
    state.residuals.push_back(sigma.apply(state.orthogonal_coefficients.back())-
        state.energies[i]*state.orthogonal_coefficients.back());
  }
  return state;
}

/** @brief Independent complete-carrier coefficient-response differential.
 * Uses ordinary sigma/adjoint entry points, never the generated product jet.
 * Full product arrays are restricted to this explicit small-system oracle.
 */
xmvb::vb::StructureActiveIntegralAdjoint explicit_response(const Fixture& fixture,
    const xmvb::vb::SameSpinPairCacheContext& cache,const xmvb::vb::StructureAction& action,
    const xmvb::vb::StructureAdjointState& accepted,const Eigen::MatrixXd& response,
    const Eigen::MatrixXd& multipliers) {
  const auto state=explicit_product_state(fixture,cache,action,accepted);
  const auto integrals=xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap,fixture.one_electron,fixture.two_electron,fixture.n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(cache.alpha_reuse_table.unique_determinants,
      cache.beta_reuse_table.unique_determinants,integrals);
  const Eigen::MatrixXd q=action.orthogonalize_structure_block(response);
  const int nb=fixture.beta.size(),n=fixture.n,d=integrals.pair_kernel.rows();
  xmvb::vb::StructureResponseSweep sweep{{},Eigen::MatrixXd::Zero(n,n),
      Eigen::MatrixXd::Zero(d,d),Eigen::MatrixXd::Zero(n,n)};
  for (int i=0;i<response.cols();++i) {
    const auto qi=q.middleCols(i*nb,nb);
    const auto& y=state.orthogonal_coefficients[i];
    const Eigen::MatrixXd residual=sigma.apply(qi)-state.energies[i]*qi;
    const auto adjoint=sigma.integral_adjoint(qi,y);
    sweep.one_electron.noalias()+=2*state.weights[i]*adjoint.one_electron;
    sweep.pair_kernel.noalias()+=2*state.weights[i]*adjoint.pair_kernel;
    sweep.generator.noalias()+=2*state.weights[i]*(sigma.one_body_generator_adjoint(residual,y)+
        sigma.one_body_generator_adjoint(state.residuals[i],qi));
  }
  return action.finish_response(accepted,std::move(sweep),multipliers);
}

/** @brief Sparse coupled responses retain both full-carrier residual terms. */
void check_factor_structure_response(int parity, bool unequal = false) {
  const auto fixture = unequal ? make_fixture(5,2,1) : make_fixture(6,3,3);
  CovalentExpansion expansion;
  for (int s = 0; s < 2; ++s) {
    expansion.product_alpha.push_back(fixture.alpha[0]);
    expansion.product_beta.push_back(fixture.beta[1+2*s]);
    expansion.terms.push_back({{s,0.7}});
    if (parity) {
      expansion.product_alpha.push_back(fixture.alpha[1+2*s]);
      expansion.product_beta.push_back(fixture.beta[0]);
      expansion.terms.push_back({{s,0.7*parity}});
    }
  }
  const auto cache = topology(fixture,expansion.product_alpha,expansion.product_beta);
  const xmvb::vb::StructureAction action(expansion.terms,2,cache,fixture.overlap,
      fixture.one_electron,fixture.two_electron,fixture.n,nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  check_fixed_states(action);
  const Eigen::MatrixXd c = (Eigen::Matrix2d() << 0.7,0.2,-0.3,0.8).finished();
  const Eigen::MatrixXd q = (Eigen::Matrix2d() << 0.02,-0.03,0.04,0.01).finished();
  const Eigen::MatrixXd mu = (Eigen::Matrix2d() << 0.01,0.02,-0.03,0.005).finished();
  const auto accepted = action.prepare_active_adjoint(c,{0.3,0.7},{-1.1,-0.7});
  require(accepted.source_factors && !accepted.product_images,
          "source accepted state retained a complete product array");
  const auto dense = explicit_product_state(fixture,cache,action,accepted);
  const auto reference = explicit_response(fixture,cache,action,dense,q,mu);
  // No accepted product arrays may be needed by this consumer.
  auto sparse = accepted;
  auto sweep = action.apply_response(sparse,q);
  require(sweep.images.tiles.source_columns > 0 &&
      sweep.images.tiles.source_workspace_bound <= xmvb::mcscf::kFactorWorkspaceBytes,
          "sparse structure response missed source contraction or exceeded its workspace");
  check_images(sweep.images,reference_images(fixture,expansion.terms,
      expansion.product_alpha,expansion.product_beta,q));
  const auto actual = action.finish_response(sparse,std::move(sweep),mu);
  require((actual.overlap-reference.overlap).norm() < 2e-11 &&
      (actual.one_electron-reference.one_electron).norm() < 2e-11 &&
      (actual.pair_kernel-reference.pair_kernel).norm() < 2e-11,
      "source structure response lost residual ordering, parity or multistate gauge");
  const auto product=action.active_integral_response_adjoint(dense,q,mu);
  require((product.overlap-reference.overlap).norm()<3e-11 &&
          (product.one_electron-reference.one_electron).norm()<3e-11 &&
          (product.pair_kernel-reference.pair_kernel).norm()<3e-11,
          "product-tile structure response lost an oriented residual or selected-frame multiplier");
  // Coefficients and multipliers are independent linear directions, including
  // off-diagonal state-average multipliers and nonstationary carrier residuals.
  const xmvb::vb::StructureAdjointState* variants[] = {&sparse,&dense};
  for (const auto* state : variants) {
    const auto only_q = action.active_integral_adjoint_direction(*state,&q,nullptr,
        {},{},{},nullptr,false);
    const auto only_mu = action.active_integral_adjoint_direction(*state,nullptr,&mu,
        {},{},{},nullptr,false);
    require((only_q.overlap+only_mu.overlap-reference.overlap).norm() < 3e-11 &&
            (only_q.one_electron+only_mu.one_electron-reference.one_electron).norm() < 3e-11 &&
            (only_q.pair_kernel+only_mu.pair_kernel-reference.pair_kernel).norm() < 3e-11,
            "independent coefficient/multiplier directions changed a state-average adjoint");
  }
  const auto zero = action.apply_response(sparse,Eigen::MatrixXd::Zero(2,2));
  require(zero.images.hamiltonian.isZero(0) && zero.images.overlap.isZero(0) &&
      zero.generator.isZero(0) && zero.one_electron.isZero(0) && zero.pair_kernel.isZero(0),
      "exactly zero structure response performed a nonzero contraction");
  const Eigen::MatrixXd ds = 0.001*Eigen::MatrixXd::Ones(fixture.n,fixture.n);
  const Eigen::MatrixXd dh = -0.002*Eigen::MatrixXd::Identity(fixture.n,fixture.n);
  const std::vector<double> overlap(ds.data(),ds.data()+ds.size()), one(dh.data(),dh.data()+dh.size());
  const std::vector<double> two(fixture.two_electron.packed_active_two_electron_integrals.size(),0.0001);
  xmvb::vb::StructureIntegralDirection source_cache, product_cache;
  source_cache.capture_connection = product_cache.capture_connection = true;
  const auto source_images = action.apply_integral_direction(sparse,overlap,one,two,&source_cache);
  check_images(source_images,action.apply_integral_direction(dense,overlap,one,two,&product_cache));
  require(source_images.tiles.source_workspace_bound <= xmvb::mcscf::kFactorWorkspaceBytes,
          "source image/VJP exceeded the single shared numerical workspace");
  require(source_cache.connection.has_value(),"source direction lost its requested diagnostic");
  require((source_cache.connection->representation_forcing-product_cache.connection->representation_forcing).norm() < 2e-11 &&
      (source_cache.connection->integral_forcing-product_cache.connection->integral_forcing).norm() < 2e-11 &&
      (source_cache.connection->overlap-product_cache.connection->overlap).norm() < 2e-11 &&
      (source_cache.connection->squared_norm-product_cache.connection->squared_norm).norm() < 2e-11,
      "source connection diagnostic differs from the independent product calculation");
  const auto source_hvp = action.active_integral_adjoint_direction(sparse,&q,&mu,
      overlap,one,two,&source_cache,true);
  const auto product_hvp = action.active_integral_adjoint_direction(dense,&q,&mu,
      overlap,one,two,&product_cache,true);
  require((source_hvp.overlap-product_hvp.overlap).norm() < 2e-11 &&
      (source_hvp.one_electron-product_hvp.one_electron).norm() < 2e-11 &&
      (source_hvp.pair_kernel-product_hvp.pair_kernel).norm() < 2e-11,
      "source HVP lost a left/right representation, delta-H or gauge term");
  const Eigen::MatrixXd single_c = c.leftCols(1), scaling_q = 0.17*single_c;
  const Eigen::MatrixXd single_mu = Eigen::MatrixXd::Constant(1,1,0.02);
  const auto single = action.prepare_active_adjoint(single_c,{1.0},{-1.1});
  const auto single_dense = explicit_product_state(fixture,cache,action,single);
  const xmvb::vb::StructureAdjointState* single_variants[] = {&single,&single_dense};
  for (const auto* state : single_variants) {
    const auto base = action.active_integral_adjoint(*state);
    for (const bool integral_response : {false,true}) {
      const auto full = action.active_integral_adjoint_direction(*state,&scaling_q,&single_mu,
          overlap,one,two,nullptr,integral_response);
      const auto reused = action.active_integral_adjoint_direction(*state,nullptr,&single_mu,
          overlap,one,two,nullptr,integral_response);
      require((reused.overlap+0.34*base.overlap-full.overlap).norm() < 3e-11 &&
              (reused.one_electron+0.34*base.one_electron-full.one_electron).norm() < 3e-11 &&
              (reused.pair_kernel+0.34*base.pair_kernel-full.pair_kernel).norm() < 3e-11,
              "known coefficient scaling cannot be reused from accepted active adjoints");
    }
  }
}

/** @brief Joint orbital/CI product jet agrees with the two separate actions. */
void check_joint_product_jet(const xmvb::vb::StructureAction& action,
    const xmvb::vb::StructureAdjointState& state,const Eigen::MatrixXd& q,
    const std::vector<double>& overlap,const std::vector<double>& one,
    const std::vector<double>& two,bool capture=true) {
  require(!state.source_factors,"joint product-jet fixture selected source factors");
  xmvb::vb::StructureIntegralDirection separate_direction,joint_direction;
  separate_direction.capture_connection=joint_direction.capture_connection=capture && q.rows()<16;
  const auto separate_p=action.apply_integral_direction(state,overlap,one,two,&separate_direction);
  const auto separate_q=action.apply_response(state,q);
  xmvb::vb::StructureResponseRequest request{q,{}};
  const auto joint_p=action.apply_integral_direction(state,overlap,one,two,&joint_direction,&request);
  const auto close=[](const Eigen::MatrixXd& actual,const Eigen::MatrixXd& expected) {
    return (actual-expected).norm()<3e-11*std::max(1.0,expected.norm());
  };
  require(close(joint_p.hamiltonian,separate_p.hamiltonian) &&
          close(joint_p.overlap,separate_p.overlap),
          "joint product jet changed the orbital-direction H/S images");
  require(close(request.result.images.hamiltonian,separate_q.images.hamiltonian) &&
          close(request.result.images.overlap,separate_q.images.overlap),
          "joint product jet changed the CI-response H/S images");
  check_images(joint_p,separate_p);
  check_images(request.result.images,separate_q.images);
  require(close(request.result.one_electron,separate_q.one_electron) &&
          close(request.result.pair_kernel,separate_q.pair_kernel) &&
          close(request.result.generator,separate_q.generator),
          "joint product jet mixed the CI-response active cotangents");
  require(close(joint_direction.one_electron_direction,separate_direction.one_electron_direction) &&
          close(joint_direction.pair_kernel_direction,separate_direction.pair_kernel_direction) &&
          close(joint_direction.generator_direction,separate_direction.generator_direction),
          "joint product jet mixed the orbital-direction active cotangents");
  if (separate_direction.capture_connection) {
    require(separate_direction.connection.has_value() && joint_direction.connection.has_value(),
            "joint product jet lost the orbital connection diagnostic");
    const auto& a=*joint_direction.connection;
    const auto& b=*separate_direction.connection;
    require(close(a.representation_forcing,b.representation_forcing) &&
            close(a.integral_forcing,b.integral_forcing) && close(a.overlap,b.overlap) &&
            close(a.squared_norm,b.squared_norm),
            "joint product jet changed orbital connection components");
  }
}

/** @brief Product response: selected nonclosed spaces, two roots and zero weights. */
void check_product_response(int parity,bool unequal) {
  const auto fixture=make_fixture(unequal ? 5:4,2,unequal ? 1:2);
  Terms terms;
  Strings alpha,beta;
  const std::vector<std::pair<int,int>> edges=unequal ?
      std::vector<std::pair<int,int>>{{0,0},{1,2},{2,4},{4,1},{5,3},{7,0},{9,4}} :
      std::vector<std::pair<int,int>>{{0,2},{1,4},{3,5},{0,5}};
  for (int i=0;i<static_cast<int>(edges.size());++i) {
    const auto [a,b]=edges[i];
    alpha.push_back(fixture.alpha[a]); beta.push_back(fixture.beta[b]); terms.push_back({{i,0.7}});
    if (parity) {
      alpha.push_back(fixture.alpha[b]); beta.push_back(fixture.beta[a]);
      terms.push_back({{i,0.7*parity}});
    }
  }
  const int ns=edges.size();
  const auto cache=topology(fixture,alpha,beta);
  const xmvb::vb::StructureAction action(terms,ns,cache,fixture.overlap,
      fixture.one_electron,fixture.two_electron,fixture.n,nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  check_fixed_states(action);
  const Eigen::MatrixXd c=test_vectors(ns,2),q=0.02*c.array().cos().matrix();
  const Eigen::MatrixXd mu=(Eigen::Matrix2d()<<0.01,0.02,-0.03,0.005).finished();
  const Eigen::MatrixXd ds=0.001*Eigen::MatrixXd::Ones(fixture.n,fixture.n);
  const Eigen::MatrixXd dh=-0.002*Eigen::MatrixXd::Identity(fixture.n,fixture.n);
  const std::vector<double> overlap(ds.data(),ds.data()+ds.size());
  const std::vector<double> one(dh.data(),dh.data()+dh.size());
  const std::vector<double> two(
      fixture.two_electron.packed_active_two_electron_integrals.size(),0.0001);
  for (const auto& weights:{std::vector<double>{0.3,0.7},std::vector<double>{1.0,0.0}}) {
    const auto state=action.prepare_active_adjoint(c,weights,{-1.1,-0.7});
    Eigen::MatrixXd multipliers=mu;
    // The zero-weight test requests no gauge coupling to its unprepared root.
    if (weights.back()==0.0) multipliers.setZero();
    require(!state.source_factors,"product-response fixture selected source factors");
    check_joint_product_jet(action,state,q,overlap,one,two);
    check_joint_product_jet(action,state,Eigen::MatrixXd::Zero(q.rows(),q.cols()),overlap,one,two);
    auto nonresident=state;
    nonresident.product_images.reset();
    check_joint_product_jet(action,nonresident,q,overlap,one,two);
    auto sweep=action.apply_response(state,q);
    require(sweep.images.tiles.input_panels>0 && sweep.images.tiles.integral_adjoint_sweeps>0,
            "product response bypassed the fused generated sigma/adjoint graph");
    check_images(sweep.images,reference_images(fixture,terms,alpha,beta,q));
    const auto actual=action.finish_response(state,std::move(sweep),multipliers);
    auto streamed=state;
    streamed.product_images=std::make_shared<xmvb::vb::detail::ProductImages>();
    streamed.product_images->budget=128;
    check_joint_product_jet(action,streamed,q,overlap,one,two);
    const auto evicted=action.active_integral_response_adjoint(streamed,q,multipliers);
    require((actual.overlap-evicted.overlap).norm()<3e-11 &&
            (actual.one_electron-evicted.one_electron).norm()<3e-11 &&
            (actual.pair_kernel-evicted.pair_kernel).norm()<3e-11,
            "bounded replay/partial-image eviction changed a product response");
    const auto oracle=explicit_response(fixture,cache,action,state,q,multipliers);
    require((actual.overlap-oracle.overlap).norm()<3e-11 &&
            (actual.one_electron-oracle.one_electron).norm()<3e-11 &&
            (actual.pair_kernel-oracle.pair_kernel).norm()<3e-11,
            "product response differs from the independent complete-carrier differential");
    const auto adjoint=action.active_integral_response_adjoint(state,q,multipliers);
    require((actual.overlap-adjoint.overlap).norm()<3e-11 &&
            (actual.one_electron-adjoint.one_electron).norm()<3e-11 &&
            (actual.pair_kernel-adjoint.pair_kernel).norm()<3e-11,
            "disabling structure projection changed the physical response adjoint");
    if (weights.back()!=0.0) {
      const Eigen::MatrixXd tiny=1e-180*q;
      auto scaled=action.apply_response(state,tiny);
      scaled.images.hamiltonian*=1e180;
      scaled.images.overlap*=1e180;
      check_images(scaled.images,reference_images(fixture,terms,alpha,beta,q));
      const auto small=action.finish_response(state,std::move(scaled),Eigen::MatrixXd::Zero(2,2));
      const auto pure=action.active_integral_adjoint_direction(state,&q,nullptr,{},{},{},nullptr,false);
      require((1e180*small.overlap-pure.overlap).norm()<3e-11 &&
              (1e180*small.one_electron-pure.one_electron).norm()<3e-11 &&
              (1e180*small.pair_kernel-pure.pair_kernel).norm()<3e-11,
              "exact-zero pruning discarded a representable tiny linear response");
    }
  }
}

/** @brief Broad selected support: one final projection, independent of closure. */
void check_broad_product_projection(bool unequal,bool partial=false,int parity=0) {
  const auto fixture=unequal ? make_fixture(5,2,1):make_fixture(6,3,3);
  Terms terms;
  Strings alpha,beta;
  for (int a=0;a<static_cast<int>(fixture.alpha.size());++a)
    for (int b=0;b<static_cast<int>(fixture.beta.size());++b) {
      // A broad but nonclosed raw support, including an interior column hole.
      // Ordered transforms must not assume all Ua*Ub products were selected.
      if (partial && a==static_cast<int>(fixture.alpha.size())-1 && b>=3) continue;
      if (parity<0 && a==b) continue;
      alpha.push_back(fixture.alpha[a]); beta.push_back(fixture.beta[b]);
      const int i=parity ? std::min(a,b):a,j=parity ? std::max(a,b):b;
      const double sign=parity && a>b ? parity:1;
      terms.push_back({{0,sign*0.1*std::sin(i+2*j+1)},
                       {1,sign*0.2*std::cos(2*i+j+1)},
                       {2,sign*0.15*std::sin(3*i+j+2)}});
    }
  const auto cache=topology(fixture,alpha,beta);
  const auto build=[&](const Fixture& f) {
    return xmvb::vb::StructureAction(terms,3,cache,f.overlap,f.one_electron,
        f.two_electron,f.n,nullptr,xmvb::vb::StructureActionMode::ActionOnly);
  };
  const auto action=build(fixture);
  const Eigen::MatrixXd c=test_vectors(3,2),q=0.02*c.array().cos().matrix();
  if (partial || parity) {
    const auto hs=action.apply_streamed(c);
    require(hs.tiles.source_workspace_bound==0 && hs.tiles.input_resident_bytes>0,
        "broad selected H/M regression did not execute the product contraction");
    check_images(hs,reference_images(fixture,terms,alpha,beta,c));
    check_images(hs,action.apply_streamed(c,16*1024));
    require((action.apply_hamiltonian(c)-hs.hamiltonian).norm()<3e-11,
        "ordered Hamiltonian-only contraction changed selected-space projection");
  }
  auto wide=action.prepare_active_adjoint(c,{0.4,0.6},{-1.1,-0.7});
  wide.source_factors=false;
  wide.product_images=std::make_shared<xmvb::vb::detail::ProductImages>();
  auto narrow=wide;
  narrow.product_images=std::make_shared<xmvb::vb::detail::ProductImages>();
  narrow.product_images->budget=128;
  const auto response=action.apply_response(wide,q);
  // Exercise the source image order on the same nonclosed, arbitrary three
  // structures. The wider equal-spin fixture favours the support-map image
  // consumer; the unequal-spin fixture also certifies the resident order.
  auto source_state=wide;
  source_state.source_factors=true; source_state.product_images.reset();
  const auto source_response=action.apply_response(source_state,q);
  check_images(source_response.images,response.images);
  require((source_response.generator-response.generator).norm()<3e-11 &&
          (source_response.one_electron-response.one_electron).norm()<3e-11 &&
          (source_response.pair_kernel-response.pair_kernel).norm()<3e-11 &&
          source_response.images.tiles.source_workspace_bound<=xmvb::mcscf::kFactorWorkspaceBytes,
          "single-pass source image changed an arbitrary-space adjoint or exceeded its budget");
  check_images(response.images,reference_images(fixture,terms,alpha,beta,q));
  check_images(response.images,action.apply_response(narrow,q).images);
  require(response.images.tiles.structure_projections==2*c.cols(),
          "broad response repeated its final structure projection");
  const Eigen::MatrixXd ds=0.001*Eigen::MatrixXd::Ones(fixture.n,fixture.n);
  const Eigen::MatrixXd dh=-0.002*Eigen::MatrixXd::Identity(fixture.n,fixture.n);
  const std::vector<double> overlap(ds.data(),ds.data()+ds.size());
  const std::vector<double> one(dh.data(),dh.data()+dh.size());
  const std::vector<double> two(fixture.two_electron.packed_active_two_electron_integrals.size(),0.0001);
  xmvb::vb::StructureIntegralDirection dw,dn;
  dw.capture_connection=dn.capture_connection=true;
  const auto image=action.apply_integral_direction(wide,overlap,one,two,&dw);
  check_images(image,action.apply_integral_direction(narrow,overlap,one,two,&dn));
  require(image.tiles.structure_projections==5*c.cols(),
          "broad orbital jet repeated its final structure projection");
  require((dw.generator_direction-dn.generator_direction).norm()<3e-11 &&
          (dw.connection->integral_forcing-dn.connection->integral_forcing).norm()<3e-11 &&
          (dw.connection->representation_forcing-dn.connection->representation_forcing).norm()<3e-11 &&
          (dw.connection->overlap-dn.connection->overlap).norm()<3e-11,
          "one final beta adjoint changed the generator or connection jet");
  // Independently populate immutable accepted Y/(H-E)Y panels. This exercises
  // accepted-state jet reuse without generating it through the implementation.
  const auto oracle=explicit_product_state(fixture,cache,action,wide);
  if (partial) {
    // Populate the real compact accepted cache, not only the independent
    // full-column oracle below. Interior holes must survive ordered input.
    auto compact=action.prepare_active_adjoint(c,{0.4,0.6},{-1.1,-0.7});
    require(!compact.source_factors && compact.product_images &&
        !compact.product_images->states.empty() && !compact.product_images->states.front().empty(),
        "broad selected regression did not retain accepted product tiles");
    xmvb::vb::StructureIntegralDirection dcompact;
    dcompact.capture_connection=true;
    const auto compact_image=action.apply_integral_direction(compact,overlap,one,two,&dcompact);
    check_images(compact_image,image);
    require((dcompact.generator_direction-dw.generator_direction).norm()<3e-11,
        "sparse accepted-image retention changed the orbital generator jet");
  }
  auto resident=wide;
  resident.product_images=std::make_shared<xmvb::vb::detail::ProductImages>();
  resident.product_images->extent=fixture.beta.size();
  resident.product_images->states.resize(c.cols());
  for (int s=0;s<c.cols();++s) {
    auto pattern=std::make_shared<xmvb::vb::detail::ColumnPattern>();
    pattern->full=true; pattern->offsets.resize(fixture.beta.size()+1);
    for (int b=0;b<=static_cast<int>(fixture.beta.size());++b)
      pattern->offsets[b]=b*fixture.alpha.size();
    Eigen::MatrixXd values(oracle.orthogonal_coefficients[s].size(),2);
    values.col(0)=Eigen::Map<const Eigen::VectorXd>(
        oracle.orthogonal_coefficients[s].data(),values.rows());
    values.col(1)=Eigen::Map<const Eigen::VectorXd>(oracle.residuals[s].data(),values.rows());
    resident.product_images->states[s].push_back({pattern,std::move(values)});
    resident.product_images->bytes+=resident.product_images->states[s][0].bytes();
  }
  xmvb::vb::StructureIntegralDirection dcached;
  dcached.capture_connection=true;
  const auto cached_image=action.apply_integral_direction(resident,overlap,one,two,&dcached);
  check_images(cached_image,image);
  require((dcached.generator_direction-dw.generator_direction).norm()<3e-11 &&
          (dcached.connection->representation_forcing-dw.connection->representation_forcing).norm()<3e-11 &&
          (dcached.connection->overlap-dw.connection->overlap).norm()<3e-11 &&
          (dcached.connection->squared_norm-dw.connection->squared_norm).norm()<3e-11 &&
          cached_image.tiles.input_resident_bytes<=resident.product_images->budget,
          "accepted-state jet reuse changed the derivative or exceeded its shared input budget");
  for (bool capture:{false,true}) {
    check_joint_product_jet(action,wide,q,overlap,one,two,capture);
    check_joint_product_jet(action,narrow,q,overlap,one,two,capture);
    check_joint_product_jet(action,resident,q,overlap,one,two,capture);
  }
  constexpr double step=2e-5;
  const auto shifted=[&](double scale) {
    auto f=fixture;
    for (int i=0;i<ds.size();++i) f.overlap[i]+=scale*ds.data()[i];
    f.one_electron+=scale*dh;
    for (int i=0;i<static_cast<int>(two.size());++i)
      f.two_electron.packed_active_two_electron_integrals[i]+=scale*two[i];
    return build(f).apply(c);
  };
  const auto plus=shifted(step),minus=shifted(-step);
  require(((plus.hamiltonian-minus.hamiltonian)/(2*step)-image.hamiltonian).norm()<2e-7 &&
          ((plus.overlap-minus.overlap)/(2*step)-image.overlap).norm()<2e-7,
          "broad orbital jet disagrees with independent H/M differences");
}

/** @brief Production accepted gradient keeps the nonstationary carrier term. */
void check_factor_accepted_gradient(bool sparse = false) {
  const int n = sparse ? 5 : 8;
  const auto fixture = sparse ? make_fixture(n, 2, 1) : make_fixture(n, 4, 4);
  CovalentExpansion expansion;
  if (sparse) {
    expansion.product_alpha = {fixture.alpha[0], fixture.alpha[0]};
    expansion.product_beta = {fixture.beta[1], fixture.beta[3]};
    expansion.terms = {{{0, 0.7}}, {{0, -0.2}}};
  } else {
    Pairing adjacent;
    for (int i = 0; i < n; i += 2) adjacent.emplace_back(i, i + 1);
    expansion = covalent_rumer_terms(n, fixture.alpha, fixture.beta, {adjacent});
  }
  const auto cache = topology(fixture, expansion.product_alpha, expansion.product_beta);
  const xmvb::vb::StructureAction action(expansion.terms, 1, cache, fixture.overlap,
      fixture.one_electron, fixture.two_electron, n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  Eigen::MatrixXd c = Eigen::MatrixXd::Ones(1, 1);
  const auto images = action.apply(c);
  const double energy = images.hamiltonian(0,0) / images.overlap(0,0);
  c /= std::sqrt(images.overlap(0,0));
  const auto state = action.prepare_active_adjoint(c, {1.0}, {energy});
  require(state.source_factors || state.product_images,
          "accepted gradient has no bounded contraction plan");
  require(state.structure_coefficients.rows() == 1 && state.structure_coefficients.cols() == 1,
          "accepted raw state was not retained in the selected structure space");
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(
      cache.alpha_reuse_table.unique_determinants,
      cache.beta_reuse_table.unique_determinants, integrals);
  const auto y = action.orthogonalize_structure_block(c);
  const auto reference = sigma.integral_adjoint(y, y);
  const Eigen::MatrixXd residual = sigma.apply(y) - energy * y;
  const Eigen::MatrixXd generator = 2 * sigma.one_body_generator_adjoint(residual, y);
  require(generator.norm() > 1e-7,
          "accepted-gradient fixture does not probe a nonzero carrier residual");
  const auto expected = xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
      integrals, reference.one_electron, reference.pair_kernel, generator);
  const auto actual = action.active_integral_adjoint(state);
  require((actual.overlap-expected.overlap).norm() < 2e-11 &&
      (actual.one_electron-expected.one_electron).norm() < 2e-11 &&
      (actual.pair_kernel-expected.pair_kernel).norm() < 2e-11,
      "factor accepted gradient differs from complete integral/generator pullback");
  Eigen::MatrixXd ds(n,n), dh(n,n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) {
      ds(i,j) = ds(j,i) = 0.003 * std::cos(0.13 * (i + 1) * (j + 2));
      dh(i,j) = dh(j,i) = 0.004 * std::sin(0.17 * (i + 2) * (j + 1));
    }
  std::vector<double> dg(fixture.two_electron.packed_active_two_electron_integrals.size());
  for (int i = 0; i < static_cast<int>(dg.size()); ++i) dg[i] = 0.0003 * std::cos(0.11 * (i + 1));
  constexpr double step = 2e-5;
  const auto shifted = [&](double scale, double response_scale) {
    auto moved = fixture;
    for (int i = 0; i < static_cast<int>(moved.overlap.size()); ++i) moved.overlap[i] += scale * ds.data()[i];
    moved.one_electron += scale * dh;
    for (int i = 0; i < static_cast<int>(dg.size()); ++i)
      moved.two_electron.packed_active_two_electron_integrals[i] += scale * dg[i];
    const xmvb::vb::StructureAction shifted_action(expansion.terms, 1, cache, moved.overlap,
        moved.one_electron, moved.two_electron, n, nullptr,
        xmvb::vb::StructureActionMode::ActionOnly);
    return shifted_action.active_integral_adjoint(
        shifted_action.prepare_active_adjoint((1+response_scale*scale)*c, {1.0}, {energy}));
  };
  for (bool response : {false, true}) {
    const Eigen::MatrixXd dc = 0.21*c;
    const Eigen::MatrixXd multipliers = Eigen::MatrixXd::Zero(1,1);
    const auto analytic = action.active_integral_adjoint_direction(state,
        response ? &dc : nullptr, response ? &multipliers : nullptr,
        std::vector<double>(ds.data(), ds.data()+ds.size()),
        std::vector<double>(dh.data(), dh.data()+dh.size()), dg, nullptr, true);
    const auto plus = shifted(step, response ? 0.21 : 0), minus = shifted(-step, response ? 0.21 : 0);
    require(((plus.overlap-minus.overlap)/(2*step)-analytic.overlap).norm() < 2e-7 &&
        ((plus.one_electron-minus.one_electron)/(2*step)-analytic.one_electron).norm() < 2e-7 &&
        ((plus.pair_kernel-minus.pair_kernel)/(2*step)-analytic.pair_kernel).norm() < 2e-7,
        "factor HVP differs from the complete orbital/structure gradient derivative");
  }
}

/** @brief One-column images reuse adjoints without assuming a normalized CI state. */
void check_scalar_source_direction(int parity, bool unequal = false) {
  const auto fixture = unequal ? make_fixture(5,2,1) : make_fixture(6,3,3);
  Terms terms{{{0,0.7}}};
  Strings a{fixture.alpha[0]}, b{fixture.beta[1]};
  if (parity) {
    terms.push_back({{0,parity*0.7}});
    a.push_back(fixture.alpha[1]); b.push_back(fixture.beta[0]);
  }
  const auto cache = topology(fixture,a,b);
  const xmvb::vb::StructureAction action(terms,1,cache,fixture.overlap,
      fixture.one_electron,fixture.two_electron,fixture.n,nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto state = action.prepare_active_adjoint(Eigen::MatrixXd::Constant(1,1,1.3),{0.6},{-1.1});
  require(state.source_factors,"single-column fixture missed source factors");
  const auto dense = explicit_product_state(fixture,cache,action,state);
  const Eigen::MatrixXd ds = 0.001*Eigen::MatrixXd::Ones(fixture.n,fixture.n);
  const Eigen::MatrixXd dh = -0.002*Eigen::MatrixXd::Identity(fixture.n,fixture.n);
  const std::vector<double> overlap(ds.data(),ds.data()+ds.size()), one(dh.data(),dh.data()+dh.size());
  const std::vector<double> two(fixture.two_electron.packed_active_two_electron_integrals.size(),0.0001);
  xmvb::vb::StructureIntegralDirection direction, reference;
  reference.capture_connection = true;
  const auto images = action.apply_integral_direction(state,overlap,one,two,&direction);
  const auto expected = action.apply_integral_direction(dense,overlap,one,two,&reference);
  check_images(images,expected);
  check_images(action.apply_integral_direction(state,overlap,one,two,nullptr),expected);
  require(images.tiles.structure_projections == 0 && images.tiles.source_columns > 0,
          "scalar direction retained unnecessary exterior/structure projections");
  const auto integrals=xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap,fixture.one_electron,fixture.two_electron,fixture.n);
  const auto delta=xmvb::vb::orthogonalize_active_integral_direction(
      integrals,overlap,dh,two,fixture.n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha,fixture.beta,integrals), delta_sigma(sigma,delta);
  const Eigen::MatrixXd generator=integrals.orbital_transform.transpose().triangularView<Eigen::Lower>()
      .solve(delta.orbital_transform.transpose()).transpose();
  const Eigen::MatrixXd y=action.orthogonalize_structure_block(state.structure_coefficients);
  const Eigen::MatrixXd dy=sigma.apply_one_body_generator(y,generator);
  const Eigen::MatrixXd residual=sigma.apply(y)-state.energies[0]*y;
  const Eigen::MatrixXd dr=sigma.apply(dy)-state.energies[0]*dy+delta_sigma.apply(y);
  const Eigen::MatrixXd expected_generator=2*state.weights[0]*(sigma.one_body_generator_adjoint(dr,y)+
      sigma.one_body_generator_adjoint(residual,dy));
  require((direction.generator_direction-expected_generator).norm()<3e-11,
          "fused residual source sweep changed the scalar generator derivative");
  // The Cholesky direction is upper triangular. The product oracle may omit
  // unused lower-triangular covector entries; compare its physical pullback.
  const Eigen::MatrixXd product_error=reference.generator_direction-expected_generator;
  require(product_error.triangularView<Eigen::Upper>().toDenseMatrix().norm()<3e-11,
          "product oracle changed the Cholesky generator covector");
  const Eigen::MatrixXd q=Eigen::MatrixXd::Constant(1,1,0.03), mu=Eigen::MatrixXd::Constant(1,1,-0.02);
  const auto hvp=action.active_integral_adjoint_direction(state,&q,&mu,overlap,one,two,&direction,true);
  const auto oracle=action.active_integral_adjoint_direction(dense,&q,&mu,overlap,one,two,&reference,true);
  require((hvp.overlap-oracle.overlap).norm()<3e-11 &&
          (hvp.one_electron-oracle.one_electron).norm()<3e-11 &&
          (hvp.pair_kernel-oracle.pair_kernel).norm()<3e-11,
          "scalar source shortcut changed the complete moving adjoint");
  // No normalization or orbital motion in this direction; integral adjoints
  // alone must recover delta H, and delta M must be exactly zero.
  const std::vector<double> zero_overlap(overlap.size(),0.0);
  xmvb::vb::StructureIntegralDirection fixed, fixed_reference;
  const auto fixed_images=action.apply_integral_direction(state,zero_overlap,one,two,&fixed);
  check_images(fixed_images,action.apply_integral_direction(dense,zero_overlap,one,two,&fixed_reference));
  require(fixed_images.overlap.isZero(0),"fixed-overlap scalar direction changed M");
}

/** @brief Factor HVP keeps multistate weights and off-diagonal gauge terms. */
void check_factor_state_average() {
  const auto fixture = make_fixture(5,2,1);
  const auto cache = topology(fixture, {fixture.alpha[0],fixture.alpha[0]},
      {fixture.beta[1],fixture.beta[3]});
  const Terms terms = {{{0,1.0}}, {{1,1.0}}};
  const xmvb::vb::StructureAction action(terms, 2, cache, fixture.overlap,
      fixture.one_electron, fixture.two_electron, 5, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const Eigen::MatrixXd c = (Eigen::Matrix2d() << 0.7,0.2,-0.3,0.8).finished();
  const Eigen::MatrixXd dc = (Eigen::Matrix2d() << 0.02,-0.03,0.04,0.01).finished();
  const Eigen::MatrixXd mu = (Eigen::Matrix2d() << 0.01,0.02,-0.03,0.005).finished();
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, 5);
  const xmvb::mcscf::DirectCiSigmaAction sigma(cache.alpha_reuse_table.unique_determinants,
      cache.beta_reuse_table.unique_determinants, integrals);
  const Eigen::MatrixXd ds = 0.001 * Eigen::MatrixXd::Ones(5,5);
  const Eigen::MatrixXd dh = -0.002 * Eigen::MatrixXd::Ones(5,5);
  const std::vector<double> overlap(ds.data(), ds.data()+ds.size());
  const std::vector<double> one(dh.data(), dh.data()+dh.size());
  const std::vector<double> two(fixture.two_electron.packed_active_two_electron_integrals.size(), 0.0001);
  const auto direction = xmvb::vb::orthogonalize_active_integral_direction(integrals, overlap, dh, two, 5);
  const xmvb::mcscf::DirectCiSigmaAction delta_sigma(sigma, direction);
  const Eigen::MatrixXd kappa = integrals.orbital_transform.transpose()
      .triangularView<Eigen::Lower>().solve(direction.orbital_transform.transpose()).transpose();
  const auto y = action.orthogonalize_structure_block(c);
  const auto dy_c = action.orthogonalize_structure_block(dc);
  const int nb = fixture.beta.size();
  const std::vector<double> energies{-1.1,-0.7};
  for (const std::vector<double>& weights : {std::vector<double>{0.5,0.5}, {0.3,0.7}}) {
    const auto state = action.prepare_active_adjoint(c, weights, energies);
    require(state.source_factors || state.product_images,
            "multistate gradient has no bounded contraction plan");
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j <= i; ++j)
        require((state.overlap_generators[i*(i+1)/2+j] -
            sigma.one_body_generator_adjoint(y.middleCols(j*nb,nb),y.middleCols(i*nb,nb))).norm() < 2e-11,
            "factor transition density lost an off-diagonal state orientation");
    Eigen::MatrixXd g1 = Eigen::MatrixXd::Zero(5,5), g2 = Eigen::MatrixXd::Zero(15,15), gg = g1;
    for (int i = 0; i < 2; ++i) {
      const auto yi = y.middleCols(i*nb, nb);
      const Eigen::MatrixXd dy = sigma.apply_one_body_generator(yi,kappa) + dy_c.middleCols(i*nb,nb);
      const Eigen::MatrixXd r = sigma.apply(yi) - energies[i] * yi;
      Eigen::MatrixXd dr = sigma.apply(dy) + delta_sigma.apply(yi) - energies[i] * dy;
      for (int j = 0; j < 2; ++j)
        dr.noalias() += (0.5 * mu(j,i) + 0.5 * mu(i,j) * weights[j] / weights[i]) * y.middleCols(j*nb,nb);
      const auto integral = sigma.integral_adjoint(dy,yi);
      g1.noalias() += 2 * weights[i] * integral.one_electron;
      g2.noalias() += 2 * weights[i] * integral.pair_kernel;
      gg.noalias() += 2 * weights[i] * (sigma.one_body_generator_adjoint(dr,yi) +
          sigma.one_body_generator_adjoint(r,dy));
    }
    const auto expected = xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(
        integrals, direction, state.one_electron_gradient, state.pair_kernel_gradient,
        state.generator_gradient, g1,g2,gg);
    const auto actual = action.active_integral_adjoint_direction(
        state, &dc, &mu, overlap, one, two, nullptr, true);
    require((actual.overlap-expected.overlap).norm() < 2e-11 &&
        (actual.one_electron-expected.one_electron).norm() < 2e-11 &&
        (actual.pair_kernel-expected.pair_kernel).norm() < 2e-11,
        "factor HVP lost a multistate weight, orbital response or gauge multiplier");
  }
}

void check_complete_unequal_spin_case() {
  const Fixture fixture = make_fixture(5, 2, 1);
  const auto [product_alpha, product_beta] = full_products(fixture);
  constexpr int n_structures = 7;
  Terms terms(product_alpha.size());
  for (int product = 0; product < static_cast<int>(terms.size()); ++product)
    for (int structure = 0; structure < n_structures; ++structure)
      if ((product + 3 * structure) % 11 < 3)
        terms[product].push_back({structure,
            std::sin(0.23 * (product + 1) * (structure + 2))});
  const auto cache = topology(fixture, product_alpha, product_beta);
  const xmvb::vb::StructureAction action(
      terms, n_structures, cache, fixture.overlap, fixture.one_electron,
      fixture.two_electron, fixture.n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  require(action.storage().orthogonal_direct_ci,
          "unequal complete carrier did not select direct CI");
  const Eigen::MatrixXd vectors = test_vectors(n_structures, 3);
  check_images(action.apply(vectors), reference_images(
      fixture, terms, product_alpha, product_beta, vectors));
}

void check_mixed_exchange_space() {
  const Fixture fixture = make_fixture(4, 2, 2);
  const auto [alpha, beta] = full_products(fixture);
  const int n = fixture.alpha.size();
  Terms terms(alpha.size());
  // Equal spin dimensions do not imply a common exchange sector. Include
  // even, odd, and nonsymmetric columns in one selected structure map.
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      const double x = std::sin(0.17 * (a + 1) * (b + 2));
      const double y = std::sin(0.17 * (b + 1) * (a + 2));
      terms[a * n + b] = {{0, x + y}, {1, x - y}, {2, x}};
    }
  const auto cache = topology(fixture, alpha, beta);
  const xmvb::vb::StructureAction action(
      terms, 3, cache, fixture.overlap, fixture.one_electron,
      fixture.two_electron, fixture.n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto vectors = test_vectors(3, 2);
  const auto reference = reference_images(fixture, terms, alpha, beta, vectors);
  check_images(action.apply(vectors), reference);
  check_images(action.apply_streamed(vectors,16*1024),reference);
}

void check_input_columns() {
  constexpr int n_alpha=7,width=2;
  const std::vector<std::uint8_t> support{1,1,1,1,1,1,0,1,1,1};
  std::vector<int> generated;
  const auto value=[](int beta,int alpha,int v) { return 100.0*beta+10*v+alpha+1; };
  xmvb::vb::detail::InputColumns input(n_alpha,width,3,48,support,
      [&](std::span<const int> betas) {
    require(betas.size()<=3,"input generation exceeded its bounded batch");
    std::vector<xmvb::vb::detail::ColumnTile> result;
    for (const int beta:betas) {
      generated.push_back(beta);
      auto p=std::make_shared<xmvb::vb::detail::ColumnPattern>();
      p->offsets.resize(2); p->offsets<<0,2;
      p->rows.resize(2); p->rows<<beta%n_alpha,(beta+2)%n_alpha;
      std::sort(p->rows.data(),p->rows.data()+2);
      Eigen::MatrixXd values(2,width);
      for (int v=0;v<width;++v) for (int a=0;a<2;++a)
        values(a,v)=value(beta,p->rows[a],v);
      result.push_back({std::move(p),std::move(values)});
    }
    return result;
  });
  const std::vector<int> betas{0,3,6,9},rows{0,1,3,5,6};
  Eigen::MatrixXd actual(rows.size(),betas.size()*width);
  const auto read=[&] { input.read(betas,rows,actual); };
  read();
  require(generated==std::vector<int>({0,3,9}),"reader generated an unrequested or strictly zero beta column");
  const auto check=[&] {
    for (int b=0;b<static_cast<int>(betas.size());++b)
      for (int v=0;v<width;++v) for (int a=0;a<static_cast<int>(rows.size());++a) {
        const int beta=betas[b],alpha=rows[a];
        const double expected=support[beta] && (alpha==beta%n_alpha || alpha==(beta+2)%n_alpha)
            ? value(beta,alpha,v):0.0;
        require(actual(a,b*width+v)==expected,"column eviction/gather changed a multi-RHS coefficient");
      }
  };
  check();
#pragma omp parallel num_threads(2)
#pragma omp single
  read();
  check();
  const auto before=generated.size();
  input.prepare({3});
  require(generated.size()==before && input.cache().evictions>0,
      "column hit regenerated data or the tiny-budget test did not evict");
  input.release(3);
  require(!input.cache().contains(3) && input.cache().releases==1,
      "completed input column was not released");
  input.release(3);
  require(input.cache().releases==1,"an absent column was released twice");
}

void check_exterior_team() {
  const auto strings=complete_strings(8,4);
  Eigen::MatrixXd upper=Eigen::MatrixXd::Identity(8,8),delta=Eigen::MatrixXd::Zero(8,8);
  for (int j=0;j<8;++j)
    for (int i=0;i<=j;++i) {
      upper(i,j)+=0.01*std::sin(i+2*j+1);
      delta(i,j)=0.03*std::cos(2*i+j+1);
    }
  const xmvb::vb::ExteriorOrbitalTransform transform(strings,upper);
  const auto direction=transform.direction(delta);
  const int size=strings.size();
  Eigen::MatrixXd seed(size,size);
  for (int j=0;j<size;++j)
    for (int i=0;i<size;++i) seed(i,j)=std::sin(0.2*(i+1)*(j+3));
  const auto compare=[&](auto operation) {
    Eigen::MatrixXd reference=seed,actual=seed;
    operation(reference);
#pragma omp parallel num_threads(2)
#pragma omp single
    operation(actual);
    require((actual-reference).norm()<1e-13,"exterior action changed inside an existing team");
  };
  compare([&](auto& x){transform.apply_left(&x);});
  compare([&](auto& x){transform.apply_adjoint_left(&x);});
  compare([&](auto& x){transform.apply_inverse_left(&x);});
  compare([&](auto& x){transform.apply_inverse_adjoint_left(&x);});
  compare([&](auto& x){transform.apply_right(&x);});
  compare([&](auto& x){transform.apply_adjoint_right(&x);});
  compare([&](auto& x){transform.apply_inverse_right_block(x);});
  compare([&](auto& x){transform.apply_inverse_adjoint_right_block(x);});
  const auto projection=transform.projection({0,4,19});
  compare([&](auto& x){transform.apply_adjoint_left(&x,projection);});
  const xmvb::mcscf::ProductSupport support(8,transform.occupations(),transform.occupations());
  std::vector<int> support_offsets{0}, support_rows;
  for (int beta=0;beta<size;++beta) {
    auto cursor=support.column(beta,0);
    int alpha, emitted_beta;
    bool in_support;
    while (cursor.next(alpha,emitted_beta,in_support)) {
      require(emitted_beta==beta && in_support,"depth-zero column changed its support label");
      support_rows.push_back(alpha);
    }
    std::sort(support_rows.begin()+support_offsets.back(),support_rows.end());
    support_offsets.push_back(support_rows.size());
  }
  compare([&](auto& x){transform.apply_supported_left(&x,support_offsets,support_rows);});
  compare([&](auto& x){transform.apply_supported_left(&x,support_offsets,support_rows,true);});
  Eigen::MatrixXd two_rhs(size,2*size);
  for (int j=0;j<two_rhs.cols();++j)
    for (int i=0;i<size;++i) two_rhs(i,j)=std::cos(0.17*(i+1)*(j+2));
  for (const bool adjoint : {false,true}) {
    Eigen::MatrixXd reference=two_rhs,actual=two_rhs;
    for (int first=0;first<two_rhs.cols();first+=size) {
      Eigen::MatrixXd block=two_rhs.middleCols(first,size);
      transform.apply_supported_left(&block,support_offsets,support_rows,adjoint);
      reference.middleCols(first,size)=block;
    }
#pragma omp parallel num_threads(2)
#pragma omp single
    transform.apply_supported_left(&actual,support_offsets,support_rows,adjoint);
    require((actual-reference).norm()<1e-13,
            "shared supported exterior graph differs from independent RHS actions");
  }
  // Zero primal shears are identities, but their directional coefficients
  // need not vanish. Primal shortcuts must never remove those derivative edges.
  const Eigen::MatrixXd identity=Eigen::MatrixXd::Identity(8,8);
  const xmvb::vb::ExteriorOrbitalTransform at_identity(strings,identity);
  const auto identity_direction=at_identity.direction(delta);
  Eigen::MatrixXd primal=two_rhs,tangent=Eigen::MatrixXd::Zero(size,two_rhs.cols());
  at_identity.apply_directional_left(identity_direction,&primal,&tangent);
  constexpr double h=1e-5;
  const xmvb::vb::ExteriorOrbitalTransform plus(strings,identity+h*delta),minus(strings,identity-h*delta);
  Eigen::MatrixXd plus_image=two_rhs,minus_image=two_rhs;
  plus.apply_left(&plus_image); minus.apply_left(&minus_image);
  require((primal-two_rhs).norm()<1e-13 && tangent.norm()>1e-3 &&
              (tangent-(plus_image-minus_image)/(2*h)).norm()<1e-8,
          "zero primal shears lost a nonzero orbital derivative");
  Eigen::MatrixXd x=seed,dx=0.3*seed,y=x,dy=dx;
  transform.apply_directional_left(direction,&x,&dx);
#pragma omp parallel num_threads(2)
#pragma omp single
  transform.apply_directional_left(direction,&y,&dy);
  require((x-y).norm()<1e-13 && (dx-dy).norm()<1e-13,"directional exterior team action changed");
  // A separate transform ensures an actual cache miss inside the team.
  const xmvb::vb::ExteriorOrbitalTransform fresh(strings,upper);
  std::shared_ptr<const Eigen::MatrixXd> rows;
#pragma omp parallel num_threads(2)
#pragma omp single
  rows=fresh.row_block(3,7);
  require((*rows-*transform.row_block(3,7)).norm()<1e-13,"cooperative exterior row generation changed");

  auto pattern=std::make_shared<xmvb::vb::detail::ColumnPattern>();
  pattern->offsets.resize(4); pattern->offsets<<0,2,2,3;
  pattern->rows.resize(3); pattern->rows<<0,4,2;
  Eigen::MatrixXd dense=Eigen::MatrixXd::Zero(5,6);
  dense(0,0)=1e-180; dense(4,0)=2; dense(2,2)=-3; dense(4,3)=4;
  auto tile=xmvb::vb::detail::pack_columns(dense,pattern);
  require(tile.values.rows()==3 && tile.values.cols()==2 &&
      (tile.unpack(5).array()==dense.array()).all(),"compact columns changed structural zeros or tiny entries");
  const Eigen::MatrixXd left=dense.leftCols(3),right=dense.rightCols(3);
  const auto lanes=xmvb::vb::detail::pack_columns({&left,&right},pattern);
  require((lanes.values.array()==tile.values.array()).all(),
      "direct lane packing changed compact RHS order or tiny entries");
  auto full=std::make_shared<xmvb::vb::detail::ColumnPattern>();
  full->full=true; full->offsets.resize(4); full->offsets<<0,5,10,15;
  require((xmvb::vb::detail::pack_columns({&left,&right},full).unpack(5).array()==dense.array()).all(),
      "direct lane packing changed a full column block");
}

void check_sigma_columns(const Fixture& fixture, int width) {
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, fixture.n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(
      fixture.alpha, fixture.beta, integrals);
  const auto spin_bytes=[&](int strings,int electrons)->std::size_t {
    // A single fixed-stride record array, shared by all numerical consumers.
    constexpr auto link_bytes=sizeof(xmvb::mcscf::SpinLink);
    return static_cast<std::size_t>(strings)*
        (link_bytes*electrons*(fixture.n-electrons)+electrons);
  };
  const std::size_t expected_links=spin_bytes(fixture.alpha.size(),fixture.alpha.front().size())+
      (fixture.alpha==fixture.beta ? 0:spin_bytes(fixture.beta.size(),fixture.beta.front().size()));
  const auto pairs=integrals.pair_kernel.rows();
  require(sigma.dynamic_bytes()==expected_links+2*sizeof(double)*pairs*pairs,
          "direct-CI links are not compact/shared as declared");
  const xmvb::vb::ExteriorOrbitalTransform transform(
      fixture.alpha, integrals.orbital_transform);
  const int row_begin=fixture.alpha.size()>1 ? 1:0;
  const int row_count=std::min<int>(3,fixture.alpha.size()-row_begin);
  const auto rows = transform.row_block(row_begin,row_count);
  require(rows.get() == transform.row_block(row_begin,row_count).get(),
          "accepted-point exterior rows were regenerated");
  Eigen::MatrixXd selected = Eigen::MatrixXd::Zero(row_count,fixture.alpha.size());
  selected.middleCols(row_begin,row_count).setIdentity();
  transform.apply_adjoint_right_block(selected);
  require((*rows - selected).norm() < 1e-14,
          "cached exterior rows differ from the shear transform");
  const int n_beta = static_cast<int>(fixture.beta.size());
  Eigen::MatrixXd carrier(fixture.alpha.size(), width * n_beta);
  for (int j = 0; j < carrier.cols(); ++j)
    for (int i = 0; i < carrier.rows(); ++i)
      carrier(i, j) = std::sin(0.14 * (i + 2) * (j + 1));
  Eigen::MatrixXd tiled = Eigen::MatrixXd::Zero(carrier.rows(), carrier.cols());
  const auto read = [&](const std::vector<int>& betas, const std::vector<int>& alpha_rows,
                        Eigen::Ref<Eigen::MatrixXd> block) {
        for (int beta_local = 0; beta_local < static_cast<int>(betas.size()); ++beta_local)
          for (int vector = 0; vector < width; ++vector)
            for (int row = 0; row < static_cast<int>(alpha_rows.size()); ++row)
              block(row, beta_local * width + vector) =
                  carrier(alpha_rows[row], vector * n_beta + betas[beta_local]);
      };
  const auto add = [&](const std::vector<int>& betas, const std::vector<int>& alpha_rows,
                       const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
        for (int beta_local = 0; beta_local < static_cast<int>(betas.size()); ++beta_local)
          for (int vector = 0; vector < width; ++vector)
            accumulate(beta_local, vector, 0, static_cast<int>(alpha_rows.size()),
                tiled.col(vector * n_beta + betas[beta_local]).data(), nullptr, 0);
      };
  const auto check_adjoint = [&](const xmvb::mcscf::DirectCiIntegralAdjoint& value,
                                 const xmvb::mcscf::DirectCiIntegralAdjoint& reference) {
    require((value.one_electron-reference.one_electron).norm() <
                3e-11*std::max(1.0,reference.one_electron.norm()) &&
            (value.pair_kernel-reference.pair_kernel).norm() <
                3e-11*std::max(1.0,reference.pair_kernel.norm()),
            "generated sigma/adjoint channel sweep differs from the full oracle");
  };
  std::vector<bool> released(n_beta,false);
  int release_calls=0;
  sigma.apply_tiled(width,
      [&](const auto& betas,const auto& rows,Eigen::Ref<Eigen::MatrixXd> block) {
        for (const int beta:betas)
          require(!released[beta],"sigma read a released input column");
        read(betas,rows,block);
      },[&](const auto& betas,const auto& rows,const auto& accumulate) {
        for (const int beta:betas)
          require(!released[beta],"sigma wrote a released output column");
        add(betas,rows,accumulate);
      },nullptr,0,[&](std::span<const int> betas) {
        if (!betas.empty()) ++release_calls;
        for (const int beta:betas) {
          require(!released[beta],"sigma released a column more than once");
          released[beta]=true;
        }
      });
  require(release_calls>0 && std::all_of(released.begin(),released.end(),[](bool x){return x;}),
      "sigma did not finish every input/output column lifetime");
  const Eigen::MatrixXd ordinary = sigma.apply(carrier);
  require((tiled - ordinary).norm() <=
              2e-11 * std::max(1.0, ordinary.norm()),
          "multi-column tiled direct-CI sigma differs from ordinary sigma");
  tiled.setZero();
  check_adjoint(sigma.apply_tiled_with_adjoint(width,read,add),
                sigma.integral_adjoint(carrier,carrier));
  require((tiled-ordinary).norm() < 2e-11*std::max(1.0,ordinary.norm()),
          "collecting the generated integral adjoint changed sigma");
  Eigen::MatrixXd right=carrier.array().cos().matrix();
  const xmvb::mcscf::DirectCiTileReader read_right = [&](const std::vector<int>& betas,
      const std::vector<int>& rows,Eigen::Ref<Eigen::MatrixXd> values) {
    for (int j=0;j<static_cast<int>(betas.size());++j)
      for (int v=0;v<width;++v)
        for (int a=0;a<static_cast<int>(rows.size());++a)
          values(a,j*width+v)=right(rows[a],v*n_beta+betas[j]);
  };
  tiled.setZero();
  const auto cross=sigma.integral_adjoint(carrier,right);
  check_adjoint(sigma.apply_tiled_with_adjoint(width,read,add,read_right),cross);
  require((tiled-ordinary).norm() < 2e-11*std::max(1.0,ordinary.norm()),
          "state-channel slab reuse corrupted the response sigma");
  check_adjoint(sigma.apply_tiled_with_adjoint(width,read,{},read_right),cross);
  // Each response writer must finish before the next halo read starts.
  // Include an adjoint-only middle request: it still consumes the common B,
  // but has no image callback to synchronize or trigger the next reader.
  tiled.setZero();
  int completed_writes=0;
  const xmvb::mcscf::DirectCiTileWriter ordered_add=[&](const auto& betas,
      const auto& rows,const auto& accumulate) {
    add(betas,rows,accumulate);
    ++completed_writes;
  };
  const xmvb::mcscf::DirectCiTileReader next_read=[&](const auto& betas,
      const auto& rows,Eigen::Ref<Eigen::MatrixXd> values) {
    require(completed_writes>0,"next response read before its preceding writer");
    read_right(betas,rows,values);
  };
  const xmvb::mcscf::DirectCiTileRequest sequence[]={
      {read,ordered_add},{next_read,{}},{read,ordered_add}};
  const auto sequence_adjoints=sigma.apply_tiled_responses(width,sequence,read_right);
  check_adjoint(sequence_adjoints[0],cross);
  check_adjoint(sequence_adjoints[1],sigma.integral_adjoint(right,right));
  check_adjoint(sequence_adjoints[2],cross);
  require((tiled-2*ordinary).norm()<4e-11*std::max(1.0,ordinary.norm()),
          "temporal halo refill overwrote a live image or accepted channels");
  if (fixture.n<=5) {
    // Retire only after ALL response writers have finished a generated tile.
    // The small workspace forces multiple tiles independently of residency.
    tiled.setZero();
    std::fill(released.begin(),released.end(),false);
    const auto guarded_read=[&](const auto& betas,const auto& rows,
        Eigen::Ref<Eigen::MatrixXd> values) {
      for (const int b:betas) require(!released[b],"response read after final release");
      read(betas,rows,values);
    };
    const auto guarded_write=[&](const auto& betas,const auto& rows,const auto& accumulate) {
      for (const int b:betas) require(!released[b],"temporal writer after final release");
      add(betas,rows,accumulate);
    };
    const xmvb::mcscf::DirectCiTileRequest repeated[]={
        {guarded_read,guarded_write},{guarded_read,guarded_write}};
    const auto derivatives=sigma.apply_tiled_responses(width,repeated,guarded_read,nullptr,0,
        [&](std::span<const int> columns) {
          for (const int b:columns) {
            require(!released[b],"response column released twice");
            released[b]=true;
          }
        },16*1024);
    require(std::all_of(released.begin(),released.end(),[](bool value){return value;}),
        "response did not retire all columns");
    require((tiled-2*ordinary).norm()<4e-11*std::max(1.0,ordinary.norm()),
        "retiring temporal outputs changed sigma");
    for (const auto& derivative:derivatives)
      check_adjoint(derivative,sigma.integral_adjoint(carrier,carrier));
  }
  Eigen::MatrixXd orbital(fixture.n,fixture.n);
  for (int q=0;q<fixture.n;++q)
    for (int p=0;p<fixture.n;++p) orbital(p,q)=std::sin(0.29*(p+2)*(q+1));
  Eigen::MatrixXd panel_generator=Eigen::MatrixXd::Zero(fixture.n,fixture.n);
  for (int first=0;first<n_beta;first+=3) {
    const int count=std::min(3,n_beta-first);
    Eigen::MatrixXd panel(carrier.rows(),width*count);
    for (int v=0;v<width;++v) panel.middleCols(v*count,count)=carrier.middleCols(v*n_beta+first,count);
    panel_generator+=sigma.generator_adjoint_columns(first,count,panel,
        {{right,-0.37,false},{carrier,0.23,true}});
  }
  const Eigen::MatrixXd image=-0.37*sigma.apply_one_body_generator(right,orbital)+
      0.23*sigma.apply_one_body_generator(carrier,orbital.transpose());
  require(std::abs((panel_generator.array()*orbital.array()).sum()-
                  (carrier.array()*image.array()).sum())<
              3e-11*std::max(1.0,carrier.norm()*image.norm()),
          "column-panel generator requests failed the independent forward-action identity");
  Eigen::MatrixXd team_generator;
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    team_generator=sigma.generator_adjoint_columns(0,n_beta,carrier,
        {{right,-0.37,false},{carrier,0.23,true}});
  }
  require((team_generator-panel_generator).norm()<
              3e-11*std::max(1.0,panel_generator.norm()),
          "column generator failed inside an existing OpenMP team");
  const auto upper_generator=sigma.generator_adjoint_columns(0,n_beta,carrier,
      {{right,-0.37,false},{carrier,0.23,true}},true);
  Eigen::MatrixXd upper_reference=team_generator;
  upper_reference.triangularView<Eigen::StrictlyLower>().setZero();
  require((upper_generator-upper_reference).norm()<
              3e-11*std::max(1.0,upper_reference.norm()),
          "upper generator discarded lower information before a requested transpose");
  if (fixture.alpha == fixture.beta) {
    const Eigen::MatrixXd unsymmetrized = carrier;
    for (const int parity : {1, -1}) {
      for (int v = 0; v < width; ++v) {
        const Eigen::MatrixXd block = unsymmetrized.middleCols(v * n_beta, n_beta);
        carrier.middleCols(v * n_beta, n_beta) = block + parity * block.transpose();
      }
      tiled.setZero();
      sigma.apply_tiled(width, read, add, nullptr, parity);
      // The quotient returns a representative, not the full product image.
      // Only its projection to the same parity sector must equal H X.
      for (int v = 0; v < width; ++v) {
        const Eigen::MatrixXd block = tiled.middleCols(v * n_beta, n_beta);
        tiled.middleCols(v * n_beta, n_beta) = 0.5 * (block + parity * block.transpose());
      }
      const Eigen::MatrixXd reference = sigma.apply(carrier);
      require((tiled - reference).norm() <= 2e-11 * std::max(1.0, reference.norm()),
              "exchange quotient changed the projected Hamiltonian action");
      right=0.4*carrier;
      check_adjoint(sigma.apply_tiled_with_adjoint(width,read,{},read_right,nullptr,parity),
                    sigma.integral_adjoint(carrier,right,parity));
      check_adjoint(sigma.apply_tiled_with_adjoint(width,read,{},{},nullptr,parity),
                    sigma.integral_adjoint(carrier,carrier,parity));
      tiled.setZero();
      check_adjoint(sigma.apply_tiled_with_adjoint(width,read,add,{},nullptr,parity),
                    sigma.integral_adjoint(carrier,carrier,parity));
      for (int v=0;v<width;++v) {
        const Eigen::MatrixXd block=tiled.middleCols(v*n_beta,n_beta);
        tiled.middleCols(v*n_beta,n_beta)=0.5*(block+parity*block.transpose());
      }
      require((tiled-reference).norm() < 2e-11*std::max(1.0,reference.norm()),
              "fused exchange-quotient sigma/adjoint changed the projected image");
    }
    carrier = unsymmetrized;
  }
  if (fixture.n == 5) {
    // Only output is projected; omitted output rows still feed the channels.
    std::vector<int> selected_rows;
    for (int a = 0; a < carrier.rows(); a += 2) selected_rows.push_back(a);
    Eigen::MatrixXd compact = Eigen::MatrixXd::Zero(selected_rows.size(), carrier.cols());
    tiled.setZero();
    sigma.apply_tiled(width, read,
        [&](const std::vector<int>& betas, const std::vector<int>& rows,
            const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
          for (int j = 0; j < static_cast<int>(betas.size()); ++j)
            for (int v = 0; v < width; ++v)
              accumulate(j, v, 0, rows.size(), compact.col(v * n_beta + betas[j]).data(),
                  selected_rows.data(), selected_rows.size());
        });
    for (int i = 0; i < static_cast<int>(selected_rows.size()); ++i)
      tiled.row(selected_rows[i]) = compact.row(i);
    Eigen::MatrixXd projected = ordinary;
    for (int a = 0; a < carrier.rows(); ++a)
      if (a % 2) projected.row(a).setZero();
    require((tiled - projected).norm() <= 2e-11 * std::max(1.0, projected.norm()),
            "output mask changed the unprojected sigma intermediates");
    bool read_failed = false, write_failed = false;
    bool adjoint_read_failed = false;
    try {
      sigma.apply_tiled(width,
          [](const std::vector<int>&, const std::vector<int>&, Eigen::Ref<Eigen::MatrixXd>) {
            throw std::runtime_error("reader failure");
          }, add);
    } catch (const std::runtime_error& error) {
      read_failed = std::string(error.what()) == "reader failure";
    }
    try {
      sigma.apply_tiled(width, read,
          [](const std::vector<int>&, const std::vector<int>&,
             const xmvb::mcscf::DirectCiColumnAccumulator&) {
            throw std::runtime_error("writer failure");
          });
    } catch (const std::runtime_error& error) {
      write_failed = std::string(error.what()) == "writer failure";
    }
    try {
      sigma.apply_tiled_with_adjoint(width,read,{},
          [](const std::vector<int>&,const std::vector<int>&,Eigen::Ref<Eigen::MatrixXd>) {
            throw std::runtime_error("state reader failure");
          });
    } catch (const std::runtime_error& error) {
      adjoint_read_failed=std::string(error.what())=="state reader failure";
    }
    require(read_failed && write_failed && adjoint_read_failed,
            "tile callback exceptions did not propagate out of the OpenMP team");
    bool next_read_failed=false;
    const xmvb::mcscf::DirectCiTileRequest failing_sequence[]={
      {read,add},
      {[](const auto&,const auto&,Eigen::Ref<Eigen::MatrixXd>) {
        throw std::runtime_error("next response reader failure");
      },add}};
    try {
      sigma.apply_tiled_responses(width,failing_sequence,read_right);
    } catch (const std::runtime_error& error) {
      next_read_failed=std::string(error.what())=="next response reader failure";
    }
    require(next_read_failed,"next-reader exception did not leave the OpenMP team");
  }
}

void check_prefix_words() {
  for (const int n : {4, 5, 12, 13, 16, 32, 63}) {
    const int na = n / 2, nb = n - na;
    const auto full = (std::uint64_t{1} << n) - 1;
    const auto low_a = (std::uint64_t{1} << na) - 1;
    const auto low_b = (std::uint64_t{1} << nb) - 1;
    const std::vector<std::uint64_t> alpha{low_a, full ^ low_b, full & 0xAAAAAAAAAAAAAAAAULL};
    const std::vector<std::uint64_t> beta{low_b, full ^ low_a, full ^ alpha.back()};
    const xmvb::mcscf::ProductSupport support(n, alpha, beta);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        for (int depth = 0; depth <= 1; ++depth) {
          int count = 0;
          int first=-1,last=-1;
          bool expected = true;
          for (int k = 0; k < n; ++k) {
            count += ((alpha[a] >> k) & 1) + ((beta[b] >> k) & 1);
            expected = expected && count + depth >= k + 1;
            if (count<k+1) { if (first<0) first=k; last=k; }
          }
          require(support.contains(a, b, depth) == expected,
                  "packed prefix comparison differs from scalar occupation counts");
          require(support.channel_envelope(a,b)==std::pair{first,last},
                  "packed channel envelope differs from scalar deficient prefixes");
        }
  }
}

int string_index(const Strings& strings, const std::vector<int>& occupied) {
  const auto found = std::find(strings.begin(), strings.end(), occupied);
  require(found != strings.end(), "test spin string is absent");
  return static_cast<int>(found - strings.begin());
}

bool dominates_seed(const std::vector<int>& target,
                    const std::vector<int>& seed) {
  if (target.size() != seed.size()) return false;
  for (std::size_t i = 0; i < target.size(); ++i)
    if (target[i] > seed[i]) return false;
  return true;
}

void check_product_support_oracle(
    const Fixture& fixture, const std::vector<xmvb::mcscf::ProductIndex>& products) {
  std::vector<std::uint64_t> alpha, beta;
  for (const auto& occupied : fixture.alpha) alpha.push_back(occupation_mask(occupied));
  for (const auto& occupied : fixture.beta) beta.push_back(occupation_mask(occupied));
  const int na = static_cast<int>(alpha.size());
  const int nb = static_cast<int>(beta.size());
  const xmvb::mcscf::ProductSupport support(fixture.n, alpha, beta, products);
  const auto direct = [&](int a, int b) {
    for (const int product : products)
      if (dominates_seed(fixture.alpha[a], fixture.alpha[product / nb]) &&
          dominates_seed(fixture.beta[b], fixture.beta[product % nb]))
        return true;
    return false;
  };
  std::vector<xmvb::mcscf::ProductPaths::Seed> seeds;
  for (const int product : products)
    seeds.emplace_back(alpha[product / nb], beta[product % nb]);
  const xmvb::mcscf::ProductPaths paths(fixture.n, fixture.alpha[0].size(),
                                      fixture.beta[0].size(), seeds);
  // Full masks exist only in this small independent oracle, not in paths.
  std::vector<unsigned char> generated(na * nb, 0), generated_halo(na * nb, 0);
  const auto check_paths = [&](int depth, std::vector<unsigned char>& found) {
    paths.visit(depth, [&](std::uint64_t a, std::uint64_t b, bool in_support) {
      const auto ai = std::find(alpha.begin(), alpha.end(), a) - alpha.begin();
      const auto bi = std::find(beta.begin(), beta.end(), b) - beta.begin();
      require(ai < na && bi < nb, "generated string is outside fixed-spin sector");
      require(found[ai * nb + bi]++ == 0, "duplicate product path");
      require(in_support == direct(ai, bi), "incorrect generated support label");
    });
  };
  check_paths(0, generated);
  check_paths(1, generated_halo);
  for (int depth = 0; depth <= 1; ++depth) {
    auto columns = support.cursor(depth);
    std::vector<unsigned char> visited(nb, 0);
    int b;
    while (columns.next_column(b)) {
      require(b >= 0 && b < nb && !visited[b]++, "invalid generated beta column");
      auto column = support.column(b, depth);
      int a, returned_beta;
      bool label;
      bool nonempty = false;
      while (column.next(a, returned_beta, label)) {
        require(returned_beta == b, "column cursor changed beta");
        require(label == direct(a, b), "column cursor support label mismatch");
        nonempty = true;
      }
      require(nonempty, "beta-only traversal emitted an empty column");
    }
    for (int beta_index = 0; beta_index < nb; ++beta_index) {
      bool nonempty = false;
      for (int a = 0; a < na; ++a)
        nonempty |= (depth == 0 ? generated : generated_halo)[a * nb + beta_index];
      require(static_cast<bool>(visited[beta_index]) == nonempty,
              "beta-only traversal omitted or added a column");
    }
  }
  for (int depth = 0; depth <= 1; ++depth)
    for (int b = 0; b < nb; ++b) {
      std::vector<unsigned char> column(na, 0);
      paths.visit_column(beta[b], depth, [&](auto a_mask, auto b_mask, bool in_support) {
        const auto a = std::find(alpha.begin(), alpha.end(), a_mask) - alpha.begin();
        require(a < na && b_mask == beta[b] && !column[a]++, "invalid fixed-beta path");
        require(in_support == direct(a, b), "fixed-beta support label mismatch");
      });
      for (int a = 0; a < na; ++a)
        require(column[a] == (depth == 0 ? generated : generated_halo)[a * nb + b],
                "fixed-beta generator differs from full cursor");
    }
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < nb; ++b) {
      const bool expected = direct(a, b);
      require(support.contains(a, b) == expected,
              "paired product downset differs from sorted-occupation oracle");
      require(static_cast<bool>(generated[a * nb + b]) == expected,
              "generated paths differ from sorted-occupation oracle");
      bool neighbor = expected;
      for (int other = 0; other < na; ++other)
        if (__builtin_popcountll(alpha[a] ^ alpha[other]) == 2)
          neighbor = neighbor || direct(other, b);
      for (int other = 0; other < nb; ++other)
        if (__builtin_popcountll(beta[b] ^ beta[other]) == 2)
          neighbor = neighbor || direct(a, other);
      require(support.contains(a, b, 1) == neighbor,
              "paired product generator closure differs from explicit neighbors");
      require(static_cast<bool>(generated_halo[a * nb + b]) == neighbor,
              "generated halo differs from explicit one-electron neighbors");
    }
}

void check_generic_product_support() {
  const auto check = [](const Fixture& fixture,
                        const std::vector<std::pair<std::vector<int>, std::vector<int>>>& seeds) {
    std::vector<xmvb::mcscf::ProductIndex> products;
    for (const auto& [alpha, beta] : seeds)
      products.push_back(string_index(fixture.alpha, alpha) * fixture.beta.size() +
                         string_index(fixture.beta, beta));
    check_product_support_oracle(fixture, products);
  };
  check(make_fixture(4, 2, 2), {{{1, 2}, {1, 2}}}); // Ionic seed.
  check(make_fixture(5, 2, 1),
        {{{0, 3}, {2}}, {{1, 4}, {0}}});
  const Fixture cross = make_fixture(6, 3, 3);
  const int a = string_index(cross.alpha, {0, 4, 5});
  const int b = string_index(cross.beta, {1, 2, 3});
  const int c = string_index(cross.alpha, {1, 2, 3});
  const int d = string_index(cross.beta, {0, 4, 5});
  check_product_support_oracle(cross,
      {a * static_cast<int>(cross.beta.size()) + b,
       c * static_cast<int>(cross.beta.size()) + d});
  std::vector<std::uint64_t> alpha, beta;
  for (const auto& occupied : cross.alpha) alpha.push_back(occupation_mask(occupied));
  for (const auto& occupied : cross.beta) beta.push_back(occupation_mask(occupied));
  const xmvb::mcscf::ProductSupport support(cross.n, alpha, beta,
      {a * static_cast<int>(cross.beta.size()) + b,
       c * static_cast<int>(cross.beta.size()) + d});
  require(!support.contains(a, d) && !support.contains(a, d, 1),
          "alpha and beta matched different support seeds");
  check_product_support_oracle(make_fixture(4, 2, 2), {});
  check_product_support_oracle(make_fixture(4, 2, 2), {0, 0, 35});
  check_product_support_oracle(make_fixture(4, 0, 2), {3});
  // No SpinStrings tables: a huge universe with tiny support is generated
  // directly. This is an ionic product test, not a 30-orbital covalent VB run.
  using Paths = xmvb::mcscf::ProductPaths;
  const auto low = (std::uint64_t{1} << 15) - 1;
  const Paths sparse(30, 15, 15, {{low, low}});
  const auto sparse0 = sparse.visit(0, [](auto, auto, bool) {});
  const auto sparse1 = sparse.visit(1, [](auto, auto, bool) {});
  require(sparse0.products == 1 && sparse1.products == 451 && sparse1.support == 1,
          "large-universe sparse path count is wrong");
  require(sparse1.prefixes <= 31 * sparse1.products,
          "prefix traversal is not output bounded");
  const Paths bit63(63, 1, 0, {{std::uint64_t{1} << 62, 0}});
  require(bit63.visit(1, [](auto, auto, bool) {}).products == 63,
          "63-orbital path generation failed");
}

/** @brief Independent full-size oracle for incremental byte admission. */
void check_sigma_payload(const xmvb::mcscf::detail::SigmaTilePlan& plan,
    xmvb::mcscf::detail::SigmaTileBudget budget,int n) {
  using xmvb::mcscf::detail::SigmaGatherRow;
  const std::size_t rows=plan.active.size(),width=budget.width;
  std::size_t entries=0;
  for (const int group:plan.channel_groups)
    entries+=group ? ((group-1)/n+1)*(n-1-(group-1)%n):budget.pairs;
  const std::size_t input=sizeof(double)*width*plan.halo.size()*plan.columns.size();
  const std::size_t images=sizeof(double)*width*rows*budget.pairs*(budget.slabs-budget.packed_slabs);
  const std::size_t packed=sizeof(double)*width*entries*budget.packed_slabs;
  const std::size_t indices=sizeof(int)*width*plan.halo.size()*plan.groups.size();
  const std::size_t layout=budget.packed_slabs ?
      rows*(sizeof(int)+width*(sizeof(int)+sizeof(std::size_t))):0;
  require(plan.payload_bytes()==input+images+packed+indices+layout+rows*sizeof(SigmaGatherRow),
      "incremental sigma admission differs from complete byte accounting");
}

/** @brief Admission metadata equals an independent completed-tile traversal. */
void check_sigma_metadata(const xmvb::mcscf::detail::SigmaTilePlan& plan,int parity) {
  std::vector<int> diagonal;
  std::size_t unsupported=0;
  for (const auto& group:plan.groups)
    for (int i=group.begin;i<group.end;++i) {
      unsupported+=!plan.active_support[i];
      if (parity && plan.active[i]==group.beta) diagonal.push_back(i);
    }
  require(plan.diagonal_rows==diagonal && plan.unsupported_targets==unsupported,
      "admission metadata differs from independent full tile traversal");
}

void check_sigma_tile_plan() {
  using namespace xmvb::mcscf;
  SpinConnections graph;
  graph.strings=4; graph.degree=3;
  graph.occupations.resize(1,4);
  for (int b=0;b<4;++b) {
    graph.occupations(0,b)=b;
    for (int a=0;a<4;++a) if (a!=b)
      graph.links.push_back({a,0,0,0,1.0});
  }
  const std::vector<std::uint64_t> masks{1,2,4,8};
  const ProductSupport support(4,masks,masks,{3,12});
  for (const ProductSupport* selected:std::array<const ProductSupport*,2>{nullptr,&support})
    for (int parity:{0,1,-1})
      for (const std::size_t bytes:{800U,8192U}) {
        detail::SigmaTilePlan plan(graph,graph,selected,parity,
            {2,10,2,bytes},true);
        std::array<int,16> visits{};
        std::array<bool,4> retired{};
        while (plan.next()) {
          check_sigma_payload(plan,{2,10,2,bytes},4);
          check_sigma_metadata(plan,parity);
          require(plan.payload_bytes()<=bytes,"sigma plan exceeds numeric payload budget");
          for (const int column:plan.columns)
            require(!retired[column],"sigma plan reads a retired column");
          for (int i=0;i<static_cast<int>(plan.active.size());++i) {
            const int a=plan.active[i],b=plan.groups[plan.group_index[i]].beta;
            ++visits[4*a+b];
            require(static_cast<bool>(plan.active_support[i])==(!selected || selected->contains(a,b)),
                "sigma plan changed the structural support label");
          }
          for (const int column:plan.released) {
            require(!retired[column],"sigma plan retires a column twice");
            retired[column]=true;
          }
        }
        for (int a=0;a<4;++a) for (int b=0;b<4;++b) {
          const bool expected=(!selected || selected->contains(a,b,1)) &&
              (!parity || a<b || (a==b && parity>0));
          require(visits[4*a+b]==static_cast<int>(expected),
              "sigma plan lost or duplicated a target at a tile boundary");
        }
        require(std::all_of(retired.begin(),retired.end(),[](bool value) { return value; }),
            "sigma plan omitted a final column release");
      }
}

/** @brief Full-channel images are column-major; ragged sources remain row-major. */
void check_full_channel_image_layout() {
  using namespace xmvb::mcscf::detail;
  constexpr int count=4,width=3,pairs=6,threads=2;
  Eigen::MatrixXd kernel(pairs,pairs),direction(pairs,pairs);
  for (int j=0;j<pairs;++j) for (int i=0;i<pairs;++i) {
    kernel(i,j)=0.13*(i+1)-0.07*(j+2)+0.011*i*j;
    direction(i,j)=0.09*(j+1)+0.023*i-0.004*i*j;
  }
  const std::vector<int> groups{0,1,5,0},zero_groups(count,0),diagonal_rows{1,3};
  for (int scenario=0;scenario<3;++scenario) for (bool exchange:{false,true}) {
    const bool restricted=scenario!=0;
    const auto& selected_groups=scenario==2 ? zero_groups:groups;
    const ChannelProduct basis(kernel,3,restricted),derivative(direction,3,restricted);
    ChannelLayout layout;
    layout.reset(basis,restricted ? std::span<const int>(selected_groups):std::span<const int>{},count,width);
    std::vector<double> source(layout.entries()),image(count*width*pairs,-1.0);
    for (std::size_t i=0;i<source.size();++i) source[i]=0.02*(i+1)-0.17;
    Eigen::MatrixXd expected=Eigen::MatrixXd::Zero(count*width,pairs);
    const auto accumulate=[&](const Eigen::MatrixXd& k,bool add) {
      for (int row=0;row<count*width;++row) {
        const int group=restricted ? selected_groups[row%count]:0;
        const bool diagonal=std::find(diagonal_rows.begin(),diagonal_rows.end(),row%count)!=diagonal_rows.end();
        const double scale=exchange ? (diagonal ? 1.0:2.0):1.0;
        for (int pair=0;pair<pairs;++pair) {
          double value=0.0;
          for (int channel=0;channel<basis.channels(group);++channel)
            value+=source[layout.packed(row,group)+channel]*k(pair,basis.global(group,channel));
          if (add) expected(layout.image(row),pair)+=scale*value;
          else expected(layout.image(row),pair)=scale*value;
        }
      }
    };
    for (int thread=0;thread<threads;++thread)
      multiply_channels(basis,layout,source.data(),image.data(),thread,threads,false,
          diagonal_rows,count,width,exchange);
    accumulate(kernel,false);
    Eigen::Map<const Eigen::MatrixXd> first(image.data(),count*width,pairs);
    require((first-expected).norm()<1e-12,"full-channel image layout changed the dense product");
    for (int thread=0;thread<threads;++thread)
      multiply_channels(derivative,layout,source.data(),image.data(),thread,threads,true,
          diagonal_rows,count,width,exchange);
    accumulate(direction,true);
    Eigen::Map<const Eigen::MatrixXd> combined(image.data(),count*width,pairs);
    require((combined-expected).norm()<1e-12,"full-channel forcing changed the dense product");
  }
}

/** @brief Composed beta addresses preserve grouped RHS, intervals and holes. */
void check_sparse_beta_gather() {
  const std::vector<int> halo{0,2,3,6,7,9},targets{0,2,4,5};
  const std::vector<int> index{2,-1,0,-1,3,1},sparse{1,3,6,7,10};
  constexpr int pairs=3,channel=2;
  const Eigen::MatrixXd kernel=Eigen::MatrixXd::Identity(pairs,pairs);
  using namespace xmvb::mcscf::detail;
  const ChannelProduct basis(kernel,2,true);
  const std::vector<int> groups{0,1,0,1};
  for (const int width:{1,2,3}) for (const bool grouped:{false,true}) {
    ChannelLayout layout;
    layout.reset(basis,grouped ? std::span<const int>(groups):std::span<const int>{},4,width);
    for (const bool column_major:{false,true}) {
      const int image_rows=4*width;
      Eigen::VectorXd values=Eigen::VectorXd::LinSpaced(image_rows*pairs,-0.7,1.1);
      std::vector<SigmaGatherRow> composed;
      for (const int h:targets) {
        const int image=layout.image(index[h]);
        composed.push_back({halo[h],image,width>1 ? layout.image(4+index[h])-image:0});
      }
      for (int rhs=0;rhs<width;++rhs) for (const bool selected:{false,true})
        for (int first=0;first<=static_cast<int>(halo.size());++first)
          for (int last=first;last<=static_cast<int>(halo.size());++last) {
            Eigen::VectorXd actual=Eigen::VectorXd::Constant(selected ? sparse.size():11,0.25);
            Eigen::VectorXd expected=actual;
            for (int out=0;out<expected.size();++out) {
              const int alpha=selected ? sparse[out]:out;
              for (int h=first;h<last;++h)
                if (halo[h]==alpha && index[h]>=0)
                  expected[out]-=values[column_major ? layout.image(index[h]+4*rhs)+channel*image_rows :
                      layout.image(index[h]+4*rhs)*pairs+channel];
            }
            const int begin=first<static_cast<int>(halo.size()) ? halo[first]:11;
            const int end=last>first ? halo[last-1]+1:begin;
            gather_beta_rows(values.data()+(column_major ? channel*image_rows:channel),
                column_major ? 1:pairs,rhs,-1.0,
                composed,begin,end,actual.data(),
                selected ? sparse.data():nullptr,selected ? sparse.size():0);
            require((actual-expected).norm()==0,"sparse beta gather changed a halo interval or destination support");
          }
    }
  }
}

/** @brief Ragged-channel capacity never drops nodes at a budget boundary. */
void check_packed_sigma_tile_plan() {
  using namespace xmvb::mcscf;
  const std::vector<std::uint64_t> masks{3,5,6,9,10,12};
  SpinConnections graph;
  graph.strings=6; graph.degree=4;
  graph.occupations.resize(2,6);
  for (int a=0;a<6;++a) {
    int entry=0;
    for (int p=0;p<4;++p) if ((masks[a]>>p)&1) graph.occupations(entry++,a)=p;
    for (int b=0;b<6;++b) if (__builtin_popcountll(masks[a]^masks[b])==2)
      graph.links.push_back({b,0,0,0,1.0});
  }
  const ProductSupport support(4,masks,masks);
  for (int parity:{0,1,-1}) for (int packed:{1,2})
    for (const std::size_t bytes:{2048U,8192U}) {
      detail::SigmaTilePlan plan(graph,graph,&support,parity,{2,10,packed+1,bytes,packed},true);
      std::array<int,36> visits{};
      while (plan.next()) {
        check_sigma_payload(plan,{2,10,packed+1,bytes,packed},4);
        check_sigma_metadata(plan,parity);
        require(plan.payload_bytes()<=bytes,"packed sigma plan exceeds shared budget");
        require(plan.channel_groups.size()==plan.active.size(),"missing packed channel groups");
        for (int i=0;i<static_cast<int>(plan.active.size());++i) {
          const int a=plan.active[i],b=plan.groups[plan.group_index[i]].beta;
          ++visits[6*a+b];
          const auto [first,last]=support.channel_envelope(a,b);
          require(plan.channel_groups[i]==(first<0 ? 0:1+4*first+last),
              "packed sigma planner changed the exact channel envelope");
        }
      }
      for (int a=0;a<6;++a) for (int b=0;b<6;++b)
        require(visits[6*a+b]==static_cast<int>(support.contains(a,b,1) &&
            (!parity || a<b || (a==b && parity>0))),"packed sigma plan lost or duplicated a target");
    }
}

/** @brief Test-only full snapshots certify bounded symbolic boundary replay. */
void check_symbolic_replay() {
  using namespace xmvb::mcscf;
  using namespace xmvb::mcscf::detail;
  constexpr int n=6,strings=20,pairs=n*(n+1)/2;
  std::vector<std::uint64_t> masks;
  SpinConnections graph;
  graph.strings=strings; graph.degree=9;
  graph.occupations.resize(3,strings);
  const auto occupations=complete_strings(n,3);
  require(occupations.size()==strings,"symbolic replay fixture has wrong string count");
  for (int a=0;a<strings;++a) {
    masks.push_back(occupation_mask(occupations[a]));
    for (int i=0;i<3;++i) graph.occupations(i,a)=occupations[a][i];
  }
  for (int a=0;a<strings;++a) for (int b=0;b<strings;++b)
    if (__builtin_popcountll(masks[a]^masks[b])==2)
      graph.links.push_back({b,0,0,0,1.0});
  require(graph.links.size()==static_cast<std::size_t>(strings*graph.degree),
      "symbolic replay fixture has wrong link degree");
  const auto index=[&](std::uint64_t mask) {
    const auto found=std::find(masks.begin(),masks.end(),mask);
    require(found!=masks.end(),"symbolic replay seed is absent");
    return static_cast<int>(found-masks.begin());
  };
  // One covalent and one ionic seed have overlapping but incomplete beta
  // neighborhoods. No final product is seeded, so empty columns remain.
  const ProductSupport selected(n,masks,masks,
      {index(7)*strings+index(56),index(21)*strings+index(41)});
  const ProductSupport covalent(n,masks,masks);

  struct Snapshot {
    SigmaTile tile;
    SigmaTileBoundary boundary;
    std::size_t payload;
    bool exhausted;
  };
  const auto compare=[](const SigmaTilePlan& actual,const Snapshot& expected) {
    const auto& tile=static_cast<const SigmaTile&>(actual);
    require(tile.groups.size()==expected.tile.groups.size(),
        "symbolic replay changed target group count");
    for (std::size_t i=0;i<tile.groups.size();++i) {
      const auto& lhs=tile.groups[i],&rhs=expected.tile.groups[i];
      require(lhs.beta==rhs.beta && lhs.begin==rhs.begin && lhs.end==rhs.end &&
              lhs.column==rhs.column,"symbolic replay changed target group");
    }
    require(tile.active==expected.tile.active &&
            tile.group_index==expected.tile.group_index &&
            tile.halo==expected.tile.halo && tile.columns==expected.tile.columns &&
            tile.halo_index==expected.tile.halo_index &&
            tile.column_index==expected.tile.column_index &&
            tile.released==expected.tile.released &&
            tile.active_support==expected.tile.active_support &&
            tile.channel_groups==expected.tile.channel_groups &&
            tile.diagonal_rows==expected.tile.diagonal_rows &&
            tile.unsupported_targets==expected.tile.unsupported_targets,
        "symbolic replay changed tile metadata");
    require(actual.payload_bytes()==expected.payload &&
            actual.exhausted()==expected.exhausted,
        "symbolic replay changed payload or exhausted state");
    const auto boundary=actual.boundary();
    require(boundary.targets==expected.boundary.targets &&
            boundary.payload==expected.boundary.payload &&
            boundary.full_alpha==expected.boundary.full_alpha &&
            boundary.full_beta==expected.boundary.full_beta,
        "symbolic replay changed boundary certificate");
  };
  const auto check=[&](const SpinConnections& alpha,const SpinConnections& beta,
                       const ProductSupport* support,int parity,
                       SigmaTileBudget budget,bool release,bool large) {
    SigmaTilePlan reference(alpha,beta,support,parity,budget,release);
    std::vector<Snapshot> snapshots;
    while (reference.next()) {
      check_sigma_payload(reference,budget,n);
      check_sigma_metadata(reference,parity);
      snapshots.push_back({static_cast<const SigmaTile&>(reference),
          reference.boundary(),reference.payload_bytes(),reference.exhausted()});
    }
    require(!snapshots.empty(),"symbolic replay fixture generated no tiles");
    require(large ? snapshots.size()==1 : snapshots.size()>2,
        "symbolic replay did not exercise requested tile size");
    SigmaTilePlan replay(alpha,beta,support,parity,budget,release);
    for (const auto& snapshot:snapshots) {
      require(replay.next_known(snapshot.boundary),"symbolic replay lost a tile");
      compare(replay,snapshot);
    }
    require(!replay.next() && replay.exhausted(),
        "symbolic replay did not finish at the same target boundary");
    // A feasible old partition need not be the greedy partition of a new
    // allowance. Changing free bytes alone must not change its target stream.
    auto roomy_budget=budget;
    roomy_budget.bytes+=4096;
    SigmaTilePlan roomy(alpha,beta,support,parity,roomy_budget,release);
    for (const auto& snapshot:snapshots) {
      require(roomy.next_known(snapshot.boundary),"budget change lost a reusable tile");
      compare(roomy,snapshot);
    }
    require(!roomy.next(),"budget change duplicated a reusable target");
    if (large) return;
    // Emulate a small scalar boundary cache: replay only its first one or two
    // entries, then resume ordinary admission without rewinding the cursor.
    for (int prefix:{1,2}) {
      SigmaTilePlan mixed(alpha,beta,support,parity,budget,release);
      for (int i=0;i<prefix;++i) {
        require(mixed.next_known(snapshots[i].boundary),
            "symbolic replay lost a cached prefix tile");
        compare(mixed,snapshots[i]);
      }
      for (std::size_t i=prefix;i<snapshots.size();++i) {
        require(mixed.next(),"symbolic replay lost an uncached tail tile");
        compare(mixed,snapshots[i]);
      }
      require(!mixed.next() && mixed.exhausted(),
          "symbolic replay left an uncached tail target");
    }
  };
  for (int width:{1,2,3}) for (int parity:{0,1,-1})
    for (bool release:{false,true}) for (std::size_t bytes:{4096U,4U*1024U*1024U}) {
      const bool large=bytes>4096;
      check(graph,graph,nullptr,parity,{width,pairs,2,bytes},release,large);
      check(graph,graph,&selected,parity,{width,pairs,2,bytes},release,large);
      for (int packed:{0,1,2})
        check(graph,graph,&covalent,parity,{width,pairs,packed+1,bytes,packed},release,large);
    }
  // Larger, unequal spin sectors and a permuted caller ordering exercise the
  // current-column cursor independently of exchange symmetry or COV closure.
  const auto sector=[](int electrons,bool reverse) {
    auto strings=complete_strings(12,electrons);
    if (reverse) std::reverse(strings.begin(),strings.end());
    std::vector<std::uint64_t> masks;
    SpinConnections result;
    result.strings=strings.size();result.degree=electrons*(12-electrons);
    result.occupations.resize(electrons,result.strings);
    for (int a=0;a<result.strings;++a) {
      masks.push_back(occupation_mask(strings[a]));
      for (int i=0;i<electrons;++i) result.occupations(i,a)=strings[a][i];
    }
    for (int a=0;a<result.strings;++a) for (int b=0;b<result.strings;++b)
      if (std::popcount(masks[a]^masks[b])==2)
        result.links.push_back({b,0,0,0,1.0});
    return std::pair{std::move(result),std::move(masks)};
  };
  const auto [a12,am12]=sector(6,false);
  for (bool reverse:{false,true}) {
    const auto [b12,bm12]=sector(5,reverse);
    const ProductSupport restricted(12,am12,bm12,{0,17*b12.strings+412});
    for (const auto* support:{static_cast<const ProductSupport*>(nullptr),&restricted})
      for (bool release:{false,true})
        check(a12,b12,support,0,{1,78,2,8U*1024U*1024U},release,false);
  }
}

void check_supported_sigma(int n, int n_alpha, int n_beta_electrons, int width,
                           const std::vector<xmvb::mcscf::ProductIndex>& products = {}) {
  const auto fixture = make_fixture(n, n_alpha, n_beta_electrons);
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha, fixture.beta, integrals);
  std::vector<std::uint64_t> alpha, beta;
  for (const auto& occupied : fixture.alpha) alpha.push_back(occupation_mask(occupied));
  for (const auto& occupied : fixture.beta) beta.push_back(occupation_mask(occupied));
  const xmvb::mcscf::ProductSupport support = products.empty()
      ? xmvb::mcscf::ProductSupport(n, alpha, beta)
      : xmvb::mcscf::ProductSupport(n, alpha, beta, products);
  const int na = static_cast<int>(alpha.size()), nb = static_cast<int>(beta.size());
  Eigen::MatrixXd carrier = Eigen::MatrixXd::Zero(na, width * nb);
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < nb; ++b) {
      // Independently enumerate all one-electron neighbors to certify that
      // prefix deficit <=1 is the closure, not just an assumed input mask.
      bool reachable = support.contains(a, b);
      for (int other = 0; other < na; ++other)
        if (__builtin_popcountll(alpha[a] ^ alpha[other]) == 2)
          reachable = reachable || support.contains(other, b);
      for (int other = 0; other < nb; ++other)
        if (__builtin_popcountll(beta[b] ^ beta[other]) == 2)
          reachable = reachable || support.contains(a, other);
      require(reachable == support.contains(a, b, 1), "one-generator closure is incorrect");
      if (support.contains(a, b))
        for (int v = 0; v < width; ++v)
          carrier(a, v * nb + b) = std::sin(0.17 * (a + 1) * (b + v + 2));
    }
  Eigen::MatrixXd actual = Eigen::MatrixXd::Zero(na, width * nb);
  const xmvb::mcscf::DirectCiTileReader read =
      [&](const std::vector<int>& betas, const std::vector<int>& rows,
          Eigen::Ref<Eigen::MatrixXd> values) {
        for (int beta_local = 0; beta_local < static_cast<int>(betas.size()); ++beta_local)
          for (int v = 0; v < width; ++v)
            for (int i = 0; i < static_cast<int>(rows.size()); ++i)
              values(i, beta_local * width + v) =
                  carrier(rows[i], v * nb + betas[beta_local]);
      };
  const xmvb::mcscf::DirectCiTileWriter add =
      [&](const std::vector<int>& betas, const std::vector<int>& rows,
          const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
        for (int beta_local = 0; beta_local < static_cast<int>(betas.size()); ++beta_local)
          for (int v = 0; v < width; ++v)
            accumulate(beta_local, v, 0, static_cast<int>(rows.size()),
                actual.col(v * nb + betas[beta_local]).data(), nullptr, 0);
      };
  sigma.apply_tiled(width,read,add,&support);
  auto expected = sigma.apply(carrier);
  require((actual - expected).norm() <= 2e-11 * std::max(1.0, expected.norm()),
          "generated closure sigma differs from full sigma, including outside-S outputs");
  if (n==4 && n_alpha==2 && n_beta_electrons==2) {
    actual.setZero();
    sigma.apply_tiled(width,read,add,&support,0,{},2048);
    require((actual-expected).norm()<=2e-11*std::max(1.0,expected.norm()),
        "small-budget sigma changed the Hamiltonian action");
  }
  actual.setZero();
  const auto fused=sigma.apply_tiled_with_adjoint(width,read,add,{},&support);
  const auto integral=sigma.integral_adjoint(carrier,carrier);
  std::cout << "supported_adjoint n=" << n << " na=" << n_alpha << " nb=" << n_beta_electrons
            << " sigma_error=" << (actual-expected).norm()
            << " one_error=" << (fused.one_electron-integral.one_electron).norm()
            << " pair_error=" << (fused.pair_kernel-integral.pair_kernel).norm() << '\n';
  require((actual-expected).norm() < 2e-11*std::max(1.0,expected.norm()) &&
          (fused.one_electron-integral.one_electron).norm() <
              3e-11*std::max(1.0,integral.one_electron.norm()) &&
          (fused.pair_kernel-integral.pair_kernel).norm() <
              3e-11*std::max(1.0,integral.pair_kernel.norm()),
          "support-aware sigma/adjoint dropped intermediate or outside-support contributions");
  // Both directional readers have exact depth-zero support. The shared
  // scatter must retain off-support depth-one channels for both of them.
  Eigen::MatrixXd state_carrier = Eigen::MatrixXd::Zero(na, width * nb);
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < nb; ++b)
      if (support.contains(a, b))
        for (int v = 0; v < width; ++v)
          state_carrier(a, v * nb + b) = std::cos(0.23 * (a + 2) * (b + v + 1));
  const xmvb::mcscf::DirectCiTileReader read_state =
      [&](const std::vector<int>& betas, const std::vector<int>& rows,
          Eigen::Ref<Eigen::MatrixXd> values) {
        for (int beta_local = 0; beta_local < static_cast<int>(betas.size()); ++beta_local)
          for (int v = 0; v < width; ++v)
            for (int i = 0; i < static_cast<int>(rows.size()); ++i)
              values(i, beta_local * width + v) =
                  state_carrier(rows[i], v * nb + betas[beta_local]);
      };
  Eigen::MatrixXd forcing = Eigen::MatrixXd::Zero(na, width * nb);
  const xmvb::mcscf::DirectCiTileWriter add_forcing =
      [&](const std::vector<int>& betas, const std::vector<int>& rows,
          const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
        for (int beta_local = 0; beta_local < static_cast<int>(betas.size()); ++beta_local)
          for (int v = 0; v < width; ++v)
            accumulate(beta_local, v, 0, static_cast<int>(rows.size()),
                forcing.col(v * nb + betas[beta_local]).data(), nullptr, 0);
      };
  actual.setZero();
  const xmvb::mcscf::DirectCiSigmaAction direction(sigma, integrals);
  const auto directional = sigma.apply_tiled_direction(
      width, read, read_state, direction, add, add_forcing, &support);
  const auto expected_forcing = direction.apply(state_carrier);
  const auto expected_combined = expected + expected_forcing;
  const auto expected_directional = sigma.integral_adjoint(carrier, state_carrier);
  require((actual - expected_combined).norm() <
              2e-11 * std::max(1.0, expected_combined.norm()) &&
          (forcing - expected_forcing).norm() <
              2e-11 * std::max(1.0, expected_forcing.norm()) &&
          (directional.one_electron - expected_directional.one_electron).norm() <
              3e-11 * std::max(1.0, expected_directional.one_electron.norm()) &&
          (directional.pair_kernel - expected_directional.pair_kernel).norm() <
              3e-11 * std::max(1.0, expected_directional.pair_kernel.norm()),
          "support-aware directional sigma/adjoint dropped depth-one channels");
  if (products.empty() && n_alpha == n_beta_electrons) {
    for (const int parity : {1, -1}) {
      Eigen::MatrixXd parity_carrier = carrier;
      for (int v = 0; v < width; ++v) {
        const Eigen::MatrixXd block = carrier.middleCols(v * nb, nb);
        parity_carrier.middleCols(v * nb, nb) = block + parity * block.transpose();
      }
      Eigen::MatrixXd quotient = Eigen::MatrixXd::Zero(na, width * nb);
      const xmvb::mcscf::DirectCiTileReader parity_read =
          [&](const std::vector<int>& betas, const std::vector<int>& rows,
              Eigen::Ref<Eigen::MatrixXd> values) {
            for (int j = 0; j < static_cast<int>(betas.size()); ++j)
              for (int v = 0; v < width; ++v)
                for (int i = 0; i < static_cast<int>(rows.size()); ++i)
                  values(i, j * width + v) = parity_carrier(rows[i], v * nb + betas[j]);
          };
      const xmvb::mcscf::DirectCiTileWriter parity_add =
          [&](const std::vector<int>& betas, const std::vector<int>& rows,
              const xmvb::mcscf::DirectCiColumnAccumulator& accumulate) {
            for (int j = 0; j < static_cast<int>(betas.size()); ++j)
              for (int v = 0; v < width; ++v)
                accumulate(j, v, 0, static_cast<int>(rows.size()),
                    quotient.col(v * nb + betas[j]).data(), nullptr, 0);
          };
      sigma.apply_tiled(width, parity_read, parity_add, &support, parity);
      for (int v = 0; v < width; ++v) {
        const Eigen::MatrixXd block = quotient.middleCols(v * nb, nb);
        quotient.middleCols(v * nb, nb) = 0.5 * (block + parity * block.transpose());
      }
      const auto parity_expected = sigma.apply(parity_carrier);
      require((quotient - parity_expected).norm() <
                  2e-11 * std::max(1.0, parity_expected.norm()),
              "supported exchange quotient changed the projected sigma image");
    }
  }
  Eigen::MatrixXd generator = Eigen::MatrixXd::Ones(n, n);
  const auto one_body = sigma.apply_one_body_generator(carrier, generator);
  double outside_input = 0.0;
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < nb; ++b)
      if (!support.contains(a, b)) outside_input += one_body(a, b) * one_body(a, b);
  require(outside_input > 1e-10, "fixture does not exercise intermediate-only support");

  // Cholesky's relative direction is upper triangular: its exterior action
  // stays in the downset S, although S is not a selected VB structure space.
  generator.setZero();
  for (int q=0;q<n;++q)
    for (int p=0;p<=q;++p) generator(p,q)=0.03*std::sin(0.21*(p+1)*(q+2));
  const auto tangent=sigma.apply_one_body_generator(carrier,generator);
  const Eigen::MatrixXd residual=expected-0.37*carrier;
  const Eigen::MatrixXd dr=sigma.apply(tangent)-0.37*tangent;
  Eigen::MatrixXd projected=residual,projected_dr=dr;
  double tangent_outside=0;
  for (int b=0;b<nb;++b)
    for (int a=0;a<na;++a)
      if (!support.contains(a,b))
        for (int v=0;v<width;++v) {
          tangent_outside+=std::abs(tangent(a,v*nb+b));
          projected(a,v*nb+b)=0;
          projected_dr(a,v*nb+b)=0;
        }
  require(tangent_outside<1e-13,"upper Cholesky generator left its exact downset");
  const auto full_generator=sigma.one_body_generator_adjoint(residual,carrier);
  const auto restricted_generator=sigma.one_body_generator_adjoint(projected,carrier);
  const Eigen::MatrixXd full_direction=sigma.one_body_generator_adjoint(dr,carrier)+
      sigma.one_body_generator_adjoint(residual,tangent);
  const Eigen::MatrixXd restricted_direction=sigma.one_body_generator_adjoint(projected_dr,carrier)+
      sigma.one_body_generator_adjoint(projected,tangent);
  for (int q=0;q<n;++q)
    for (int p=0;p<=q;++p) {
      require(std::abs(full_generator(p,q)-restricted_generator(p,q))<3e-11 &&
              std::abs(full_direction(p,q)-restricted_direction(p,q))<3e-11,
              "downset projection changed the upper generator VJP or its derivative");
    }
  const auto full_pullback=xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
      integrals,integral.one_electron,integral.pair_kernel,full_generator);
  const auto restricted_pullback=xmvb::vb::backpropagate_orthogonal_active_integral_adjoint(
      integrals,integral.one_electron,integral.pair_kernel,restricted_generator);
  require((full_pullback.overlap-restricted_pullback.overlap).norm()<3e-11,
          "downset projection changed the physical active-overlap pullback");
  xmvb::vb::OrthogonalActiveIntegrals delta;
  delta.orbital_transform=generator*integrals.orbital_transform;
  delta.one_electron=Eigen::MatrixXd::Zero(n,n);
  delta.pair_kernel=Eigen::MatrixXd::Zero(integrals.pair_kernel.rows(),integrals.pair_kernel.cols());
  const auto dj=sigma.integral_adjoint(tangent,carrier);
  const auto full_hvp=xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(
      integrals,delta,integral.one_electron,integral.pair_kernel,full_generator,
      2*dj.one_electron,2*dj.pair_kernel,full_direction);
  const auto restricted_hvp=xmvb::vb::backpropagate_orthogonal_active_integral_adjoint_direction(
      integrals,delta,integral.one_electron,integral.pair_kernel,restricted_generator,
      2*dj.one_electron,2*dj.pair_kernel,restricted_direction);
  require((full_hvp.overlap-restricted_hvp.overlap).norm()<3e-11,
          "downset projection changed the differentiated physical overlap pullback");
  std::cout<<"downset_pullback n="<<n<<" upper_tangent_outside="<<tangent_outside
           <<" discarded_generator_norm="<<(full_generator-restricted_generator).norm()<<'\n';

  // Independently contract the ordered E_pq E_rs Hamiltonian using only
  // generated I paths. Dense generator images are small-test oracles, not
  // proposed production storage. This checks paths that leave S and return.
  std::vector<xmvb::mcscf::ProductPaths::Seed> seeds;
  if (products.empty()) {
    const std::uint64_t full = (std::uint64_t{1} << n) - 1;
    for (const auto a : alpha) seeds.emplace_back(a, full ^ a);
  } else {
    for (const int product : products)
      seeds.emplace_back(alpha[product / nb], beta[product % nb]);
  }
  const xmvb::mcscf::ProductPaths paths(n, n_alpha, n_beta_electrons, seeds);
  std::vector<Eigen::MatrixXd> moments;
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q) {
      Eigen::MatrixXd e = Eigen::MatrixXd::Zero(n, n);
      e(p, q) = 1.0;
      moments.push_back(sigma.apply_one_body_generator(carrier, e));
    }
  const auto eri = [&](int p, int q, int r, int s) {
    return integrals.pair_kernel(xmvb::core::PairIndex::packed_pair_index(p, q),
                                xmvb::core::PairIndex::packed_pair_index(r, s));
  };
  Eigen::MatrixXd h = integrals.one_electron;
  for (int p = 0; p < n; ++p)
    for (int s = 0; s < n; ++s)
      for (int q = 0; q < n; ++q) h(p, s) -= 0.5 * eri(p, q, q, s);
  Eigen::VectorXd contracted = Eigen::VectorXd::Zero(width);
  Eigen::VectorXd halo_contribution = Eigen::VectorXd::Zero(width);
  paths.visit(1, [&](std::uint64_t a, std::uint64_t b, bool in_support) {
    const int ai = std::find(alpha.begin(), alpha.end(), a) - alpha.begin();
    const int bi = std::find(beta.begin(), beta.end(), b) - beta.begin();
    for (int v = 0; v < width; ++v) {
      const int col = v * nb + bi;
      for (int p = 0; p < n; ++p)
        for (int q = 0; q < n; ++q) {
          if (in_support)
            contracted[v] += carrier(ai, col) * h(p, q) * moments[p * n + q](ai, col);
          for (int r = 0; r < n; ++r)
            for (int s = 0; s < n; ++s) {
              const double term = 0.5 * eri(p, q, r, s) *
                  moments[q * n + p](ai, col) * moments[r * n + s](ai, col);
              contracted[v] += term;
              if (!in_support) halo_contribution[v] += term;
            }
        }
    }
  });
  require(halo_contribution.cwiseAbs().maxCoeff() > 1e-10,
          "fixture does not require leave-and-return Hamiltonian paths");
  double maximum_error = 0.0;
  for (int v = 0; v < width; ++v) {
    const double reference = carrier.middleCols(v * nb, nb)
        .cwiseProduct(expected.middleCols(v * nb, nb)).sum();
    require(std::abs(contracted[v] - reference) <= 3e-11 * std::max(1.0, std::abs(reference)),
            "generated two-generator contraction differs from direct-CI Hamiltonian");
    maximum_error = std::max(maximum_error, std::abs(contracted[v] - reference));
  }
  std::cout << std::setprecision(17) << "generated_contraction n=" << n
            << " na=" << n_alpha << " nb=" << n_beta_electrons
            << " absolute_error=" << maximum_error
            << " outside_support_contribution=" << halo_contribution.cwiseAbs().maxCoeff()
            << '\n';
}

void check_covalent_subset(int selected_count, bool add_ionic) {
  const auto fixture = make_fixture(8, 4, 4);
  const auto pairings = rumer_pairings(0, 8);
  std::vector<Pairing> selected;
  const int selected_indices[] = {0, 5, 13};
  for (int i = 0; i < selected_count; ++i)
    selected.push_back(pairings[selected_indices[i]]);
  auto expansion = covalent_rumer_terms(
      8, fixture.alpha, fixture.beta, selected);
  if (add_ionic) {
    expansion.product_alpha.push_back({0, 1, 2, 3});
    expansion.product_beta.push_back({0, 4, 5, 6});
    expansion.terms.push_back({{selected_count, 0.3}});
  }
  const int n_structures = selected_count + static_cast<int>(add_ionic);
  const auto cache = topology(
      fixture, expansion.product_alpha, expansion.product_beta);
  const xmvb::vb::StructureAction action(expansion.terms, n_structures, cache,
      fixture.overlap, fixture.one_electron, fixture.two_electron, 8, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  check_fixed_states(action);
  require(action.storage().orthogonal_direct_ci, "selected covalent fixture missed direct CI");
  const auto vectors = test_vectors(n_structures, 2);
  const auto expected = reference_images(fixture, expansion.terms,
      expansion.product_alpha, expansion.product_beta, vectors);
  check_images(action.apply(vectors), expected);
  check_images(action.apply_streamed(vectors,16*1024),expected);
  check_images(action.apply_sources(vectors), expected);
  require((action.apply_overlap(vectors) - expected.overlap).norm() <=
              2e-11 * std::max(1.0, expected.overlap.norm()),
          "selected covalent overlap action differs from reference");
}

void benchmark_selected12(int selected_count, bool compare_reference,
                          bool paths_only = false, bool gram_only = false,
                          bool sources_only = false) {
  require(selected_count == 1 || selected_count == 3 ||
          selected_count == 20 || selected_count == 132,
          "selected12 count must be 1, 3, 20, or 132");
  const Fixture fixture = make_fixture(12, 6, 6);
  const auto pairings = rumer_pairings(0, 12);
  const std::vector<Pairing> selected(pairings.begin(),
                                      pairings.begin() + selected_count);
  const auto expansion = covalent_rumer_terms(
      12, fixture.alpha, fixture.beta, selected);
  std::size_t term_count = 0;
  for (const auto& terms : expansion.terms) term_count += terms.size();
  std::map<int, int> alpha_index, beta_index;
  std::vector<std::uint64_t> alpha_masks, beta_masks;
  for (int i = 0; i < static_cast<int>(fixture.alpha.size()); ++i) {
    const int mask = occupation_mask(fixture.alpha[i]);
    alpha_index.emplace(mask, i);
    alpha_masks.push_back(mask);
  }
  for (int i = 0; i < static_cast<int>(fixture.beta.size()); ++i) {
    const int mask = occupation_mask(fixture.beta[i]);
    beta_index.emplace(mask, i);
    beta_masks.push_back(mask);
  }
  std::vector<xmvb::mcscf::ProductIndex> products;
  products.reserve(expansion.terms.size());
  for (int i = 0; i < static_cast<int>(expansion.terms.size()); ++i)
    products.push_back(alpha_index.at(occupation_mask(expansion.product_alpha[i])) *
        static_cast<int>(fixture.beta.size()) +
        beta_index.at(occupation_mask(expansion.product_beta[i])));
  std::vector<xmvb::mcscf::ProductPaths::Seed> seeds;
  for (const int product : products)
    seeds.emplace_back(alpha_masks[product / fixture.beta.size()],
                       beta_masks[product % fixture.beta.size()]);
  const auto paths_begin = Clock::now();
  const xmvb::mcscf::ProductPaths paths(12, 6, 6, seeds);
  const auto paths_built = Clock::now();
  const auto generated0 = paths.visit(0, [](auto, auto, bool) {});
  const auto paths_zero = Clock::now();
  const auto generated1 = paths.visit(1, [](auto, auto, bool) {});
  const auto paths_done = Clock::now();
  require(generated0.products == generated1.support,
          "generated halo lost the input support");
  const int index = selected_count == 1 ? 0 : selected_count == 3 ? 1 : selected_count == 20 ? 2 : 3;
  const std::size_t expected0[] = {136374, 157108, 192380, 226512};
  const std::size_t expected1[] = {523307, 559367, 603701, 608751};
  require(generated0.products == expected0[index] && generated1.products == expected1[index],
          "generated 12-orbital supports differ from the earlier exhaustive oracle");

  const auto seconds = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << std::setprecision(17)
            << "case=CAS(12,12)_selected\n"
            << "structures=" << selected_count << '\n'
            << "products=" << expansion.terms.size() << '\n'
            << "terms=" << term_count << '\n'
            << "support0_products=" << generated0.products << '\n'
            << "support1_products=" << generated1.products << '\n'
            << "paths_build_seconds=" << seconds(paths_begin, paths_built) << '\n'
            << "paths_support_seconds=" << seconds(paths_built, paths_zero) << '\n'
            << "paths_halo_seconds=" << seconds(paths_zero, paths_done) << '\n'
            << "paths_support_prefixes=" << generated0.prefixes << '\n'
            << "paths_halo_prefixes=" << generated1.prefixes << '\n'
            << "paths_halo_word_updates=" << generated1.words << '\n'
            << "paths_metadata_bytes=" << paths.metadata_bytes() << '\n'
            << "paths_workspace_bytes=" << paths.workspace_bytes(1) << '\n';
  if (paths_only) return;

  const auto cache = topology(fixture, expansion.product_alpha, expansion.product_beta);
  const xmvb::vb::StructureAction action(
      expansion.terms, selected_count, cache, fixture.overlap,
      fixture.one_electron, fixture.two_electron, 12, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  require(action.storage().orthogonal_direct_ci,
          "selected12 did not retain direct CI");

  const Eigen::MatrixXd vectors = test_vectors(selected_count, 1);
  if (gram_only) {
    const xmvb::vb::RepresentationOverlap overlap(action, fixture.alpha, fixture.beta, 12);
    Eigen::MatrixXd left = Eigen::MatrixXd::Identity(12, 12), right = left;
    Eigen::MatrixXd dl = Eigen::MatrixXd::Zero(12, 12), dr = dl;
    for (int i = 0; i < 12; ++i)
      for (int j = i; j < 12; ++j) {
        left(i, j) += 0.02 * std::sin(i + 2 * j + 1.0);
        right(i, j) += 0.03 * std::cos(2 * i + j + 1.0);
        dl(i, j) = 0.01 * std::cos(i + j + 0.5);
        dr(i, j) = 0.01 * std::sin(i + j + 0.5);
      }
    const Eigen::VectorXd v = vectors.col(0);
    const auto begin = Clock::now();
    const auto value = overlap.apply(left, right, vectors);
    const auto value_end = Clock::now();
    const auto tangent = overlap.apply_direction(left, right, dl, dr, vectors);
    const auto tangent_end = Clock::now();
    const auto gradient = overlap.adjoint(left, right, v, v);
    const auto gradient_end = Clock::now();
    const auto curvature = overlap.adjoint_direction(left, right, dl, dr, v, v);
    const auto curvature_end = Clock::now();
    constexpr double h = 1e-6;
    const auto plus = overlap.adjoint(left + h * dl, right + h * dr, v, v);
    const auto minus = overlap.adjoint(left - h * dl, right - h * dr, v, v);
    const double dual = (gradient.left.array() * dl.array()).sum() +
                        (gradient.right.array() * dr.array()).sum();
    const double dual_error = std::abs(dual - v.dot(tangent.col(0))) /
                             std::max(1.0, std::abs(dual));
    const double curvature_error = std::max(
        (curvature.left - (plus.left - minus.left) / (2 * h)).norm(),
        (curvature.right - (plus.right - minus.right) / (2 * h)).norm()) /
        std::max({1.0, curvature.left.norm(), curvature.right.norm()});
    require(value.allFinite() && dual_error < 2e-10 && curvature_error < 2e-7,
            "CAS(12,12) selected Gram derivative validation failed");
    const auto prepared=overlap.prepare_metric(right);
    const Eigen::MatrixXd block=test_vectors(selected_count,3);
    Eigen::MatrixXd streamed(selected_count,block.cols());
    // Independent contraction order: each RHS stays coefficient-weighted in
    // the source factors instead of the common-basis selected Gram build.
    for (int j=0;j<block.cols();++j)
      streamed.col(j)=overlap.apply(right,right,block.col(j));
    const double metric_error=(prepared(block)-streamed).norm()/std::max(1.0,streamed.norm());
    require(metric_error<2e-12,"CAS(12,12) prepared selected metric differs from streamed action");
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    std::cout << "gram_action_seconds=" << seconds(begin, value_end) << '\n'
              << "gram_tangent_seconds=" << seconds(value_end, tangent_end) << '\n'
              << "gram_adjoint_seconds=" << seconds(tangent_end, gradient_end) << '\n'
              << "gram_curvature_seconds=" << seconds(gradient_end, curvature_end) << '\n'
              << "gram_duality_error=" << dual_error << '\n'
              << "gram_curvature_fd_error=" << curvature_error << '\n'
              << "prepared_metric_error=" << metric_error << '\n'
              << "max_rss_kib=" << usage.ru_maxrss << '\n';
    return;
  }
  const auto start = Clock::now();
  const auto actual = sources_only ? action.apply_sources(vectors) : action.apply(vectors);
  const auto cold = Clock::now();
  const auto warm = sources_only ? action.apply_sources(vectors) : action.apply(vectors);
  const auto warmed = Clock::now();
  check_images(warm, actual);
  xmvb::vb::StructureActionResult reference;
  if (compare_reference) {
    reference = reference_images(fixture, expansion.terms,
        expansion.product_alpha, expansion.product_beta, vectors);
    check_images(actual, reference);
    require((action.apply_overlap(vectors) - reference.overlap).norm() <
        2e-11 * std::max(1.0, reference.overlap.norm()),
        "source-planned metric-only action differs from full reference");
  }
  const double denominator = vectors.col(0).dot(actual.overlap.col(0));
  require(denominator > 0.0, "selected12 Rayleigh denominator is nonpositive");
  const double energy =
      vectors.col(0).dot(actual.hamiltonian.col(0)) / denominator;
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  std::cout << "cold_apply_seconds=" << seconds(start, cold) << '\n'
            << "warm_apply_seconds=" << seconds(cold, warmed) << '\n'
            << "expansions=" << warm.tiles.expansions << '\n'
            << "structure_projections=" << warm.tiles.structure_projections << '\n'
            << "input_panels=" << warm.tiles.input_panels << '\n'
            << "hamiltonian_projections=" << warm.tiles.hamiltonian_projections << '\n'
            << "overlap_projections=" << warm.tiles.overlap_projections << '\n'
            << "input_evictions=" << warm.tiles.input_evictions << '\n'
            << "output_evictions=" << warm.tiles.output_evictions << '\n'
            << "input_resident_bytes=" << warm.tiles.input_resident_bytes << '\n'
            << "output_resident_bytes=" << warm.tiles.output_resident_bytes << '\n'
            << "source_tiles=" << warm.tiles.source_tiles << '\n'
            << "source_columns=" << warm.tiles.source_columns << '\n'
            << "channel_blocks=" << warm.tiles.channel_blocks << '\n'
            << "source_workspace_bound=" << warm.tiles.source_workspace_bound << '\n'
            << "max_rss_kib=" << usage.ru_maxrss << '\n'
            << "rayleigh_energy=" << energy << '\n'
            << "reference_checked=" << static_cast<int>(compare_reference) << '\n';
  if (compare_reference)
    std::cout << "hamiltonian_reference_error="
              << (actual.hamiltonian - reference.hamiltonian).norm() << '\n'
              << "overlap_reference_error="
              << (actual.overlap - reference.overlap).norm() << '\n';
}

/** @brief Complete response consumer; native and oracle runs stay separate. */
void benchmark_structure_response(int n,int selected_count,bool compare_reference,bool single_sweep=false) {
  require(selected_count == 1 || selected_count == 2 || selected_count == 3 ||
          selected_count == 20 || selected_count == 132,
          "structure-response count must be 1, 2, 3, 20, or 132");
  const auto fixture = make_fixture(n,n/2,n/2);
  const auto pairings = rumer_pairings(0,n);
  const auto expansion = covalent_rumer_terms(n,fixture.alpha,fixture.beta,
      std::vector<Pairing>(pairings.begin(),pairings.begin()+selected_count));
  const auto cache = topology(fixture,expansion.product_alpha,expansion.product_beta);
  const xmvb::vb::StructureAction action(expansion.terms,selected_count,cache,fixture.overlap,
      fixture.one_electron,fixture.two_electron,n,nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  Eigen::MatrixXd c = test_vectors(selected_count,1)/std::sqrt(selected_count);
  Eigen::MatrixXd q = 0.3*c.array().sin().matrix();
  if (n == 14 && selected_count > 1) {
    const auto mc = action.apply_overlap(c);
    q.noalias() -= c * (q.col(0).dot(mc.col(0))/c.col(0).dot(mc.col(0)));
    require(q.norm() > 1e-8,"target response has no nontrivial CI tangent");
    const auto mq = action.apply_overlap(q);
    q *= 0.2/std::sqrt(q.col(0).dot(mq.col(0)));
    require(std::abs(c.col(0).dot(action.apply_overlap(q).col(0))) < 1e-12,
            "target response retained the normalization direction");
  }
  const auto begin = Clock::now();
  const auto accepted = action.prepare_active_adjoint(c,{1.0},{-5.3});
  const auto prepared = Clock::now();
  auto cold = action.apply_response(accepted,q);
  const auto end_cold = Clock::now();
  auto warm = single_sweep ? cold : action.apply_response(accepted,q);
  const auto end_warm = Clock::now();
  const auto source_columns = warm.images.tiles.source_columns;
  const auto workspace = warm.images.tiles.source_workspace_bound;
  const auto images = warm.images;
  const Eigen::MatrixXd mu = Eigen::MatrixXd::Constant(1,1,0.017);
  const auto gradient = action.finish_response(accepted,std::move(warm),mu);
  const auto cold_gradient = action.finish_response(accepted,std::move(cold),mu);
  require((gradient.overlap-cold_gradient.overlap).norm() < 1e-12,
          "warm structure response differs from the cold call");
  rusage usage{};
  require(getrusage(RUSAGE_SELF,&usage) == 0,"getrusage failed");
  const auto native_rss = usage.ru_maxrss;
  double reference_seconds = 0, image_error = 0, adjoint_error = 0;
  if (compare_reference) {
    const auto reference_images = ::reference_images(fixture,expansion.terms,
        expansion.product_alpha,expansion.product_beta,q);
    check_images(images,reference_images);
    image_error = (images.hamiltonian-reference_images.hamiltonian).norm();
    const auto product_state = explicit_product_state(fixture,cache,action,accepted);
    const auto start = Clock::now();
    // Independent complete-carrier differential, not the migrated tile sweep.
    const auto reference = explicit_response(fixture,cache,action,product_state,q,mu);
    reference_seconds = std::chrono::duration<double>(Clock::now()-start).count();
    adjoint_error = std::max({(gradient.overlap-reference.overlap).norm(),
        (gradient.one_electron-reference.one_electron).norm(),
        (gradient.pair_kernel-reference.pair_kernel).norm()});
    require(adjoint_error < 3e-10*std::max({1.0,reference.overlap.norm(),
        reference.one_electron.norm(),reference.pair_kernel.norm()}),
        "structure response differs from the current product reference");
  }
  const auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b-a).count(); };
  std::cout << std::setprecision(17)
            << "case=CAS(" << n << ',' << n << ")_structure_response\nstructures=" << selected_count << '\n'
            << "orbitals=" << n << '\n'
            << "prepare_seconds=" << seconds(begin,prepared) << '\n'
            << "cold_response_seconds=" << seconds(prepared,end_cold) << '\n';
  if (!single_sweep) std::cout << "warm_response_seconds=" << seconds(end_cold,end_warm) << '\n';
  std::cout << "single_sweep=" << single_sweep << '\n'
            << "source_columns=" << source_columns << '\n'
            << "source_workspace_bound=" << workspace << '\n'
            << "input_resident_bytes=" << images.tiles.input_resident_bytes << '\n'
            << "output_resident_bytes=" << images.tiles.output_resident_bytes << '\n'
            << "input_panels=" << images.tiles.input_panels << '\n'
            << "input_evictions=" << images.tiles.input_evictions << '\n'
            << "output_evictions=" << images.tiles.output_evictions << '\n'
            << "integral_adjoint_sweeps=" << images.tiles.integral_adjoint_sweeps << '\n'
            << "checksum=" << images.hamiltonian.sum()+gradient.overlap.sum()+
                gradient.one_electron.sum()+gradient.pair_kernel.sum() << '\n'
            << "native_max_rss_kib=" << native_rss << '\n'
            << "reference_checked=" << compare_reference << '\n'
            << "reference_adjoint_only_seconds=" << reference_seconds << '\n'
            << "image_error=" << image_error << '\n'
            << "adjoint_error=" << adjoint_error << '\n';
}

/** @brief Accepted gradient and complete orbital direction, with separate oracles. */
void benchmark_source_state(int n, bool reference, bool accepted_only = false, int selected = 0) {
  require(n == 12 || n == 14 || (n == 16 && accepted_only),"unsupported source-state benchmark");
  const auto fixture = make_fixture(n,n/2,n/2);
  const int count = selected ? selected : (n == 14 ? 2 : 1);
  const auto pairings = rumer_pairings(0,n);
  const auto expansion = covalent_rumer_terms(n,fixture.alpha,fixture.beta,
      std::vector<Pairing>(pairings.begin(),pairings.begin()+count));
  const auto cache = topology(fixture,expansion.product_alpha,expansion.product_beta);
  const xmvb::vb::StructureAction action(expansion.terms,count,cache,fixture.overlap,
      fixture.one_electron,fixture.two_electron,n,nullptr,xmvb::vb::StructureActionMode::ActionOnly);
  const Eigen::MatrixXd c = test_vectors(count,1)/std::sqrt(count);
  constexpr double energy = -5.3;
  const auto start = Clock::now();
  const auto state = action.prepare_active_adjoint(c,{1.0},{energy});
  const auto gradient = action.active_integral_adjoint(state);
  const auto prepared = Clock::now();
  std::size_t accepted_product_doubles = 0;
  if (state.product_images)
    for (const auto& panels:state.product_images->states)
      for (const auto& panel:panels) accepted_product_doubles+=panel.values.size();
  Eigen::MatrixXd ds(n,n), dh(n,n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) {
      ds(i,j) = ds(j,i) = 0.0004*std::cos(0.13*(i+2)*(j+1));
      dh(i,j) = dh(j,i) = 0.0007*std::sin(0.17*(i+1)*(j+3));
    }
  const int pairs = xmvb::integrals::packed_active_pair_count(n);
  std::vector<double> dg(fixture.two_electron.packed_active_two_electron_integrals.size());
  for (int i = 0; i < static_cast<int>(dg.size()); ++i) dg[i] = 0.00001*std::cos(0.19*(i+1));
  Eigen::MatrixXd dg_full(pairs,pairs);
  for (int j = 0; j < pairs; ++j)
    for (int i = 0; i < pairs; ++i)
      dg_full(i,j) = dg[xmvb::core::PairIndex::packed_pair_of_pairs_index(i,j)];
  const std::vector<double> overlap(ds.data(),ds.data()+ds.size()), one(dh.data(),dh.data()+dh.size());
  xmvb::vb::StructureIntegralDirection direction;
  xmvb::vb::StructureActionResult images;
  xmvb::vb::StructureActiveIntegralAdjoint hvp;
  double image_seconds = 0, hvp_seconds = 0;
  if (!accepted_only) {
    const auto a = Clock::now();
    images = action.apply_integral_direction(state,overlap,one,dg,&direction);
    const auto b = Clock::now();
    hvp = action.active_integral_adjoint_direction(state,nullptr,nullptr,overlap,one,dg,&direction,true);
    const auto end = Clock::now();
    image_seconds = std::chrono::duration<double>(b-a).count();
    hvp_seconds = std::chrono::duration<double>(end-b).count();
    if (n==12 && selected==132) {
      require(!state.source_factors,"CAS12 COV132 missed the product-jet route");
      const Eigen::MatrixXd q=0.03*c.array().cos().matrix();
      require(q.norm()>0.0,"CAS12 COV132 joint CI response is zero");
      check_joint_product_jet(action,state,q,overlap,one,dg);
    }
  }
  constexpr double step = 2e-5;
  const auto lagrangian = [&](double scale) {
    auto moved = fixture;
    for (int i = 0; i < static_cast<int>(moved.overlap.size()); ++i) moved.overlap[i] += scale*ds.data()[i];
    moved.one_electron += scale*dh;
    for (int i = 0; i < static_cast<int>(dg.size()); ++i)
      moved.two_electron.packed_active_two_electron_integrals[i] += scale*dg[i];
    const xmvb::vb::StructureAction shifted(expansion.terms,count,cache,moved.overlap,
        moved.one_electron,moved.two_electron,n,nullptr,xmvb::vb::StructureActionMode::ActionOnly);
    const auto value = shifted.apply(c);
    return c.col(0).dot(value.hamiltonian.col(0)-energy*value.overlap.col(0));
  };
  const double slope = (gradient.overlap.array()*ds.array()).sum()+
      (gradient.one_electron.array()*dh.array()).sum()+
      (gradient.pair_kernel.array()*dg_full.array()).sum();
  const double fd_error = std::abs(slope-(lagrangian(step)-lagrangian(-step))/(2*step));
  require(fd_error < 2e-7,"source accepted gradient violates the Lagrangian directional derivative");
  double hvp_fd_error=0;
  if (selected && !accepted_only) {
    const auto moved_gradient=[&](double scale) {
      auto moved=fixture;
      for (std::size_t i=0;i<moved.overlap.size();++i) moved.overlap[i]+=scale*ds.data()[i];
      moved.one_electron+=scale*dh;
      for (std::size_t i=0;i<dg.size();++i)
        moved.two_electron.packed_active_two_electron_integrals[i]+=scale*dg[i];
      const xmvb::vb::StructureAction shifted(expansion.terms,count,cache,moved.overlap,
          moved.one_electron,moved.two_electron,n,nullptr,xmvb::vb::StructureActionMode::ActionOnly);
      return shifted.active_integral_adjoint(shifted.prepare_active_adjoint(c,{1.0},{energy}));
    };
    const auto plus=moved_gradient(step),minus=moved_gradient(-step);
    hvp_fd_error=std::max({(hvp.overlap-(plus.overlap-minus.overlap)/(2*step)).norm(),
        (hvp.one_electron-(plus.one_electron-minus.one_electron)/(2*step)).norm(),
        (hvp.pair_kernel-(plus.pair_kernel-minus.pair_kernel)/(2*step)).norm()});
    require(hvp_fd_error<2e-7,"product jet HVP violates the gradient directional derivative");
  }
  rusage usage{};
  require(getrusage(RUSAGE_SELF,&usage) == 0,"getrusage failed");
  const auto native_rss = usage.ru_maxrss;
  double gradient_error = 0, image_error = 0, hvp_error = 0;
  double reference_image_seconds = 0, reference_hvp_seconds = 0;
  if (reference) {
    auto dense = explicit_product_state(fixture,cache,action,state);
    const auto integrals = xmvb::vb::orthogonalize_active_integrals(
        fixture.overlap,fixture.one_electron,fixture.two_electron,n);
    const xmvb::mcscf::DirectCiSigmaAction sigma(cache.alpha_reuse_table.unique_determinants,
        cache.beta_reuse_table.unique_determinants,integrals);
    const auto integral = sigma.integral_adjoint(dense.orthogonal_coefficients[0],dense.orthogonal_coefficients[0]);
    dense.one_electron_gradient = integral.one_electron; dense.pair_kernel_gradient = integral.pair_kernel;
    dense.generator_gradient = 2*sigma.one_body_generator_adjoint(dense.residuals[0],dense.orthogonal_coefficients[0]);
    const auto expected = action.active_integral_adjoint(dense);
    gradient_error = std::max({(gradient.overlap-expected.overlap).norm(),
        (gradient.one_electron-expected.one_electron).norm(),(gradient.pair_kernel-expected.pair_kernel).norm()});
    xmvb::vb::StructureIntegralDirection dense_direction;
    const auto a = Clock::now();
    const auto reference_images = action.apply_integral_direction(dense,overlap,one,dg,&dense_direction);
    const auto b = Clock::now();
    const auto expected_hvp = action.active_integral_adjoint_direction(dense,nullptr,nullptr,
        overlap,one,dg,&dense_direction,true);
    const auto end = Clock::now();
    reference_image_seconds = std::chrono::duration<double>(b-a).count();
    reference_hvp_seconds = std::chrono::duration<double>(end-b).count();
    image_error = std::max((images.hamiltonian-reference_images.hamiltonian).norm(),
        (images.overlap-reference_images.overlap).norm());
    hvp_error = std::max({(hvp.overlap-expected_hvp.overlap).norm(),
        (hvp.one_electron-expected_hvp.one_electron).norm(),(hvp.pair_kernel-expected_hvp.pair_kernel).norm()});
    require(std::max({gradient_error,image_error,hvp_error}) < 3e-10,
            "source accepted/orbital chain differs from the independent product oracle");
  }
  std::cout << std::setprecision(17) << "orbitals=" << n << "\nstructures=" << count << '\n'
            << "source_order=" << state.source_factors << "\nhvp_fd_error=" << hvp_fd_error << '\n'
            << "accepted_image_cache_bytes=" << (state.product_images ? state.product_images->bytes:0) << '\n'
            << "accepted_seconds=" << std::chrono::duration<double>(prepared-start).count() << '\n'
            << "direction_checked=" << !accepted_only << "\ndirection_seconds=" << image_seconds << '\n'
            << "hvp_seconds=" << hvp_seconds << "\nlagrangian_fd_error=" << fd_error << '\n'
            << "accepted_product_doubles=" << accepted_product_doubles << "\ndirection_product_doubles="
            << 0 << '\n'
            << "direction_workspace_bound=" << images.tiles.source_workspace_bound << '\n'
            << "direction_source_columns=" << images.tiles.source_columns << '\n'
            << "cached_generator_doubles=" << direction.generator_direction.size() << '\n'
            << "native_max_rss_kib=" << native_rss << "\nreference_checked=" << reference << '\n'
            << "gradient_error=" << gradient_error << "\nimage_error=" << image_error << "\nhvp_error=" << hvp_error << '\n'
            << "reference_direction_seconds=" << reference_image_seconds << '\n'
            << "reference_hvp_seconds=" << reference_hvp_seconds << '\n';
}

/** @brief A target-size single covalent structure, with optional full-carrier oracle. */
void benchmark_source_size(int n, bool compare_reference) {
  require(n == 12 || n == 14, "source-size benchmark supports 12 or 14 orbitals");
  const auto begin = Clock::now();
  const auto fixture = make_fixture(n, n / 2, n / 2);
  Pairing adjacent;
  for (int i = 0; i < n; i += 2) adjacent.emplace_back(i, i + 1);
  const auto expansion = covalent_rumer_terms(n, fixture.alpha, fixture.beta, {adjacent});
  const auto cache = topology(fixture, expansion.product_alpha, expansion.product_beta);
  const auto built = Clock::now();
  const xmvb::vb::StructureAction action(expansion.terms, 1, cache,
      fixture.overlap, fixture.one_electron, fixture.two_electron, n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto constructed = Clock::now();
  const Eigen::MatrixXd vectors = Eigen::MatrixXd::Ones(1, 1);
  const auto cold = action.apply(vectors);
  const auto applied = Clock::now();
  const auto warm = action.apply(vectors);
  const auto finished = Clock::now();
  check_images(cold, warm);
  xmvb::vb::StructureActionResult reference;
  if (compare_reference) {
    reference = reference_images(fixture, expansion.terms,
        expansion.product_alpha, expansion.product_beta, vectors);
    check_images(warm, reference);
  }
  const auto checked = Clock::now();
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  const auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b-a).count(); };
  std::cout << std::setprecision(17)
            << "orbitals=" << n << '\n'
            << "alpha_strings=" << fixture.alpha.size() << '\n'
            << "beta_strings=" << fixture.beta.size() << '\n'
            << "products=" << expansion.terms.size() << '\n'
            << "prepare_seconds=" << seconds(begin, built) << '\n'
            << "construct_seconds=" << seconds(built, constructed) << '\n'
            << "cold_apply_seconds=" << seconds(constructed, applied) << '\n'
            << "warm_apply_seconds=" << seconds(applied, finished) << '\n'
            << "reference_seconds=" << seconds(finished, checked) << '\n'
            << "source_tiles=" << warm.tiles.source_tiles << '\n'
            << "source_columns=" << warm.tiles.source_columns << '\n'
            << "channel_blocks=" << warm.tiles.channel_blocks << '\n'
            << "source_workspace_bound=" << warm.tiles.source_workspace_bound << '\n'
            << "rayleigh_energy=" << warm.hamiltonian(0,0) / warm.overlap(0,0) << '\n'
            << "max_rss_kib=" << usage.ru_maxrss << '\n';
  if (compare_reference)
    std::cout << "hamiltonian_reference_error=" << (warm.hamiltonian-reference.hamiltonian).norm() << '\n'
              << "overlap_reference_error=" << (warm.overlap-reference.overlap).norm() << '\n';
}

/** @brief Arbitrary cross support: two exact sources instead of U beta columns.
 * This is an algebraic selected-space stress test, not a molecular benchmark.
 */
void benchmark_source_cover(int n, bool compare_reference) {
  require(n == 12 || n == 14, "source-cover benchmark supports 12/14 orbitals");
  const auto fixture = make_fixture(n, n / 2, n / 2);
  const int nb = fixture.beta.size();
  Terms terms;
  Strings alpha, beta;
  for (int spin = 0; spin < 2; ++spin)
    for (int string = 1; string < nb; ++string) {
      alpha.push_back(fixture.alpha[spin ? string : 0]);
      beta.push_back(fixture.beta[spin ? 0 : string]);
      terms.push_back({{spin, std::sin(0.17 * (string + 1)) / std::sqrt(nb)}});
    }
  const auto cache = topology(fixture, alpha, beta);
  const auto begin = Clock::now();
  const xmvb::vb::StructureAction action(terms, 2, cache, fixture.overlap,
      fixture.one_electron, fixture.two_electron, n, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  const auto built = Clock::now();
  const auto vectors = test_vectors(2, 1);
  const auto cold = action.apply(vectors);
  const auto first = Clock::now();
  const auto warm = action.apply(vectors);
  const auto finished = Clock::now();
  check_images(cold, warm);
  require(warm.tiles.source_columns == 2, "large cross support did not select two sources");
  xmvb::vb::StructureActionResult reference;
  if (compare_reference) {
    reference = reference_images(fixture, terms, alpha, beta, vectors);
    check_images(warm, reference);
  }
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  const auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b-a).count(); };
  std::cout << std::setprecision(17)
            << "orbitals=" << n << '\n' << "structures=2\n"
            << "alpha_strings=" << nb << '\n' << "beta_strings=" << nb << '\n'
            << "original_beta_sources=" << nb << '\n' << "products=" << terms.size() << '\n'
            << "construct_seconds=" << seconds(begin, built) << '\n'
            << "cold_apply_seconds=" << seconds(built, first) << '\n'
            << "warm_apply_seconds=" << seconds(first, finished) << '\n'
            << "source_columns=" << warm.tiles.source_columns << '\n'
            << "source_workspace_bound=" << warm.tiles.source_workspace_bound << '\n'
            << "rayleigh_energy=" << vectors.col(0).dot(warm.hamiltonian.col(0)) /
                vectors.col(0).dot(warm.overlap.col(0)) << '\n'
            << "reference_checked=" << compare_reference << '\n'
            << "max_rss_kib=" << usage.ru_maxrss << '\n';
  if (compare_reference)
    std::cout << "hamiltonian_reference_error=" << (warm.hamiltonian-reference.hamiltonian).norm() << '\n'
              << "overlap_reference_error=" << (warm.overlap-reference.overlap).norm() << '\n';
}

/** @brief Generated-support sigma/VJP probe; the native process has no product array. */
void benchmark_tile_adjoint(bool oracle) {
  constexpr int n=12;
  const auto fixture=make_fixture(n,n/2,n/2);
  const auto integrals=xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap,fixture.one_electron,fixture.two_electron,n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha,fixture.beta,integrals);
  std::vector<std::uint64_t> alpha,beta;
  for (const auto& string:fixture.alpha) alpha.push_back(occupation_mask(string));
  for (const auto& string:fixture.beta) beta.push_back(occupation_mask(string));
  const xmvb::mcscf::ProductSupport support(n,alpha,beta);
  const int na=alpha.size(),nb=beta.size();
  const double scale=1.0/std::sqrt(static_cast<double>(na)*nb);
  const auto coefficient=[&](int a,int b) {
    return support.contains(a,b) ? scale*std::sin(0.013*(a+1)*(b+2)) : 0.0;
  };
  const auto read=[&](const std::vector<int>& betas,const std::vector<int>& rows,
      Eigen::Ref<Eigen::MatrixXd> values) {
    for (int j=0;j<static_cast<int>(betas.size());++j)
      for (int a=0;a<static_cast<int>(rows.size());++a)
        values(a,j)=coefficient(rows[a],betas[j]);
  };
  Eigen::VectorXd column(na);
  double energy=0;
  const auto begin=Clock::now();
  const auto gradient=sigma.apply_tiled_with_adjoint(1,read,
      [&](const std::vector<int>& betas,const std::vector<int>& rows,
          const xmvb::mcscf::DirectCiColumnAccumulator& add) {
        for (int j=0;j<static_cast<int>(betas.size());++j) {
          column.setZero();
          add(j,0,0,rows.size(),column.data(),nullptr,0);
          for (const int a:rows) energy+=column[a]*coefficient(a,betas[j]);
        }
      },{},&support);
  const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
  const double adjoint_energy=(gradient.one_electron.array()*integrals.one_electron.array()).sum()+
      (gradient.pair_kernel.array()*integrals.pair_kernel.array()).sum();
  require(std::abs(energy-adjoint_energy)<3e-11*std::max(1.0,std::abs(energy)),
          "streamed energy differs from the contracted integral cotangent");
  double sigma_error=0,one_error=0,pair_error=0,reference_seconds=0;
  if (oracle) {
    Eigen::MatrixXd carrier(na,nb);
    for (int b=0;b<nb;++b)
      for (int a=0;a<na;++a) carrier(a,b)=coefficient(a,b);
    const auto start=Clock::now();
    const auto reference=sigma.apply_with_adjoint(carrier,carrier);
    reference_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    sigma_error=std::abs(energy-(carrier.array()*reference.sigma.array()).sum());
    one_error=(gradient.one_electron-reference.integrals.one_electron).norm();
    pair_error=(gradient.pair_kernel-reference.integrals.pair_kernel).norm();
    require(std::max({sigma_error,one_error,pair_error})<3e-11,
            "generated-support sigma/VJP differs from the complete-carrier oracle");
  }
  rusage usage{};
  require(getrusage(RUSAGE_SELF,&usage)==0,"getrusage failed");
  std::cout<<std::setprecision(17)<<"orbitals="<<n<<'\n'
      <<"alpha_strings="<<na<<'\n'<<"beta_strings="<<nb<<'\n'
      <<"sigma_adjoint_seconds="<<elapsed<<'\n'<<"bilinear_energy="<<energy<<'\n'
      <<"energy_adjoint_error="<<std::abs(energy-adjoint_energy)<<'\n'
      <<"reference_checked="<<oracle<<'\n'<<"reference_seconds="<<reference_seconds<<'\n'
      <<"sigma_error="<<sigma_error<<'\n'<<"one_electron_error="<<one_error<<'\n'
      <<"pair_kernel_error="<<pair_error<<'\n'<<"max_rss_kib="<<usage.ru_maxrss<<'\n';
}

/** @brief Physical covalent-state integral adjoint with bounded spin factors. */
void benchmark_factor_adjoint(int n, bool compare_reference, bool directional = false) {
  require(n == 12 || n == 14, "factor adjoint benchmark supports 12/14 orbitals");
  const auto fixture = make_fixture(n, n / 2, n / 2);
  Pairing adjacent;
  for (int i = 0; i < n; i += 2) adjacent.emplace_back(i, i + 1);
  const auto expansion = covalent_rumer_terms(n, fixture.alpha, fixture.beta, {adjacent});
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha, fixture.beta, integrals);
  const xmvb::vb::ExteriorOrbitalTransform ta(fixture.alpha, integrals.orbital_transform);
  const xmvb::vb::ExteriorOrbitalTransform tb(fixture.beta, integrals.orbital_transform);
  std::map<int, int> index;
  for (int a = 0; a < static_cast<int>(fixture.alpha.size()); ++a)
    index.emplace(occupation_mask(fixture.alpha[a]), a);
  const int na = fixture.alpha.size(), nb = fixture.beta.size(), parity = (n / 2) % 2 ? -1 : 1;
  std::vector<xmvb::mcscf::ProductIndex> products;
  Eigen::VectorXd coefficients(expansion.terms.size());
  for (int group = 0; group < static_cast<int>(expansion.terms.size()); ++group) {
    products.push_back(index.at(occupation_mask(expansion.product_alpha[group])) * nb +
        index.at(occupation_mask(expansion.product_beta[group])));
    coefficients[group] = expansion.terms[group][0].coefficient;
  }
  const xmvb::vb::SpinSources representatives(products, na, nb, parity);
  const auto bra = representatives.orthogonal_source(products, coefficients, ta, tb, true);
  const auto ket = representatives.orthogonal_source(products, coefficients, ta, tb);
  Eigen::MatrixXd generator = Eigen::MatrixXd::Zero(n,n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i)
      generator(i,j) = 0.01 * std::sin(0.17 * (i+1) * (j+2));
  const auto left = directional ? xmvb::mcscf::generator_factors(sigma, bra, generator) : bra;
  const auto begin = Clock::now();
  const auto cold = xmvb::mcscf::factor_integral_adjoint(sigma, left, ket);
  const auto first = Clock::now();
  const auto warm = xmvb::mcscf::factor_integral_adjoint(sigma, left, ket);
  const auto finished = Clock::now();
  require((cold.one_electron-warm.one_electron).norm() < 1e-12 &&
      (cold.pair_kernel-warm.pair_kernel).norm() < 1e-12, "factor adjoint is not repeatable");
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  const long native_rss = usage.ru_maxrss;
  xmvb::mcscf::DirectCiIntegralAdjoint reference;
  double reference_seconds = 0;
  if (compare_reference) {
    Eigen::MatrixXd a(na, bra.count), b(nb, bra.count);
    bra.load(0, bra.count, a, b);
    const Eigen::MatrixXd carrier = a * b.transpose();
    const Eigen::MatrixXd delta = directional ? sigma.apply_one_body_generator(carrier, generator) : carrier;
    const auto start = Clock::now();
    reference = sigma.integral_adjoint(delta, carrier, parity);
    reference_seconds = std::chrono::duration<double>(Clock::now()-start).count();
    require((warm.one_electron-reference.one_electron).norm() < 2e-10 &&
        (warm.pair_kernel-reference.pair_kernel).norm() < 2e-10,
        "covalent factor adjoint differs from complete-carrier reference");
  }
  std::cout << std::setprecision(17)
            << "orbitals=" << n << '\n' << "structures=1\n"
            << "alpha_strings=" << na << '\n' << "beta_strings=" << nb << '\n'
            << "bra_sources=" << left.count << '\n' << "ket_sources=" << ket.count << '\n'
            << "directional=" << directional << '\n'
            << "cold_adjoint_seconds=" << std::chrono::duration<double>(first-begin).count() << '\n'
            << "warm_adjoint_seconds=" << std::chrono::duration<double>(finished-first).count() << '\n'
            << "reference_seconds=" << reference_seconds << '\n'
            << "native_max_rss_kib=" << native_rss << '\n'
            << "reference_checked=" << compare_reference << '\n';
  if (compare_reference)
    std::cout << "one_electron_error=" << (warm.one_electron-reference.one_electron).norm() << '\n'
              << "pair_kernel_error=" << (warm.pair_kernel-reference.pair_kernel).norm() << '\n';
}

/** @brief Generator/residual adjoints; full carriers are reference-process only. */
void benchmark_factor_generators(int n, bool compare_reference, bool check_direction = false) {
  require(n == 12 || n == 14 || n == 16, "factor generator benchmark supports 12/14/16 orbitals");
  const auto fixture = make_fixture(n,n/2,n/2);
  Pairing adjacent;
  for (int i = 0; i < n; i += 2) adjacent.emplace_back(i,i+1);
  const auto expansion = covalent_rumer_terms(n,fixture.alpha,fixture.beta,{adjacent});
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap,fixture.one_electron,fixture.two_electron,n);
  const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha,fixture.beta,integrals);
  const xmvb::vb::ExteriorOrbitalTransform ta(fixture.alpha,integrals.orbital_transform);
  const xmvb::vb::ExteriorOrbitalTransform tb(fixture.beta,integrals.orbital_transform);
  std::map<int,int> indices;
  for (int i = 0; i < static_cast<int>(fixture.alpha.size()); ++i)
    indices.emplace(occupation_mask(fixture.alpha[i]),i);
  const int na = fixture.alpha.size(), nb = fixture.beta.size(), parity = (n/2)%2 ? -1 : 1;
  std::vector<xmvb::mcscf::ProductIndex> products;
  Eigen::VectorXd coefficients(expansion.terms.size());
  for (int i = 0; i < static_cast<int>(expansion.terms.size()); ++i) {
    products.push_back(indices.at(occupation_mask(expansion.product_alpha[i])) * nb +
        indices.at(occupation_mask(expansion.product_beta[i])));
    coefficients[i] = expansion.terms[i][0].coefficient;
  }
  const xmvb::vb::SpinSources plan(products,na,nb,parity);
  const auto bra = plan.orthogonal_source(products,coefficients,ta,tb,true);
  const auto ket = plan.orthogonal_source(products,coefficients,ta,tb);
  const auto begin = Clock::now();
  const auto cold_g = xmvb::mcscf::factor_generator_adjoint(sigma,bra,ket);
  const auto middle = Clock::now();
  const auto g = xmvb::mcscf::factor_generator_adjoint(sigma,bra,ket);
  const auto before_r = Clock::now();
  const auto cold_r = xmvb::mcscf::factor_residual_generator_adjoint(sigma,bra,ket,0.37);
  const auto warm_start = Clock::now();
  const auto r = xmvb::mcscf::factor_residual_generator_adjoint(sigma,bra,ket,0.37);
  const auto finished = Clock::now();
  require((g-cold_g).norm() < 1e-12 && (r-cold_r).norm() < 1e-12,
          "factor generator kernels are not repeatable");
  rusage usage{};
  require(getrusage(RUSAGE_SELF,&usage) == 0, "getrusage failed");
  const auto native_rss = usage.ru_maxrss;
  double sigma_seconds = 0, g_seconds = 0, r_seconds = 0, g_error = 0, r_error = 0;
  double direction_seconds = 0, directional_g_error = 0, directional_r_error = 0;
  if (check_direction) {
    Eigen::MatrixXd kappa(n,n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) kappa(i,j) = 0.02 * std::sin(0.13 * (i+1) * (j+3));
    const auto tangent = xmvb::mcscf::generator_factors(sigma,ket,kappa);
    const auto start = Clock::now();
    const auto integral = xmvb::mcscf::factor_integral_adjoint(sigma,bra,tangent);
    const double h = (integral.one_electron.array()*integrals.one_electron.array()).sum() +
        (integral.pair_kernel.array()*integrals.pair_kernel.array()).sum();
    // Number-operator trace gives <Y,G(kappa)Y> without a product carrier.
    const double overlap = xmvb::mcscf::factor_generator_adjoint(sigma,bra,tangent).trace()/n;
    directional_g_error = std::abs((g.array()*kappa.array()).sum()-overlap);
    directional_r_error = std::abs((r.array()*kappa.array()).sum()-(h-0.37*overlap));
    direction_seconds = std::chrono::duration<double>(Clock::now()-start).count();
    require(directional_g_error < 2e-10 && directional_r_error < 2e-10,
            "large factor generator violates its independent integral/number contraction");
  }
  if (compare_reference) {
    Eigen::MatrixXd a(na,bra.count), b(nb,bra.count);
    bra.load(0,bra.count,a,b);
    const Eigen::MatrixXd y = a * b.transpose();
    const auto g_start = Clock::now();
    const auto expected_g = sigma.one_body_generator_adjoint(y,y);
    const auto sigma_start = Clock::now();
    const Eigen::MatrixXd residual = sigma.apply(y,parity) - 0.37 * y;
    const auto r_start = Clock::now();
    const auto expected_r = sigma.one_body_generator_adjoint(residual,y);
    const auto reference_end = Clock::now();
    g_seconds = std::chrono::duration<double>(sigma_start-g_start).count();
    sigma_seconds = std::chrono::duration<double>(r_start-sigma_start).count();
    r_seconds = std::chrono::duration<double>(reference_end-r_start).count();
    g_error = (g-expected_g).norm(); r_error = (r-expected_r).norm();
    require(g_error < 2e-10 && r_error < 2e-10,
            "physical factor generator differs from complete-carrier reference");
  }
  std::cout << std::setprecision(17) << "orbitals=" << n << '\n'
            << "bra_sources=" << bra.count << '\n' << "ket_sources=" << ket.count << '\n'
            << "cold_generator_seconds=" << std::chrono::duration<double>(middle-begin).count() << '\n'
            << "warm_generator_seconds=" << std::chrono::duration<double>(before_r-middle).count() << '\n'
            << "cold_residual_seconds=" << std::chrono::duration<double>(warm_start-before_r).count() << '\n'
            << "warm_residual_seconds=" << std::chrono::duration<double>(finished-warm_start).count() << '\n'
            << "generator_reference_seconds=" << g_seconds << '\n'
            << "sigma_reference_seconds=" << sigma_seconds << '\n'
            << "residual_generator_reference_seconds=" << r_seconds << '\n'
            << "residual_total_reference_seconds=" << sigma_seconds+r_seconds << '\n'
            << "generator_error=" << g_error << '\n' << "residual_error=" << r_error << '\n'
            << "native_max_rss_kib=" << native_rss << '\n'
            << "directional_checked=" << check_direction << '\n'
            << "directional_generator_error=" << directional_g_error << '\n'
            << "directional_residual_error=" << directional_r_error << '\n'
            << "directional_reference_seconds=" << direction_seconds << '\n'
            << "reference_checked=" << compare_reference << '\n';
}

void benchmark12(bool compare_reference) {
  const auto start = Clock::now();
  const Fixture fixture = make_fixture(12, 6, 6);
  const auto pairings = rumer_pairings(0, 12);
  require(pairings.size() == 132, "CAS(12,12) Rumer count is not 132");
  require(fixture.alpha.size() == 924 && fixture.beta.size() == 924,
          "CAS(12,12) complete spin-string count is not 924");
  const CovalentExpansion expansion = covalent_rumer_terms(
      12, fixture.alpha, fixture.beta, pairings);
  int term_count = 0;
  for (const auto& product : expansion.terms) term_count += product.size();
  require(term_count == 132 * 64, "CAS(12,12) expansion is not 64 terms per structure");
  require(expansion.terms.size() == 924,
          "CAS(12,12) covalent support is not 924 spin products");
  const auto cache = topology(
      fixture, expansion.product_alpha, expansion.product_beta);
  const auto prepared = Clock::now();
  const xmvb::vb::StructureAction action(
      expansion.terms, 132, cache, fixture.overlap, fixture.one_electron,
      fixture.two_electron, 12, nullptr,
      xmvb::vb::StructureActionMode::ActionOnly);
  require(action.storage().orthogonal_direct_ci,
          "CAS(12,12) did not retain direct CI");
  const auto constructed = Clock::now();
  const Eigen::MatrixXd vectors = test_vectors(132, 1);
  const auto actual = action.apply(vectors);
  const auto applied = Clock::now();
  const auto warm = action.apply(vectors);
  const auto warmed = Clock::now();
  check_images(warm, actual);
  xmvb::vb::StructureActionResult reference;
  if (compare_reference) {
    reference = reference_images(
        fixture, expansion.terms, expansion.product_alpha,
        expansion.product_beta, vectors);
    check_images(actual, reference);
  }
  const auto referenced = Clock::now();
  const double denominator = vectors.col(0).dot(actual.overlap.col(0));
  require(denominator > 0.0, "CAS(12,12) Rayleigh denominator is nonpositive");
  const double energy =
      vectors.col(0).dot(actual.hamiltonian.col(0)) / denominator;
  require(std::isfinite(energy), "CAS(12,12) Rayleigh energy is nonfinite");
  rusage usage{};
  std::vector<std::uint64_t> alpha_masks, beta_masks;
  for (const auto& a : fixture.alpha) alpha_masks.push_back(occupation_mask(a));
  for (const auto& b : fixture.beta) beta_masks.push_back(occupation_mask(b));
  const xmvb::mcscf::ProductSupport support(12, alpha_masks, beta_masks);
  std::size_t supported = 0, closure = 0;
  for (int a = 0; a < 924; ++a)
    for (int b = 0; b < 924; ++b) {
      supported += support.contains(a, b);
      closure += support.contains(a, b, 1);
    }
  require(supported == 226512 && closure == 608751,
          "CAS(12,12) support/closure count differs from the reflection formula");
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  const auto seconds = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << std::setprecision(17)
            << "case=CAS(12,12)_fixed_orbital_covalent_kernel\n"
            << "alpha_strings=924 beta_strings=924 structures=132 products=924 terms=8448\n"
            << "supported_products=" << supported << '\n'
            << "generator_closure_products=" << closure << '\n'
            << "prepare_seconds=" << seconds(start, prepared) << '\n'
            << "construct_seconds=" << seconds(prepared, constructed) << '\n'
            << "apply_seconds=" << seconds(constructed, applied) << '\n'
            << "warm_apply_seconds=" << seconds(applied, warmed) << '\n'
            << "reference_seconds="
            << (compare_reference ? seconds(warmed, referenced) : 0.0) << '\n'
            << "reference_checked=" << (compare_reference ? 1 : 0) << '\n'
            << "max_rss_kib=" << usage.ru_maxrss << '\n'
            << "rayleigh_energy=" << energy << '\n';
  if (compare_reference)
    std::cout << "hamiltonian_reference_error="
              << (actual.hamiltonian - reference.hamiltonian).norm() << '\n'
              << "overlap_reference_error="
              << (actual.overlap - reference.overlap).norm() << '\n';
}

/** @brief Test-only dense carrier oracle for temporal direct-CI tile reuse. */
void benchmark_temporal12() {
  const auto fixture = make_fixture(12, 6, 6);
  const auto integrals = xmvb::vb::orthogonalize_active_integrals(
      fixture.overlap, fixture.one_electron, fixture.two_electron, 12);
  const xmvb::mcscf::DirectCiSigmaAction sigma(fixture.alpha, fixture.beta, integrals);
  auto delta_integrals = integrals;
  delta_integrals.one_electron *= 0.07;
  delta_integrals.pair_kernel *= -0.03;
  const xmvb::mcscf::DirectCiSigmaAction delta_sigma(sigma, delta_integrals);
  std::vector<std::uint64_t> masks;
  for (const auto& occupied : fixture.alpha) masks.push_back(occupation_mask(occupied));
  const xmvb::mcscf::ProductSupport support(12, masks, masks);
  const int count = static_cast<int>(masks.size());
  require(count == 924, "temporal12 requires the CAS(12,12) spin strings");
  std::size_t supported = 0, closure = 0;
  for (int b = 0; b < count; ++b)
    for (int a = 0; a < count; ++a) {
      supported += support.contains(a, b);
      closure += support.contains(a, b, 1);
    }
  require(supported == 226512 && closure == 608751,
          "temporal12 covalent support counts changed");

  // Complete carriers are reproducible test/oracle data only; production
  // direct-CI readers generate bounded panels instead of storing these arrays.
  Eigen::MatrixXd state = Eigen::MatrixXd::Zero(count, count);
  Eigen::MatrixXd ci = state, orbital = state;
  const double scale = 1.0 / std::sqrt(static_cast<double>(supported));
  for (int b = 0; b < count; ++b)
    for (int a = 0; a <= b; ++a) {
      if (!support.contains(a, b)) continue;
      const double s = scale * std::cos(0.013 * (a + 1) * (b + 2));
      const double q = scale * std::sin(0.017 * (a + 2) * (b + 1));
      const double p = scale * std::cos(0.019 * (a + 3) * (b + 1));
      state(a, b) = state(b, a) = s;
      ci(a, b) = ci(b, a) = q;
      orbital(a, b) = orbital(b, a) = p;
    }
  struct Counts { int state = 0, ci = 0, orbital = 0; } separate, shared;
  const auto read = [&](const Eigen::MatrixXd& source, int& calls)
      -> xmvb::mcscf::DirectCiTileReader {
    return [&, source_ptr = &source, calls_ptr = &calls](
        const std::vector<int>& betas, const std::vector<int>& rows,
        Eigen::Ref<Eigen::MatrixXd> values) {
      ++*calls_ptr;
      const int workers = xmvb::coarse_openmp_thread_count(
          xmvb::cooperative_openmp_thread_count(), rows.size() * betas.size());
      xmvb::cooperative_openmp_for(betas.size(), workers, [&](std::ptrdiff_t j) {
        for (int i = 0; i < static_cast<int>(rows.size()); ++i)
          values(i, j) = (*source_ptr)(rows[i], betas[j]);
      });
    };
  };
  const auto write = [&](Eigen::MatrixXd& target)
      -> xmvb::mcscf::DirectCiTileWriter {
    return [target_ptr = &target](const std::vector<int>& betas,
        const std::vector<int>& rows,
        const xmvb::mcscf::DirectCiColumnAccumulator& add) {
      const int workers = xmvb::coarse_openmp_thread_count(
          xmvb::cooperative_openmp_thread_count(), rows.size() * betas.size());
      xmvb::cooperative_openmp_for(betas.size(), workers, [&](std::ptrdiff_t j) {
        add(j, 0, 0, static_cast<int>(rows.size()),
            target_ptr->col(betas[j]).data(), nullptr, 0);
      });
    };
  };
  // Dense outputs are retained solely for full-image test comparisons.
  Eigen::MatrixXd old_ci = Eigen::MatrixXd::Zero(count, count), old_orb = old_ci,
                  old_force = old_ci, new_ci = old_ci, new_orb = old_ci,
                  new_force = old_ci;
  const std::array<xmvb::mcscf::DirectCiTileRequest, 2> requests{{
      {read(ci, shared.ci), write(new_ci), nullptr, {}},
      {read(orbital, shared.orbital), write(new_orb), &delta_sigma, write(new_force)}}};
  // Prime both current APIs; do not compare a cold separate path to warm reuse.
  sigma.apply_tiled_with_adjoint(1,read(ci,separate.ci),write(old_ci),
      read(state,separate.state),&support,1);
  sigma.apply_tiled_direction(1,read(orbital,separate.orbital),
      read(state,separate.state),delta_sigma,write(old_orb),write(old_force),&support,1);
  sigma.apply_tiled_responses(1,requests,read(state,shared.state),&support,1);
  old_ci.setZero(); old_orb.setZero(); old_force.setZero();
  new_ci.setZero(); new_orb.setZero(); new_force.setZero();
  separate={}; shared={};
  const auto start = Clock::now();
  const auto old_ci_adjoint = sigma.apply_tiled_with_adjoint(1,
      read(ci, separate.ci), write(old_ci), read(state, separate.state), &support, 1);
  const auto ci_done = Clock::now();
  const auto old_orb_adjoint = sigma.apply_tiled_direction(1,
      read(orbital, separate.orbital), read(state, separate.state), delta_sigma,
      write(old_orb), write(old_force), &support, 1);
  const auto separate_done = Clock::now();
  const auto joint = sigma.apply_tiled_responses(1, requests,
      read(state, shared.state), &support, 1);
  const auto shared_done = Clock::now();
  require(joint.size() == requests.size() && separate.ci > 0 &&
              separate.orbital > 0 &&
              separate.state == separate.ci + separate.orbital &&
              shared.state > 0 && shared.state == shared.ci &&
              shared.state == shared.orbital,
          "temporal12 did not share one accepted-state read per generated tile");
  const auto relative = [](const Eigen::MatrixXd& value,
                           const Eigen::MatrixXd& reference) {
    return (value - reference).norm() / std::max(1.0, reference.norm());
  };
  const std::array<double, 7> errors{{
      relative(new_ci, old_ci), relative(new_orb, old_orb),
      relative(new_force, old_force),
      relative(joint[0].one_electron, old_ci_adjoint.one_electron),
      relative(joint[0].pair_kernel, old_ci_adjoint.pair_kernel),
      relative(joint[1].one_electron, old_orb_adjoint.one_electron),
      relative(joint[1].pair_kernel, old_orb_adjoint.pair_kernel)}};
  require(std::all_of(errors.begin(), errors.end(),
          [](double error) { return error < 3e-11; }),
          "temporal12 images or integral adjoints differ from separate actions");
  rusage usage{};
  require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
  const auto seconds = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  std::cout << std::setprecision(17)
            << "case=temporal12_test_oracle_dense_carriers\n"
            << "alpha_strings=" << count << " beta_strings=" << count << '\n'
            << "supported_products=" << supported << '\n'
            << "generator_closure_products=" << closure << '\n'
            << "separate_ci_seconds=" << seconds(start, ci_done) << '\n'
            << "separate_orbital_seconds=" << seconds(ci_done, separate_done) << '\n'
            << "separate_total_seconds=" << seconds(start, separate_done) << '\n'
            << "shared_seconds=" << seconds(separate_done, shared_done) << '\n'
            << "shared_faster=" << (seconds(separate_done, shared_done) <
                                    seconds(start, separate_done)) << '\n'
            << "separate_ci_reads=" << separate.ci << '\n'
            << "separate_orbital_reads=" << separate.orbital << '\n'
            << "separate_state_reads=" << separate.state << '\n'
            << "shared_ci_reads=" << shared.ci << '\n'
            << "shared_orbital_reads=" << shared.orbital << '\n'
            << "shared_state_reads=" << shared.state << '\n'
            << "ci_image_relative_error=" << errors[0] << '\n'
            << "orbital_image_relative_error=" << errors[1] << '\n'
            << "orbital_forcing_relative_error=" << errors[2] << '\n'
            << "ci_one_electron_relative_error=" << errors[3] << '\n'
            << "ci_pair_kernel_relative_error=" << errors[4] << '\n'
            << "orbital_one_electron_relative_error=" << errors[5] << '\n'
            << "orbital_pair_kernel_relative_error=" << errors[6] << '\n'
            << "max_rss_kib=" << usage.ru_maxrss << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--channel-layout") {
      check_full_channel_image_layout();
      std::cout << "channel layout: PASS\n";
    } else if (argc == 2 && std::string(argv[1]) == "--symbolic-replay") {
      check_symbolic_replay();
      std::cout << "symbolic replay: PASS\n";
    } else if (argc == 2 && std::string(argv[1]) == "--benchmark12") {
      benchmark12(true);
    } else if (argc == 2 && std::string(argv[1]) == "--temporal12") {
      benchmark_temporal12();
    } else if (argc == 2 &&
               std::string(argv[1]) == "--benchmark12-no-reference") {
      benchmark12(false);
    } else if (argc == 3 && std::string(argv[1]) == "--selected12") {
      benchmark_selected12(std::stoi(argv[2]), false);
    } else if (argc == 3 && std::string(argv[1]) == "--selected12-parity") {
      benchmark_selected12(std::stoi(argv[2]), true);
    } else if (argc == 3 && std::string(argv[1]) == "--sources12") {
      benchmark_selected12(std::stoi(argv[2]), true, false, false, true);
    } else if (argc == 3 && std::string(argv[1]) == "--structure-response12") {
      benchmark_structure_response(12,std::stoi(argv[2]),false);
    } else if (argc == 3 && std::string(argv[1]) == "--structure-response12-reference") {
      benchmark_structure_response(12,std::stoi(argv[2]),true);
    } else if (argc==3 && (std::string(argv[1])=="--structure-response14-once" ||
                          std::string(argv[1])=="--structure-response14-once-reference")) {
      benchmark_structure_response(14,std::stoi(argv[2]),
          std::string(argv[1])=="--structure-response14-once-reference",true);
    } else if (argc == 3 && std::string(argv[1]) == "--structure-response14") {
      benchmark_structure_response(14,std::stoi(argv[2]),false);
    } else if (argc == 3 && std::string(argv[1]) == "--structure-response14-reference") {
      benchmark_structure_response(14,std::stoi(argv[2]),true);
    } else if (argc == 3 && std::string(argv[1]) == "--product-jet12") {
      benchmark_source_state(12,false,false,std::stoi(argv[2]));
    } else if (argc == 3 && std::string(argv[1]) == "--source-state") {
      benchmark_source_state(std::stoi(argv[2]),false);
    } else if (argc == 3 && std::string(argv[1]) == "--source-state-reference") {
      benchmark_source_state(std::stoi(argv[2]),true);
    } else if (argc == 2 && std::string(argv[1]) == "--source-accepted16") {
      benchmark_source_state(16,false,true);
    } else if (argc == 3 && std::string(argv[1]) == "--source-size") {
      benchmark_source_size(std::stoi(argv[2]), false);
    } else if (argc == 3 && std::string(argv[1]) == "--source-size-parity") {
      benchmark_source_size(std::stoi(argv[2]), true);
    } else if (argc == 3 && std::string(argv[1]) == "--source-cover-size") {
      benchmark_source_cover(std::stoi(argv[2]), false);
    } else if (argc == 3 && std::string(argv[1]) == "--source-cover-reference") {
      benchmark_source_cover(std::stoi(argv[2]), true);
    } else if (argc == 3 && std::string(argv[1]) == "--factor-adjoint") {
      benchmark_factor_adjoint(std::stoi(argv[2]), false);
    } else if (argc == 3 && std::string(argv[1]) == "--factor-adjoint-reference") {
      benchmark_factor_adjoint(std::stoi(argv[2]), true);
    } else if (argc == 3 && std::string(argv[1]) == "--factor-direction") {
      benchmark_factor_adjoint(std::stoi(argv[2]), false, true);
    } else if (argc == 3 && std::string(argv[1]) == "--factor-direction-reference") {
      benchmark_factor_adjoint(std::stoi(argv[2]), true, true);
    } else if (argc == 3 && std::string(argv[1]) == "--factor-generator") {
      benchmark_factor_generators(std::stoi(argv[2]), false);
    } else if (argc == 3 && std::string(argv[1]) == "--factor-generator-reference") {
      benchmark_factor_generators(std::stoi(argv[2]), true);
    } else if (argc == 2 && std::string(argv[1]) == "--factor-generator-check16") {
      benchmark_factor_generators(16, false, true);
    } else if (argc == 3 && std::string(argv[1]) == "--paths12") {
      benchmark_selected12(std::stoi(argv[2]), false, true);
    } else if (argc == 3 && std::string(argv[1]) == "--gram12") {
      benchmark_selected12(std::stoi(argv[2]), false, false, true);
    } else if (argc==2 && (std::string(argv[1])=="--tile-adjoint12" ||
                           std::string(argv[1])=="--tile-adjoint12-reference")) {
      benchmark_tile_adjoint(std::string(argv[1])=="--tile-adjoint12-reference");
    } else if (argc == 1) {
      const auto vacuum = xmvb::vb::build_orthogonal_spin_topology(
          Strings{{}}, Strings{{}}, 0);
      require(vacuum.n_unique_alpha == 1 && vacuum.n_unique_beta == 1 &&
                  vacuum.alpha_reuse_table.determinant_to_unique_id[0] == 0 &&
                  vacuum.beta_reuse_table.determinant_to_unique_id[0] == 0,
              "zero-active-space topology must retain the unique vacuum");
      check_rumer_case(4);
      check_rumer_case(6);
      check_selected_unequal_spin_case();
      check_ionic_sources();
      check_source_cover();
      check_source_tangents();
      check_mixed_source_images();
      for (int parity : {0,1,-1}) check_factor_structure_response(parity);
      check_factor_structure_response(0,true);
      for (int parity:{0,1,-1}) check_product_response(parity,false);
      check_product_response(0,true);
      {
        // The complete derivative chain, not only bare sigma: force multiple
        // generated tiles while the narrow-panel cases also force evictions.
        const auto previous=xmvb::workspace::limit();
        xmvb::workspace::set_limit(32*1024);
        for (int parity:{0,1,-1}) check_product_response(parity,false);
        check_product_response(0,true);
        xmvb::workspace::set_limit(previous);
      }
      check_broad_product_projection(false);
      check_broad_product_projection(true);
      check_broad_product_projection(true,true);
      for (int parity:{-1,1}) check_broad_product_projection(false,false,parity);
      require(!xmvb::vb::detail::prefer_ordered_beta(32,10,5,10,4) &&
          xmvb::vb::detail::prefer_ordered_beta(48,10,5,10,5),
          "beta contraction planner counted unreachable columns");
      check_factor_accepted_gradient();
      check_factor_accepted_gradient(true);
      for (int parity : {0,1,-1}) check_scalar_source_direction(parity);
      check_scalar_source_direction(0,true);
      check_factor_state_average();
      check_complete_unequal_spin_case();
      check_mixed_exchange_space();
      check_input_columns();
      check_exterior_team();
      check_sigma_columns(make_fixture(5, 2, 1), 3);
      check_sigma_columns(make_fixture(4, 2, 2), 3);
      check_sigma_columns(make_fixture(4, 0, 1), 2);
      check_sigma_columns(make_fixture(4, 4, 1), 2);
      check_sigma_columns(make_fixture(10, 5, 4), 2);
      check_sigma_tile_plan();
      check_symbolic_replay();
      check_full_channel_image_layout();
      check_sparse_beta_gather();
      check_packed_sigma_tile_plan();
      check_prefix_words();
      for (const int width:{1,2,3}) check_supported_sigma(4, 2, 2, width);
      check_supported_sigma(5, 3, 2, 3);
      check_supported_sigma(6, 3, 3, 1);
      // Enough halo scratch for the grouped per-worker Gram, including
      // both exchange parities, multiple RHS and independent HVP adjoints.
      check_supported_sigma(8, 4, 4, 2);
      check_generic_product_support();
      check_supported_sigma(5, 2, 1, 2, {3 * 5 + 2, 6 * 5 + 0});
      check_covalent_subset(1, false);
      check_covalent_subset(3, false);
      check_covalent_subset(3, true);
      std::cout << "structure tiles: PASS\n";
    } else {
      throw std::invalid_argument(
          "usage: test_structure_tiles [--channel-layout|--symbolic-replay|--benchmark12|--benchmark12-no-reference|--temporal12|"
          "--selected12 N|--selected12-parity N|--sources12 N|--paths12 N|--gram12 N|"
          "--structure-response12 N|--structure-response12-reference N|"
          "--structure-response14 N|--structure-response14-reference N|"
          "--source-state N|--source-state-reference N|--source-accepted16|"
          "--source-size N|--source-size-parity N|--source-cover-size N|--source-cover-reference N|"
          "--factor-adjoint N|--factor-adjoint-reference N|--factor-direction N|--factor-direction-reference N|"
          "--factor-generator N|--factor-generator-reference N|--factor-generator-check16]");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "structure tiles: FAIL: " << error.what() << '\n';
    return 1;
  }
}
