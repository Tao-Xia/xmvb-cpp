#include "vbscf/structures/expansion/expander.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vbscf/core/configuration/flags.hpp"
#include "vbscf/structures/selection/subspace/selector.hpp"

namespace xmvb::vb {

namespace {

struct MaskHash {
  std::size_t operator()(const std::vector<std::uint64_t>& words) const {
    std::size_t hash_value = 0;
    for (const auto word : words) {
      hash_value = hash_value * 1315423911u +
                   static_cast<std::size_t>(word ^ (word >> 32));
    }
    return hash_value;
  }
};

// The bitset counts earlier occupied orbitals above each newly appended
// orbital, which is the permutation parity of the old swap-sort sequence.
int encode_spin_string(const std::vector<int>& orbitals, bool fast,
                       std::uint64_t& mask,
                       std::vector<std::uint64_t>& words) {
  mask = 0;
  if (!fast) std::fill(words.begin(), words.end(), 0);
  int sign = 1;
  for (const int orbital : orbitals) {
    if (orbital < 0) {
      throw std::invalid_argument("active orbital index must be nonnegative");
    }
    const std::size_t word = static_cast<std::size_t>(orbital) / 64;
    const unsigned bit = static_cast<unsigned>(orbital) % 64;
    const std::uint64_t flag = std::uint64_t{1} << bit;
    if (fast) {
      if (mask & flag) {
        throw std::invalid_argument("duplicate orbital in an occupied spin string");
      }
      const auto higher = bit == 63 ? std::uint64_t{0} : mask >> (bit + 1);
      if (std::popcount(higher) & 1) sign = -sign;
      mask |= flag;
    } else {
      if (words[word] & flag) {
        throw std::invalid_argument("duplicate orbital in an occupied spin string");
      }
      unsigned higher_count = bit == 63 ? 0 :
          static_cast<unsigned>(std::popcount(words[word] >> (bit + 1)));
      for (std::size_t later = word + 1; later < words.size(); ++later) {
        higher_count += static_cast<unsigned>(std::popcount(words[later]));
      }
      if (higher_count & 1) sign = -sign;
      words[word] |= flag;
    }
  }
  return sign;
}

class SpinInterner {
public:
  explicit SpinInterner(std::vector<std::vector<int>>& strings)
      : strings_(strings) {}

  int intern_fast(std::uint64_t mask) {
    const auto found = fast_ids_.find(mask);
    if (found != fast_ids_.end()) return found->second;
    const int id = next_id();
    fast_ids_.emplace(mask, id);
    std::vector<int> decoded;
    while (mask) {
      const int orbital = std::countr_zero(mask);
      decoded.push_back(orbital);
      mask &= mask - 1;
    }
    strings_.push_back(std::move(decoded));
    return id;
  }

  int intern_general(const std::vector<std::uint64_t>& words) {
    const auto found = general_ids_.find(words);
    if (found != general_ids_.end()) return found->second;
    const int id = next_id();
    general_ids_.emplace(words, id);
    std::vector<int> decoded;
    for (std::size_t word = 0; word < words.size(); ++word) {
      auto bits = words[word];
      while (bits) {
        decoded.push_back(static_cast<int>(64 * word + std::countr_zero(bits)));
        bits &= bits - 1;
      }
    }
    strings_.push_back(std::move(decoded));
    return id;
  }

  int lookup_fast(std::uint64_t mask) const { return fast_ids_.at(mask); }
  int lookup_general(const std::vector<std::uint64_t>& words) const {
    return general_ids_.at(words);
  }

private:
  int next_id() const {
    if (strings_.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("too many unique spin strings");
    }
    return static_cast<int>(strings_.size());
  }

  std::vector<std::vector<int>>& strings_;
  std::unordered_map<std::uint64_t, int> fast_ids_;
  std::unordered_map<std::vector<std::uint64_t>, int, MaskHash> general_ids_;
};

std::uint64_t pair_key(int alpha_id, int beta_id) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(alpha_id)) << 32) |
         static_cast<std::uint32_t>(beta_id);
}

template <typename Emit>
void enumerate_structure(const RawStructureData& raw, int structure_index,
                         int n_inactive, int n_active_beta,
                         std::vector<int>& active, std::vector<int>& alpha,
                         std::vector<int>& beta, std::vector<int>& variable_pairs,
                         std::vector<unsigned char>& choices, Emit&& emit) {
  const int* structure = raw.structure_orbitals_data(structure_index);
  for (int index = 0; index < raw.n_active_electrons; ++index) {
    active[index] = structure[index + 2 * n_inactive] - n_inactive - 1;
  }

  const bool single_product = raw.wavefunction_type ==
      wavefunction_type_code(WavefunctionType::Determinant);
  variable_pairs.clear();
  choices.clear();
  if (!single_product) {
    for (int pair = 0; pair < n_active_beta; ++pair) {
      if (active[2 * pair] != active[2 * pair + 1]) {
        variable_pairs.push_back(pair);
        choices.push_back(0);
      }
    }
  }

  while (true) {
    alpha.clear();
    beta.clear();
    if (single_product) {
      for (int pair = 0; pair < n_active_beta; ++pair) {
        alpha.push_back(active[pair]);
        beta.push_back(active[pair + n_active_beta]);
      }
    } else {
      std::size_t variable = 0;
      for (int pair = 0; pair < n_active_beta; ++pair) {
        bool swapped = false;
        if (variable < variable_pairs.size() && variable_pairs[variable] == pair) {
          swapped = choices[variable] != 0;
          ++variable;
        }
        alpha.push_back(active[2 * pair + static_cast<int>(swapped)]);
        beta.push_back(active[2 * pair + 1 - static_cast<int>(swapped)]);
      }
    }
    for (int index = 2 * n_active_beta; index < raw.n_active_electrons; ++index) {
      alpha.push_back(active[index]);
    }

    emit(alpha, beta);

    std::size_t bit = 0;
    while (bit < choices.size() && choices[bit]) {
      choices[bit++] = 0;
    }
    if (bit == choices.size()) break;
    choices[bit] = 1;
  }
}

struct PairFlip {
  std::uint64_t first;
  std::uint64_t second;
  bool changes_sign;
};

template <typename Emit>
void enumerate_structure_fast(const RawStructureData& raw, int structure_index,
                              int n_inactive, int n_active_beta,
                              std::vector<int>& active, std::vector<int>& alpha,
                              std::vector<int>& beta,
                              std::vector<PairFlip>& flips,
                              std::vector<unsigned char>& choices, Emit&& emit) {
  const int* structure = raw.structure_orbitals_data(structure_index);
  for (int index = 0; index < raw.n_active_electrons; ++index) {
    active[index] = structure[index + 2 * n_inactive] - n_inactive - 1;
  }

  alpha.clear();
  beta.clear();
  flips.clear();
  choices.clear();
  const bool single_product = raw.wavefunction_type ==
      wavefunction_type_code(WavefunctionType::Determinant);
  if (single_product) {
    for (int pair = 0; pair < n_active_beta; ++pair) {
      alpha.push_back(active[pair]);
      beta.push_back(active[pair + n_active_beta]);
    }
  } else {
    for (int pair = 0; pair < n_active_beta; ++pair) {
      alpha.push_back(active[2 * pair]);
      beta.push_back(active[2 * pair + 1]);
    }
  }
  for (int index = 2 * n_active_beta; index < raw.n_active_electrons; ++index) {
    alpha.push_back(active[index]);
  }
  std::uint64_t alpha_mask = 0, beta_mask = 0;
  std::vector<std::uint64_t> unused_words;
  const int alpha_sign = encode_spin_string(alpha, true, alpha_mask, unused_words);
  const int beta_sign = encode_spin_string(beta, true, beta_mask, unused_words);
  int sign = alpha_sign * beta_sign;

  if (!single_product) {
    for (int pair = 0; pair < n_active_beta; ++pair) {
      const int first = active[2 * pair];
      const int second = active[2 * pair + 1];
      if (first == second) continue;
      const int lower = std::min(first, second);
      const int upper = std::max(first, second);
      const std::uint64_t below_upper = (std::uint64_t{1} << upper) - 1;
      const std::uint64_t through_lower =
          (std::uint64_t{1} << (lower + 1)) - 1;
      const bool changes_sign =
          (std::popcount((alpha_mask ^ beta_mask) &
                         (below_upper ^ through_lower)) & 1) != 0;
      flips.push_back({std::uint64_t{1} << first,
                       std::uint64_t{1} << second,
                       changes_sign});
      choices.push_back(0);
    }
  }

  auto toggle = [&](std::size_t bit) {
    const auto& flip = flips[bit];
    const std::uint64_t from = choices[bit] ? flip.second : flip.first;
    const std::uint64_t to = choices[bit] ? flip.first : flip.second;
    if ((alpha_mask & to) || (beta_mask & from)) {
      throw std::invalid_argument("duplicate orbital in an occupied spin string");
    }
    if (flip.changes_sign) sign = -sign;
    const std::uint64_t exchange = flip.first | flip.second;
    alpha_mask ^= exchange;
    beta_mask ^= exchange;
    choices[bit] ^= 1;
  };

  while (true) {
    emit(alpha_mask, beta_mask, sign);
    std::size_t bit = 0;
    while (bit < choices.size() && choices[bit]) toggle(bit++);
    if (bit == choices.size()) break;
    toggle(bit);
  }
}

}  // namespace

StructureExpansion StructureExpander::expand(const RawStructureData& raw) const {
  if (raw.n_total_electrons <= 0 || raw.n_active_electrons <= 0 ||
      raw.n_active_electrons > raw.n_total_electrons) {
    throw std::invalid_argument("raw structure data must define valid positive electron counts");
  }
  if (raw.n_structures <= 0 ||
      raw.raw_structure_orbitals.size() != raw.flat_orbital_count()) {
    throw std::invalid_argument("raw structure count does not match stored structures");
  }
  const int n_inactive = (raw.n_total_electrons - raw.n_active_electrons) / 2;
  const int n_open = raw.spin_multiplicity - 1;
  const int n_paired = raw.n_active_electrons - n_open;
  if ((raw.n_total_electrons - raw.n_active_electrons) % 2 != 0 ||
      n_open < 0 || n_paired < 0 || n_paired % 2 != 0) {
    throw std::invalid_argument("raw structure electron and spin counts are inconsistent");
  }
  const int n_active_beta = n_paired / 2;
  if (raw.vb_function_type != vb_function_type_code(VbFunctionType::Determinant) &&
      raw.wavefunction_type != wavefunction_type_code(WavefunctionType::Determinant)) {
    throw std::invalid_argument(
        "structure expansion supports determinant VB or determinant wavefunctions only");
  }

  StructureExpansion result;
  result.n_structures = raw.n_structures;
  int max_orbital = -1;
  for (int structure = 0; structure < raw.n_structures; ++structure) {
    const int* orbitals = raw.structure_orbitals_data(structure);
    for (int index = 2 * n_inactive; index < raw.n_total_electrons; ++index) {
      max_orbital = std::max(max_orbital, orbitals[index] - n_inactive - 1);
    }
  }
  const bool fast_mask = max_orbital < 64;
  const bool packed_pairs = max_orbital < 32;
  const unsigned packed_width = static_cast<unsigned>(max_orbital + 1);
  const std::size_t word_count = fast_mask ? 0 :
      static_cast<std::size_t>(max_orbital) / 64 + 1;
  std::vector<std::uint64_t> alpha_words(word_count), beta_words(word_count);
  std::uint64_t alpha_mask = 0, beta_mask = 0;
  SpinInterner alpha_ids(result.alpha_strings), beta_ids(result.beta_strings);
  std::unordered_map<std::uint64_t, int> pair_ids;
  std::vector<std::size_t> counts;
  std::vector<int> active(raw.n_active_electrons);
  std::vector<int> alpha, beta, variable_pairs;
  std::vector<PairFlip> flips;
  std::vector<unsigned char> choices;
  alpha.reserve(raw.n_active_electrons);
  beta.reserve(raw.n_active_electrons);
  variable_pairs.reserve(n_active_beta);
  flips.reserve(n_active_beta);
  choices.reserve(n_active_beta);

  auto record_pair = [&](std::uint64_t key, auto&& identify) {
    const auto [found, inserted] =
        pair_ids.try_emplace(key, static_cast<int>(result.pair_alpha.size()));
    if (inserted) {
      if (result.pair_alpha.size() >=
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("too many unique spin-string pairs");
      }
      const auto [alpha_id, beta_id] = identify();
      result.pair_alpha.push_back(alpha_id);
      result.pair_beta.push_back(beta_id);
      counts.push_back(1);
    } else {
      ++counts[found->second];
    }
  };

  for (int structure = 0; structure < raw.n_structures; ++structure) {
    if (packed_pairs) {
      enumerate_structure_fast(raw, structure, n_inactive, n_active_beta,
                               active, alpha, beta, flips, choices,
                               [&](std::uint64_t a, std::uint64_t b, int) {
        record_pair(a | (b << packed_width), [&] {
          return std::pair{alpha_ids.intern_fast(a), beta_ids.intern_fast(b)};
        });
      });
    } else if (fast_mask) {
      enumerate_structure_fast(raw, structure, n_inactive, n_active_beta,
                               active, alpha, beta, flips, choices,
                               [&](std::uint64_t a, std::uint64_t b, int) {
        const int alpha_id = alpha_ids.intern_fast(a);
        const int beta_id = beta_ids.intern_fast(b);
        record_pair(pair_key(alpha_id, beta_id), [&] {
          return std::pair{alpha_id, beta_id};
        });
      });
    } else {
      enumerate_structure(raw, structure, n_inactive, n_active_beta, active,
                          alpha, beta, variable_pairs, choices,
                          [&](const std::vector<int>& a, const std::vector<int>& b) {
        encode_spin_string(a, false, alpha_mask, alpha_words);
        encode_spin_string(b, false, beta_mask, beta_words);
        const int alpha_id = alpha_ids.intern_general(alpha_words);
        const int beta_id = beta_ids.intern_general(beta_words);
        record_pair(pair_key(alpha_id, beta_id), [&] {
          return std::pair{alpha_id, beta_id};
        });
      });
    }
  }

  result.offsets.reserve(counts.size() + 1);
  result.offsets.push_back(0);
  for (const std::size_t count : counts) {
    if (count > result.terms.max_size() - result.offsets.back()) {
      throw std::overflow_error("too many structure expansion terms");
    }
    result.offsets.push_back(result.offsets.back() + count);
  }
  result.terms.resize(result.offsets.back());
  // Reuse the count pass workspace as CSR write cursors.
  std::copy(result.offsets.begin(), result.offsets.end() - 1, counts.begin());

  for (int structure = 0; structure < raw.n_structures; ++structure) {
    if (packed_pairs) {
      enumerate_structure_fast(raw, structure, n_inactive, n_active_beta,
                               active, alpha, beta, flips, choices,
                               [&](std::uint64_t a, std::uint64_t b, int sign) {
        const int pair_id = pair_ids.at(a | (b << packed_width));
        result.terms[counts[pair_id]++] = {structure, static_cast<double>(sign)};
      });
    } else if (fast_mask) {
      enumerate_structure_fast(raw, structure, n_inactive, n_active_beta,
                               active, alpha, beta, flips, choices,
                               [&](std::uint64_t a, std::uint64_t b, int sign) {
        const int alpha_id = alpha_ids.lookup_fast(a);
        const int beta_id = beta_ids.lookup_fast(b);
        const int pair_id = pair_ids.at(pair_key(alpha_id, beta_id));
        result.terms[counts[pair_id]++] = {structure, static_cast<double>(sign)};
      });
    } else {
      enumerate_structure(raw, structure, n_inactive, n_active_beta, active,
                          alpha, beta, variable_pairs, choices,
                          [&](const std::vector<int>& a, const std::vector<int>& b) {
        const int alpha_sign = encode_spin_string(a, false, alpha_mask, alpha_words);
        const int beta_sign = encode_spin_string(b, false, beta_mask, beta_words);
        const int alpha_id = alpha_ids.lookup_general(alpha_words);
        const int beta_id = beta_ids.lookup_general(beta_words);
        const int pair_id = pair_ids.at(pair_key(alpha_id, beta_id));
        result.terms[counts[pair_id]++] = {
            structure, static_cast<double>(alpha_sign * beta_sign)};
      });
    }
  }

  return result;
}

StructureExpansion StructureExpander::expand_subset(
    const RawStructureData& raw,
    const std::vector<int>& selected_structure_indices) const {
  return expand(build_raw_structure_subset(raw, selected_structure_indices));
}

}  // namespace xmvb::vb
