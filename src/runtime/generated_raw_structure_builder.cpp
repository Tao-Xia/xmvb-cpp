#include "runtime/generated_raw_structure_builder.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vb/vb_model_flags.hpp"

namespace xmvb::vb {
namespace {

std::string to_ascii_upper(std::string value) {
  std::transform(
      value.begin(),
      value.end(),
      value.begin(),
      [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

int parse_integer_token(const std::string& token) {
  std::size_t consumed_characters = 0;
  const int value = std::stoi(token, &consumed_characters);
  if (consumed_characters != token.size()) {
    throw std::runtime_error(
        "invalid integer token in structure class specification: " + token);
  }
  return value;
}

void append_expanded_integer_token(
    const std::string& token,
    std::vector<int>* expanded_values) {
  if (expanded_values == nullptr) {
    throw std::invalid_argument("expanded_values must not be null");
  }

  const std::size_t colon_position = token.find(':');
  const std::size_t hyphen_position = token.find('-');
  const std::size_t star_position = token.find('*');
  if (colon_position != std::string::npos) {
    const int first_value = parse_integer_token(token.substr(0, colon_position));
    const int last_value = parse_integer_token(token.substr(colon_position + 1));
    if (last_value < first_value) {
      throw std::runtime_error("descending ':' range is not supported in structure class");
    }
    for (int value = first_value; value <= last_value; ++value) {
      expanded_values->push_back(value);
    }
    return;
  }
  if (hyphen_position != std::string::npos) {
    const int first_value = parse_integer_token(token.substr(0, hyphen_position));
    const int last_value = parse_integer_token(token.substr(hyphen_position + 1));
    if (last_value < first_value) {
      throw std::runtime_error("descending '-' range is not supported in structure class");
    }
    for (int value = first_value; value <= last_value; ++value) {
      expanded_values->push_back(value);
    }
    return;
  }
  if (star_position != std::string::npos) {
    const int repeated_value = parse_integer_token(token.substr(0, star_position));
    const int repeat_count = parse_integer_token(token.substr(star_position + 1));
    if (repeat_count < 0) {
      throw std::runtime_error("negative repetition count is not supported in structure class");
    }
    for (int repeat_index = 0; repeat_index < repeat_count; ++repeat_index) {
      expanded_values->push_back(repeated_value);
    }
    return;
  }
  expanded_values->push_back(parse_integer_token(token));
}

std::vector<int> parse_expanded_integer_list(std::string text) {
  for (char& character : text) {
    if (character == ',') {
      character = ' ';
    }
  }

  std::vector<int> expanded_values;
  std::string token;
  for (char character : text) {
    if (std::isspace(static_cast<unsigned char>(character)) != 0) {
      if (!token.empty()) {
        append_expanded_integer_token(token, &expanded_values);
        token.clear();
      }
      continue;
    }
    token.push_back(character);
  }
  if (!token.empty()) {
    append_expanded_integer_token(token, &expanded_values);
  }
  return expanded_values;
}

std::size_t checked_add(
    std::size_t left,
    std::size_t right,
    const char* label) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::runtime_error(std::string(label) + " exceeds size_t range");
  }
  return left + right;
}

std::size_t checked_mul(
    std::size_t left,
    std::size_t right,
    const char* label) {
  if (left == 0 || right == 0) {
    return 0;
  }
  if (left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::runtime_error(std::string(label) + " exceeds size_t range");
  }
  return left * right;
}

std::size_t combination_count(int m, int n) {
  if (m < 0 || n < 0 || m > n) {
    throw std::runtime_error("invalid combination arguments in structure generator");
  }
  const int reduced_m = std::min(m, n - m);
  if (reduced_m == 0) {
    return 1;
  }

  std::size_t value = static_cast<std::size_t>(n);
  for (int index = 1; index <= reduced_m - 1; ++index) {
    value = checked_mul(
        value,
        static_cast<std::size_t>(n - index),
        "combination_count");
    value /= static_cast<std::size_t>(index);
  }
  value /= static_cast<std::size_t>(reduced_m);
  return value;
}

std::size_t spin_tableau_count(int n, int spin) {
  if (n < 0 || spin <= 0 || spin > n + 1) {
    return 0;
  }
  if (n == 0) {
    return 1;
  }

  std::size_t value = static_cast<std::size_t>(spin);
  const int offset = (n - spin + 1) / 2;
  const int off2 = (n + spin + 1) / 2;
  const int nloop = (n + spin - 1) / 2;
  for (int index = 1; index <= nloop; ++index) {
    value = checked_mul(
        value,
        static_cast<std::size_t>(index + offset),
        "spin_tableau_count");
    if (index <= off2) {
      value /= static_cast<std::size_t>(index);
    }
  }
  if (off2 > nloop) {
    for (int index = nloop + 1; index <= off2; ++index) {
      value /= static_cast<std::size_t>(index);
    }
  }
  return value;
}

std::size_t structure_count_for_class(
    int ionic_class,
    int n_active_doubly_occupied_orbitals,
    int n_skipped_orbitals,
    int n_active_orbitals,
    int n_active_electrons,
    int spin_multiplicity) {
  const int tableau_orbitals =
      n_active_orbitals - 2 * ionic_class - n_skipped_orbitals +
      n_active_doubly_occupied_orbitals;
  if (tableau_orbitals < 0) {
    return 0;
  }

  std::size_t count = combination_count(ionic_class, n_active_orbitals);
  count = checked_mul(
      count,
      combination_count(
          n_skipped_orbitals + ionic_class - n_active_doubly_occupied_orbitals,
          n_active_orbitals - ionic_class),
      "structure_count_for_class");
  count = checked_mul(
      count,
      spin_tableau_count(tableau_orbitals, spin_multiplicity),
      "structure_count_for_class");
  return count;
}

bool is_single_reference_active_structure_case(
    const InputDeckMetadata& metadata) {
  const int n_active_electrons = metadata.declared_active_electrons;
  const int n_active_orbitals = metadata.declared_active_orbitals;
  const int spin_multiplicity = metadata.declared_spin_multiplicity;
  const int n_active_beta_electrons =
      (n_active_electrons - spin_multiplicity + 1) / 2;
  return (n_active_beta_electrons == 0 &&
          n_active_electrons == spin_multiplicity - 1) ||
      (n_active_beta_electrons == n_active_orbitals &&
       spin_multiplicity == 1);
}

std::vector<int> parse_structure_classes(
    const InputDeckMetadata& metadata) {
  const int n_active_orbitals = metadata.declared_active_orbitals;
  const int n_active_electrons = metadata.declared_active_electrons;
  const int spin_multiplicity = metadata.declared_spin_multiplicity;
  const int n_skipped_orbitals =
      std::max(0, n_active_orbitals - n_active_electrons);
  const int n_active_doubly_occupied_orbitals =
      std::max(0, n_active_electrons - n_active_orbitals);
  const int reduced_beta_electrons =
      (n_active_electrons - 2 * n_active_doubly_occupied_orbitals -
       spin_multiplicity + 1) /
      2;
  if (reduced_beta_electrons < 0) {
    throw std::runtime_error("structure class generation requires non-negative reduced beta count");
  }

  // The legacy `genstr.c:get_strclass()` parses `STR=...` against the reduced
  // covalent problem after removing mandatory active doubly occupied pairs.
  // The explicit class labels are therefore shifted back by `ndb` only after
  // parsing and validation. Using the total active beta count here is wrong
  // and can create impossible ion classes such as MnF2's former `ionclass=3`.
  const std::string structure_class =
      to_ascii_upper(metadata.structure_class_keyword);
  std::vector<int> ionic_classes;
  if (structure_class.find("FULL") != std::string::npos) {
    ionic_classes.reserve(reduced_beta_electrons + 1);
    for (int ionic_order = 0; ionic_order <= reduced_beta_electrons; ++ionic_order) {
      ionic_classes.push_back(ionic_order);
    }
  } else {
    if (structure_class.find("COV") != std::string::npos) {
      ionic_classes.push_back(0);
    }
    const std::size_t ion_open = structure_class.find("ION(");
    if (ion_open != std::string::npos) {
      const std::size_t value_begin = ion_open + 4;
      const std::size_t value_end = structure_class.find(')', value_begin);
      if (value_end == std::string::npos) {
        throw std::runtime_error("structure class contains ION( without a closing ')'");
      }
      const auto explicit_ionic_classes =
          parse_expanded_integer_list(
              structure_class.substr(value_begin, value_end - value_begin));
      ionic_classes.insert(
          ionic_classes.end(),
          explicit_ionic_classes.begin(),
          explicit_ionic_classes.end());
    } else if (structure_class.find("ION") != std::string::npos) {
      for (int ionic_order = 1; ionic_order <= reduced_beta_electrons; ++ionic_order) {
        ionic_classes.push_back(ionic_order);
      }
    }
  }

  if (ionic_classes.empty()) {
    throw std::runtime_error(
        "unsupported or empty structure class specification: " +
        metadata.structure_class_keyword);
  }

  std::sort(ionic_classes.begin(), ionic_classes.end());
  ionic_classes.erase(
      std::unique(ionic_classes.begin(), ionic_classes.end()),
      ionic_classes.end());
  if (ionic_classes.front() < 0) {
    throw std::runtime_error("structure class cannot be smaller than 0");
  }
  if (ionic_classes.back() > reduced_beta_electrons) {
    throw std::runtime_error("structure class exceeds the maximum ionic order");
  }
  for (int& ionic_class : ionic_classes) {
    ionic_class += n_active_doubly_occupied_orbitals;
  }
  return ionic_classes;
}

template <typename RowVisitor>
void for_each_young_pattern(
    int n_alpha,
    int n_beta,
    int n_orbitals,
    int row_stride,
    RowVisitor&& visit_row) {
  if (n_alpha <= 0) {
    return;
  }
  if (n_orbitals <= 0 || n_orbitals < n_alpha) {
    throw std::runtime_error("invalid orbital count in structure generator");
  }
  if (row_stride < n_alpha) {
    throw std::runtime_error("young-pattern row stride is smaller than n_alpha");
  }

  std::vector<int> current_row(row_stride, 0);
  for (int orbital_index = 0; orbital_index < n_alpha; ++orbital_index) {
    current_row[orbital_index] = orbital_index + 1;
  }

  const int spin_multiplicity = n_alpha - n_beta + 1;
  std::vector<int> max_values(n_alpha, 0);
  if (spin_multiplicity == 1) {
    max_values[n_beta - 1] = n_orbitals - 1;
  } else {
    max_values[n_alpha - 1] = n_orbitals;
    for (int orbital_index = n_alpha - 1; orbital_index >= n_beta + 1; --orbital_index) {
      max_values[orbital_index - 1] =
          max_values[orbital_index] - 1;
    }
    if (n_beta > 0) {
      max_values[n_beta - 1] =
          max_values[n_beta] - 2;
    }
  }
  for (int orbital_index = n_beta - 1; orbital_index >= 1; --orbital_index) {
    max_values[orbital_index - 1] =
        max_values[orbital_index] - 2;
  }

  while (true) {
    visit_row(current_row);

    int pivot = n_alpha - 1;
    while (pivot >= 0 &&
           current_row[pivot] ==
               max_values[pivot]) {
      --pivot;
    }
    if (pivot < 0) {
      break;
    }

    ++current_row[pivot];
    for (int suffix_index = pivot + 1; suffix_index < n_alpha; ++suffix_index) {
      current_row[suffix_index] =
          current_row[suffix_index - 1] + 1;
    }
  }
}

bool beta_candidate_is_compatible(
    const std::vector<int>& alpha_row,
    const std::vector<int>& beta_candidate,
    int n_alpha,
    int n_beta) {
  for (int beta_index = 0; beta_index < n_beta; ++beta_index) {
    const int beta_orbital = beta_candidate[beta_index];
    if (beta_orbital < alpha_row[beta_index]) {
      return false;
    }
    for (int alpha_index = beta_index; alpha_index < n_alpha; ++alpha_index) {
      if (beta_orbital == alpha_row[alpha_index]) {
        return false;
      }
    }
  }
  return true;
}

void append_open_shell_padding(
    std::vector<int>* beta_row,
    int n_alpha,
    int n_beta,
    int n_orbitals) {
  if (beta_row == nullptr) {
    throw std::invalid_argument("beta_row must not be null");
  }
  if (n_alpha <= n_beta) {
    return;
  }
  for (int orbital_index = n_beta; orbital_index < n_alpha; ++orbital_index) {
    (*beta_row)[orbital_index] = n_orbitals + n_alpha - orbital_index;
  }
}

template <typename PairVisitor>
void for_each_covalent_pattern_pair(
    int n_alpha,
    int n_beta,
    int n_orbitals,
    int row_stride,
    PairVisitor&& visit_pair) {
  if (n_alpha <= 0) {
    return;
  }
  if (row_stride < n_alpha) {
    throw std::runtime_error("covalent row stride is smaller than n_alpha");
  }

  if (n_beta <= 0) {
    for_each_young_pattern(
        n_alpha,
        0,
        n_orbitals,
        row_stride,
        [&](const std::vector<int>& alpha_row) {
          visit_pair(alpha_row, std::vector<int>(row_stride, 0));
        });
    return;
  }

  if (n_alpha + n_beta == n_orbitals) {
    for_each_young_pattern(
        n_alpha,
        n_beta,
        n_orbitals,
        row_stride,
        [&](const std::vector<int>& alpha_row) {
          std::vector<int> beta_row(row_stride, 0);
          int next_beta_index = 0;
          for (int orbital_label = 1; orbital_label <= n_orbitals; ++orbital_label) {
            bool needed = true;
            for (int alpha_index = 0; alpha_index < n_alpha; ++alpha_index) {
              if (alpha_row[alpha_index] == orbital_label) {
                needed = false;
                break;
              }
            }
            if (needed) {
              beta_row[next_beta_index] = orbital_label;
              ++next_beta_index;
            }
          }
          append_open_shell_padding(&beta_row, n_alpha, n_beta, n_orbitals);
          visit_pair(alpha_row, std::move(beta_row));
        });
    return;
  }

  for_each_young_pattern(
      n_alpha,
      n_beta,
      n_orbitals,
      row_stride,
      [&](const std::vector<int>& alpha_row) {
        for_each_young_pattern(
            n_beta,
            0,
            n_orbitals,
            n_beta,
            [&](const std::vector<int>& beta_candidate) {
              if (!beta_candidate_is_compatible(
                      alpha_row,
                      beta_candidate,
                      n_alpha,
                      n_beta)) {
                return;
              }
              std::vector<int> beta_row(row_stride, 0);
              for (int beta_index = 0; beta_index < n_beta; ++beta_index) {
                beta_row[beta_index] =
                    beta_candidate[beta_index];
              }
              append_open_shell_padding(&beta_row, n_alpha, n_beta, n_orbitals);
              visit_pair(alpha_row, std::move(beta_row));
            });
      });
}

void reorder_young_bonds(
    std::vector<int>* alpha_row,
    const std::vector<int>& beta_row,
    int n_vector_orbitals,
    int n_row_orbitals,
    int n_orbitals) {
  if (alpha_row == nullptr) {
    throw std::invalid_argument("alpha_row must not be null");
  }

  std::vector<int> left_index(3 * n_orbitals, 0);
  std::vector<int> right_index(3 * n_orbitals, 0);
  std::vector<int> reordered_beta(
      beta_row.begin(),
      beta_row.begin() + n_row_orbitals);
  int remaining_count = n_vector_orbitals;

  for (int vector_index = 0; vector_index < n_vector_orbitals; ++vector_index) {
    right_index[reordered_beta[vector_index] - 1] =
        vector_index + 1;
  }

  for (int vector_index = n_vector_orbitals; vector_index >= 1; --vector_index) {
    const int left_max =
        (*alpha_row)[remaining_count - 1];
    int match_index = 0;
    for (; match_index < remaining_count; ++match_index) {
      if (left_max <= reordered_beta[match_index]) {
        left_index[left_max - 1] =
            right_index[reordered_beta[match_index] - 1];
        break;
      }
    }
    for (int shift_index = match_index; shift_index < remaining_count - 1; ++shift_index) {
      reordered_beta[shift_index] =
          reordered_beta[shift_index + 1];
    }
    --remaining_count;
  }

  for (int vector_index = 0; vector_index < n_vector_orbitals; ++vector_index) {
    reordered_beta[
        left_index[(*alpha_row)[vector_index] - 1] - 1] =
        (*alpha_row)[vector_index];
  }
  for (int vector_index = 0; vector_index < n_vector_orbitals; ++vector_index) {
    (*alpha_row)[vector_index] =
        reordered_beta[vector_index];
  }
}

std::vector<int> build_remaining_orbitals(
    const std::vector<int>& ion_row,
    int ionic_class,
    int n_active_orbitals) {
  std::vector<int> used_orbitals(n_active_orbitals + 1, 0);
  for (int ion_index = 0; ion_index < ionic_class; ++ion_index) {
    used_orbitals[ion_row[ion_index]] = 1;
  }

  std::vector<int> remaining_orbitals;
  remaining_orbitals.reserve(n_active_orbitals - ionic_class);
  for (int orbital_label = 1; orbital_label <= n_active_orbitals; ++orbital_label) {
    if (used_orbitals[orbital_label] == 0) {
      remaining_orbitals.push_back(orbital_label);
    }
  }
  return remaining_orbitals;
}

void write_single_reference_structure(
    int n_inactive_doubly_occupied_orbitals,
    const InputDeckMetadata& metadata,
    RawStructureData* raw_structure_data) {
  if (raw_structure_data == nullptr) {
    throw std::invalid_argument("raw_structure_data must not be null");
  }

  const int n_active_electrons = metadata.declared_active_electrons;
  const int spin_multiplicity = metadata.declared_spin_multiplicity;
  const int n_active_beta_electrons =
      (n_active_electrons - spin_multiplicity + 1) / 2;
  int* structure_begin = raw_structure_data->raw_structure_orbitals.data();

  const int total_paired_orbitals =
      n_inactive_doubly_occupied_orbitals + n_active_beta_electrons;
  for (int orbital_index = 0; orbital_index < total_paired_orbitals; ++orbital_index) {
    structure_begin[2 * orbital_index] = orbital_index + 1;
    structure_begin[2 * orbital_index + 1] = orbital_index + 1;
  }
  for (int open_shell_index = 0;
       open_shell_index < spin_multiplicity - 1;
       ++open_shell_index) {
    structure_begin[2 * total_paired_orbitals + open_shell_index] =
        total_paired_orbitals + open_shell_index + 1;
  }
}

void write_structure_to_output(
    int n_inactive_doubly_occupied_orbitals,
    int ionic_class,
    const std::vector<int>& ion_row,
    const std::vector<int>& remaining_orbitals,
    const std::vector<int>& alpha_row,
    const std::vector<int>& beta_row,
    int n_covalent_alpha,
    int n_covalent_beta,
    int* structure_begin) {
  if (structure_begin == nullptr) {
    throw std::invalid_argument("structure_begin must not be null");
  }

  for (int orbital_index = 0;
       orbital_index < n_inactive_doubly_occupied_orbitals;
       ++orbital_index) {
    structure_begin[2 * orbital_index] = orbital_index + 1;
    structure_begin[2 * orbital_index + 1] = orbital_index + 1;
  }

  const int active_offset = 2 * n_inactive_doubly_occupied_orbitals;
  const int orbital_shift = n_inactive_doubly_occupied_orbitals;
  if (ionic_class > 0) {
    for (int ion_index = 0; ion_index < ionic_class; ++ion_index) {
      structure_begin[active_offset + 2 * ion_index] =
          ion_row[ion_index] + orbital_shift;
      structure_begin[active_offset + 2 * ion_index + 1] =
          ion_row[ion_index] + orbital_shift;
    }
    for (int beta_index = 0; beta_index < n_covalent_beta; ++beta_index) {
      structure_begin[active_offset + 2 * ionic_class + 2 * beta_index] =
          remaining_orbitals[alpha_row[beta_index] - 1] +
          orbital_shift;
      structure_begin[active_offset + 2 * ionic_class + 2 * beta_index + 1] =
          remaining_orbitals[beta_row[beta_index] - 1] +
          orbital_shift;
    }
    for (int alpha_index = n_covalent_beta; alpha_index < n_covalent_alpha; ++alpha_index) {
      structure_begin[active_offset + 2 * ionic_class + n_covalent_beta + alpha_index] =
          remaining_orbitals[alpha_row[alpha_index] - 1] +
          orbital_shift;
    }
    return;
  }

  for (int beta_index = 0; beta_index < n_covalent_beta; ++beta_index) {
    structure_begin[active_offset + 2 * beta_index] =
        alpha_row[beta_index] + orbital_shift;
    structure_begin[active_offset + 2 * beta_index + 1] =
        beta_row[beta_index] + orbital_shift;
  }
  for (int alpha_index = n_covalent_beta; alpha_index < n_covalent_alpha; ++alpha_index) {
    structure_begin[active_offset + n_covalent_beta + alpha_index] =
        alpha_row[alpha_index] + orbital_shift;
  }
}

void append_structure_class_to_output(
    int ionic_class,
    const InputDeckMetadata& metadata,
    int n_inactive_doubly_occupied_orbitals,
    RawStructureData* raw_structure_data,
    std::size_t* next_structure_index) {
  if (raw_structure_data == nullptr || next_structure_index == nullptr) {
    throw std::invalid_argument("structure append state must not be null");
  }

  const int n_active_orbitals = metadata.declared_active_orbitals;
  const int n_active_electrons = metadata.declared_active_electrons;
  const int spin_multiplicity = metadata.declared_spin_multiplicity;
  const int n_active_beta_electrons =
      (n_active_electrons - spin_multiplicity + 1) / 2;
  const int n_active_alpha_electrons =
      n_active_electrons - n_active_beta_electrons;
  if (ionic_class < 0 || ionic_class > n_active_beta_electrons) {
    throw std::runtime_error("structure class exceeds the active beta electron count");
  }

  const int n_covalent_alpha = n_active_alpha_electrons - ionic_class;
  const int n_covalent_beta = n_active_beta_electrons - ionic_class;
  if (n_covalent_alpha < 0 || n_covalent_beta < 0) {
    throw std::runtime_error("structure class produces negative covalent dimensions");
  }

  const auto append_structure =
      [&](const std::vector<int>& ion_row,
          const std::vector<int>& remaining_orbitals,
          const std::vector<int>& alpha_row,
          const std::vector<int>& beta_row) {
        if (*next_structure_index >= raw_structure_data->n_structures) {
          throw std::runtime_error("raw structure generation exceeded the precomputed count");
        }
        int* structure_begin =
            raw_structure_data->raw_structure_orbitals.data() +
            (*next_structure_index) * raw_structure_data->n_total_electrons;
        write_structure_to_output(
            n_inactive_doubly_occupied_orbitals,
            ionic_class,
            ion_row,
            remaining_orbitals,
            alpha_row,
            beta_row,
            n_covalent_alpha,
            n_covalent_beta,
            structure_begin);
        ++(*next_structure_index);
      };

  if (ionic_class > 0) {
    for_each_young_pattern(
        ionic_class,
        0,
        n_active_orbitals,
        ionic_class,
        [&](const std::vector<int>& ion_row) {
          const std::vector<int> remaining_orbitals =
              build_remaining_orbitals(
                  ion_row,
                  ionic_class,
                  n_active_orbitals);
          if (n_covalent_alpha <= 0) {
            append_structure(
                ion_row,
                remaining_orbitals,
                std::vector<int>(),
                std::vector<int>());
            return;
          }
          for_each_covalent_pattern_pair(
              n_covalent_alpha,
              n_covalent_beta,
              n_active_orbitals - ionic_class,
              n_active_alpha_electrons,
              [&](const std::vector<int>& alpha_row,
                  const std::vector<int>& beta_row) {
                std::vector<int> reordered_alpha = alpha_row;
                if (n_covalent_beta > 0) {
                  reorder_young_bonds(
                      &reordered_alpha,
                      beta_row,
                      n_covalent_alpha,
                      n_active_alpha_electrons,
                      n_active_orbitals);
                }
                append_structure(
                    ion_row,
                    remaining_orbitals,
                    reordered_alpha,
                    beta_row);
              });
        });
    return;
  }

  for_each_covalent_pattern_pair(
      n_active_alpha_electrons,
      n_active_beta_electrons,
      n_active_orbitals,
      n_active_alpha_electrons,
      [&](const std::vector<int>& alpha_row,
          const std::vector<int>& beta_row) {
        std::vector<int> reordered_alpha = alpha_row;
        if (n_active_beta_electrons > 0) {
          reorder_young_bonds(
              &reordered_alpha,
              beta_row,
              n_active_alpha_electrons,
              n_active_alpha_electrons,
              n_active_orbitals);
        }
        append_structure(
            std::vector<int>(),
            std::vector<int>(),
            reordered_alpha,
            beta_row);
      });
}

std::size_t count_active_raw_structures(
    const InputDeckMetadata& metadata) {
  if (metadata.wavefunction_type !=
      wavefunction_type_code(WavefunctionType::Structure)) {
    throw std::runtime_error(
        "pure C++ structure-class generation currently supports only WFNTYP=STR");
  }

  const int n_active_orbitals = metadata.declared_active_orbitals;
  const int n_active_electrons = metadata.declared_active_electrons;
  const int spin_multiplicity = metadata.declared_spin_multiplicity;
  const int n_active_beta_electrons =
      (n_active_electrons - spin_multiplicity + 1) / 2;
  if (n_active_beta_electrons < 0) {
    throw std::runtime_error(
        "active beta electron count is negative in structure generator");
  }
  if (n_active_beta_electrons > n_active_orbitals) {
    throw std::runtime_error(
        "active beta electron count exceeds the active orbital count");
  }
  if (is_single_reference_active_structure_case(metadata)) {
    return 1;
  }

  const int n_skipped_orbitals =
      std::max(0, n_active_orbitals - n_active_electrons);
  const int n_active_doubly_occupied_orbitals =
      std::max(0, n_active_electrons - n_active_orbitals);
  const auto ionic_classes = parse_structure_classes(metadata);

  // The active raw-structure buffer is one unavoidable O(N_structures *
  // n_total_electrons) allocation. Count the legacy classes analytically first
  // so the generator can write directly into that final flat buffer without
  // materializing intermediate `vector<vector<int>>` copies.
  std::size_t structure_count = 0;
  for (const int ionic_class : ionic_classes) {
    structure_count = checked_add(
        structure_count,
        structure_count_for_class(
            ionic_class,
            n_active_doubly_occupied_orbitals,
            n_skipped_orbitals,
            n_active_orbitals,
            n_active_electrons,
            spin_multiplicity),
        "count_active_raw_structures");
  }
  if (structure_count == 0) {
    throw std::runtime_error("pure C++ structure generator produced no raw structures");
  }
  return structure_count;
}

}  // namespace

bool can_build_generated_raw_structures_from_structure_class(
    const InputDeckMetadata& metadata,
    int n_total_electrons) noexcept {
  return !metadata.structure_class_keyword.empty() &&
      metadata.wavefunction_type ==
          wavefunction_type_code(WavefunctionType::Structure) &&
      metadata.declared_active_orbitals > 0 &&
      metadata.declared_active_electrons > 0 &&
      metadata.declared_spin_multiplicity > 0 &&
      n_total_electrons >= metadata.declared_active_electrons &&
      ((n_total_electrons - metadata.declared_active_electrons) % 2 == 0);
}

RawStructureData build_generated_raw_structures_from_structure_class(
    const InputDeckMetadata& metadata,
    int n_total_electrons) {
  if (!can_build_generated_raw_structures_from_structure_class(
          metadata,
          n_total_electrons)) {
    throw std::runtime_error(
        "input deck does not define enough information for C++ structure generation");
  }

  const int n_inactive_doubly_occupied_orbitals =
      (n_total_electrons - metadata.declared_active_electrons) / 2;
  const std::size_t n_structures =
      count_active_raw_structures(metadata);

  RawStructureData raw_structure_data;
  raw_structure_data.n_structures = n_structures;
  raw_structure_data.n_total_electrons = static_cast<std::size_t>(n_total_electrons);
  raw_structure_data.n_active_electrons =
      static_cast<std::size_t>(metadata.declared_active_electrons);
  raw_structure_data.spin_multiplicity = metadata.declared_spin_multiplicity;
  raw_structure_data.wavefunction_type = metadata.wavefunction_type;
  raw_structure_data.vb_function_type = metadata.vb_function_type;
  raw_structure_data.raw_structure_orbitals.assign(
      checked_mul(
          n_structures,
          static_cast<std::size_t>(n_total_electrons),
          "raw_structure_orbitals"),
      0);

  if (is_single_reference_active_structure_case(metadata)) {
    write_single_reference_structure(
        n_inactive_doubly_occupied_orbitals,
        metadata,
        &raw_structure_data);
    return raw_structure_data;
  }

  std::size_t next_structure_index = 0;
  const auto ionic_classes = parse_structure_classes(metadata);
  for (const int ionic_class : ionic_classes) {
    append_structure_class_to_output(
        ionic_class,
        metadata,
        n_inactive_doubly_occupied_orbitals,
        &raw_structure_data,
        &next_structure_index);
  }
  if (next_structure_index != raw_structure_data.n_structures) {
    throw std::runtime_error("raw structure generation count mismatch");
  }

  return raw_structure_data;
}

}  // namespace xmvb::vb
