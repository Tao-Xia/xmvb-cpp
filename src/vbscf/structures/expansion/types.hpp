#pragma once

#include <cstddef>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include "mcscf/ci/string_table.hpp"

namespace xmvb::vb {

/**
 * @brief One spin-string-pair-to-structure expansion term.
 *
 * Each spin-string pair can contribute to one or more structures with a signed
 * coefficient. This structure makes that mapping explicit.
 */
struct StructureExpansionTerm {
    /**
   * @brief Zero-based structure index.
   */
    Eigen::Index structure_index = 0;

    /**
   * @brief Signed coefficient contributed by the spin-string pair to the structure.
   *
   * The current expansion uses `+1` or `-1`, but this representation
   * intentionally allows any real coefficient.
   */
    double coefficient = 0.0;
};

/** @brief Borrowed spin-string pairs; topology queries never decode occupations. */
class SpinStringRows {
  public:
    SpinStringRows(const mcscf::StringTable& strings, const std::vector<mcscf::StringIndex>& ids)
        : strings_(strings), pair_ids_(&ids) {}
    SpinStringRows(const mcscf::StringTable& strings) : strings_(strings) {}
    SpinStringRows(const std::vector<std::vector<int>>& rows) : strings_(rows) {}
    std::size_t size() const {
        return pair_ids_ ? pair_ids_->size() : strings_.size();
    }
    bool empty() const {
        return size() == 0;
    }
    const mcscf::StringTable& strings() const {
        return strings_;
    }
    std::span<const mcscf::StringIndex> indices() const {
        return pair_ids_ ? std::span<const mcscf::StringIndex>(*pair_ids_)
                         : std::span<const mcscf::StringIndex>();
    }
    std::size_t string_id(std::size_t i) const {
        return pair_ids_ ? (*pair_ids_)[i] : i;
    }
    std::uint64_t mask(std::size_t i) const {
        return strings_.mask(string_id(i));
    }
    std::span<const std::uint64_t> words(std::size_t i) const {
        return strings_.words(string_id(i));
    }
    int electron_count(std::size_t i = 0) const {
        return strings_.electron_count(string_id(i));
    }
    bool less(std::size_t a, std::size_t b) const {
        return strings_.less(string_id(a), string_id(b));
    }
    std::vector<int> operator[](std::size_t i) const {
        return strings_[string_id(i)];
    }
    std::vector<int> front() const {
        return (*this)[0];
    }
    class Iterator {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<int>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = void;
        Iterator(const SpinStringRows* rows, std::size_t i) : rows_(rows), i_(i) {}
        value_type operator*() const {
            return (*rows_)[i_];
        }
        Iterator& operator++() {
            ++i_;
            return *this;
        }
        Iterator operator++(int) {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const Iterator&) const = default;

      private:
        const SpinStringRows* rows_;
        std::size_t i_;
    };
    Iterator begin() const {
        return {this, 0};
    }
    Iterator end() const {
        return {this, size()};
    }
    friend bool operator==(const SpinStringRows& a, const SpinStringRows& b) {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!a.strings_.same_row(a.string_id(i), b.strings_, b.string_id(i)))
                return false;
        return true;
    }

  private:
    mcscf::StringTable strings_;
    const std::vector<mcscf::StringIndex>* pair_ids_ = nullptr;
};

/** Borrowed pair-to-structure rows backed by CSR or existing row storage. */
class StructureIncidence {
  public:
    StructureIncidence(const std::vector<std::size_t>& offsets,
                       const std::vector<StructureExpansionTerm>& terms)
        : offsets_(&offsets), terms_(&terms) {}

    StructureIncidence(const std::vector<std::vector<StructureExpansionTerm>>& rows)
        : rows_(&rows) {}

    std::size_t size() const {
        return rows_ ? rows_->size() : (offsets_->empty() ? 0 : offsets_->size() - 1);
    }
    bool empty() const {
        return size() == 0;
    }
    std::span<const StructureExpansionTerm> operator[](std::size_t index) const {
        if (rows_)
            return (*rows_)[index];
        return std::span<const StructureExpansionTerm>(*terms_).subspan(
            (*offsets_)[index], (*offsets_)[index + 1] - (*offsets_)[index]);
    }
    std::span<const StructureExpansionTerm> front() const {
        return (*this)[0];
    }

    class Iterator {
      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::span<const StructureExpansionTerm>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;

        Iterator(const std::vector<std::size_t>* offsets,
                 const std::vector<StructureExpansionTerm>* terms,
                 const std::vector<std::vector<StructureExpansionTerm>>* rows,
                 std::size_t index)
            : offsets_(offsets), terms_(terms), rows_(rows), index_(index) {}
        reference operator*() const {
            if (rows_)
                return (*rows_)[index_];
            return std::span<const StructureExpansionTerm>(*terms_).subspan(
                (*offsets_)[index_], (*offsets_)[index_ + 1] - (*offsets_)[index_]);
        }
        Iterator& operator++() {
            ++index_;
            return *this;
        }
        Iterator operator++(int) {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const Iterator& other) const {
            return offsets_ == other.offsets_ && terms_ == other.terms_ && rows_ == other.rows_ &&
                   index_ == other.index_;
        }

      private:
        const std::vector<std::size_t>* offsets_;
        const std::vector<StructureExpansionTerm>* terms_;
        const std::vector<std::vector<StructureExpansionTerm>>* rows_;
        std::size_t index_;
    };

    Iterator begin() const {
        return {offsets_, terms_, rows_, 0};
    }
    Iterator end() const {
        return {offsets_, terms_, rows_, size()};
    }

  private:
    const std::vector<std::size_t>* offsets_ = nullptr;
    const std::vector<StructureExpansionTerm>* terms_ = nullptr;
    const std::vector<std::vector<StructureExpansionTerm>>* rows_ = nullptr;
};

/**
 * @brief Structure-level accumulated matrices.
 *
 * All matrices use column-major storage:
 * `matrix_data[column * n_structures + row]`.
 */
struct StructureAccumulationResult {
    /**
   * @brief Number of VB structures.
   */
    std::size_t n_structures = 0;

    /**
   * @brief Structure overlap matrix.
   */
    std::vector<double> overlap_matrix;

    /**
   * @brief Structure Hamiltonian matrix.
   */
    std::vector<double> hamiltonian_matrix;

    /**
   * @brief Determinant overlap cache used during structure assembly.
   *
   * For diagonal determinant pairs this stores the overlap determinant at the
   * determinant index.
   */
    std::vector<double> determinant_overlap_cache;
};

/**
 * @brief Exact structure-space Hamiltonian and overlap diagonals.
 *
 * Matrix-free eigensolvers retain these vectors for preconditioning without
 * materializing either full structure matrix.
 */
struct StructureDiagonal {
    Eigen::VectorXd hamiltonian;
    Eigen::VectorXd overlap;
};

struct HamiltonianOverlapMatrices {
    std::vector<double> hamiltonian_matrix;
    std::vector<double> overlap_matrix;
};

/**
 * @brief Raw VB structure definitions read directly from the input deck.
 *
 * This object stores the unspecialized structure representation before it is
 * expanded into unique determinants.
 */
struct RawStructureData {
    /**
   * @brief Number of structures.
   */
    std::size_t n_structures = 0;

    /**
   * @brief Total number of electrons.
   */
    int n_total_electrons = 0;

    /**
   * @brief Number of active electrons.
   */
    int n_active_electrons = 0;

    /**
   * @brief Spin multiplicity.
   */
    int spin_multiplicity = 1;

    /**
   * @brief Input wavefunction type flag.
   */
    int wavefunction_type = 0;

    /**
   * @brief Input VB function type flag.
   */
    int vb_function_type = 0;

    /**
   * @brief Flat structure orbital storage in one-based input ordering.
   *
   * The data is packed structure-by-structure. Each structure contributes
   * `n_total_electrons` one-based orbital labels.
   */
    std::vector<int> raw_structure_orbitals;

    std::size_t flat_orbital_count() const {
        return structure_orbital_offset(n_structures);
    }

    /** @brief Wide flat address; the multiplication must precede no narrowing. */
    std::size_t structure_orbital_offset(std::size_t structure_index) const {
        if (n_total_electrons < 0)
            throw std::invalid_argument("negative structure electron count");
        const auto stride = static_cast<std::size_t>(n_total_electrons);
        if (stride && structure_index > std::numeric_limits<std::size_t>::max() / stride)
            throw std::overflow_error("raw structure orbital address exceeds size_t range");
        return structure_index * stride;
    }

    const int* structure_orbitals_data(std::size_t structure_index) const {
        return raw_structure_orbitals.data() + structure_orbital_offset(structure_index);
    }
};

/**
 * @brief Compact combinatorial expansion into unique spin-string pairs.
 *
 * This object owns topology only: unique occupied spin strings, pair IDs, and
 * the sparse pair-to-structure incidence relation. Physical overlap, Hamiltonian,
 * and integral data are supplied separately through their Eigen-based
 * numerical contracts.
 */
struct StructureExpansion {
    /**
   * @brief Number of VB structures.
   */
    std::size_t n_structures = 0;

    /**
   * @brief Unique zero-based alpha occupied spin strings.
   */
    mcscf::StringTable alpha_strings;

    /**
   * @brief Unique zero-based beta occupied spin strings.
   */
    mcscf::StringTable beta_strings;

    /** Spin-string table indices for each unique pair. */
    std::vector<mcscf::StringIndex> pair_alpha;
    std::vector<mcscf::StringIndex> pair_beta;

    /** CSR row boundaries for pair-to-structure terms. */
    std::vector<std::size_t> offsets;

    /**
   * @brief Contiguous pair-to-structure expansion coefficients.
   */
    std::vector<StructureExpansionTerm> terms;

    SpinStringRows alpha() const {
        return {alpha_strings, pair_alpha};
    }
    SpinStringRows beta() const {
        return {beta_strings, pair_beta};
    }
    StructureIncidence incidence() const {
        return {offsets, terms};
    }
};

} // namespace xmvb::vb
