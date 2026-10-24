#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <span>
#include <vector>

namespace xmvb::mcscf {

/** @brief Increasing occupied orbitals decoded lazily from one spin string. */
class OccupiedOrbitals {
public:
  class Iterator {
  public:
    using value_type=int;
    using difference_type=std::ptrdiff_t;
    using iterator_category=std::forward_iterator_tag;
    explicit Iterator(std::uint64_t mask=0):mask_(mask) {}
    int operator*() const noexcept { return std::countr_zero(mask_); }
    Iterator& operator++() noexcept { mask_&=mask_-1;return *this; }
    Iterator operator++(int) noexcept { auto copy=*this;++*this;return copy; }
    bool operator==(const Iterator&) const=default;
  private:
    std::uint64_t mask_;
  };
  explicit OccupiedOrbitals(std::uint64_t mask):mask_(mask) {}
  Iterator begin() const noexcept { return Iterator(mask_); }
  Iterator end() const noexcept { return Iterator(); }
  int size() const noexcept { return std::popcount(mask_); }
private:
  std::uint64_t mask_;
};

/** @brief Number of fixed-particle strings; zero for an absent sector. */
std::size_t string_count(int n_orbitals, int n_electrons);

/** @brief Shared immutable fixed-particle spin strings, in the caller's ordering.
 * Occupations have one uint64_t representation. A combinatorial rank maps to
 * the caller's index; canonical increasing-mask order needs no permutation.
 * Copies share masks and rank metadata, with no occupation lists or hash map.
 */
class SpinStrings {
public:
  SpinStrings()=default;
  SpinStrings(int n_orbitals, int n_electrons);
  SpinStrings(int n_orbitals, std::span<const std::uint64_t> masks);
  SpinStrings(int n_orbitals, const std::vector<std::vector<int>>& occupations);
  const std::vector<std::uint64_t>& masks() const noexcept;
  int index(std::uint64_t mask) const;
  int size() const noexcept;
  int orbitals() const noexcept;
  int electrons() const noexcept;
  bool complete() const noexcept;
  OccupiedOrbitals occupied(int string) const noexcept { return OccupiedOrbitals(masks()[string]); }
  /** @brief Materialize only for consumers requiring explicit submatrix indices. */
  std::vector<std::vector<int>> decode() const;
  std::size_t dynamic_bytes() const noexcept;
  bool shares_storage(const SpinStrings& other) const noexcept { return data_==other.data_; }
private:
  struct Data;
  std::shared_ptr<const Data> data_;
};

}  // namespace xmvb::mcscf
