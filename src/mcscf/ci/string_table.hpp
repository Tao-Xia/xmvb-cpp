#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "mcscf/ci/strings.hpp"

namespace xmvb::mcscf {

/** @brief Occupation topology with one uint64_t per ordinary spin string.
 * Wider strings use consecutive uint64_t words. No orbital-index rows are
 * retained. Explicit row decoding belongs to local submatrix consumers.
 */
class StringTable {
  public:
    StringTable() = default;
    StringTable(const std::vector<std::vector<int>>& rows) {
        reserve(rows.size());
        for (const auto& row : rows)
            push_back(row);
    }
    StringTable(std::initializer_list<std::vector<int>> rows) {
        reserve(rows.size());
        for (const auto& row : rows)
            push_back(row);
    }
    static StringTable from_basis(const SpinStrings& basis) {
        StringTable result;
        result.data_->basis = basis;
        result.data_->frozen = basis.masks();
        return result;
    }
    std::size_t size() const noexcept {
        return storage().size() / data_->width;
    }
    bool empty() const noexcept {
        return storage().empty();
    }
    std::size_t word_width() const noexcept {
        return data_->width;
    }
    std::size_t capacity() const noexcept {
        return !data_->frozen.empty() ? data_->frozen.size()
                                      : data_->words.capacity() / data_->width;
    }
    std::size_t dynamic_bytes() const noexcept {
        return data_->words.capacity() * sizeof(std::uint64_t) +
               (data_->frozen.empty() ? 0 : data_->basis.dynamic_bytes());
    }
    void reserve(std::size_t count) {
        own();
        data_->words.reserve(count * data_->width);
    }
    void clear() {
        data_ = std::make_shared<Data>();
    }
    std::span<const std::uint64_t> words(std::size_t row) const noexcept {
        return storage().subspan(row * data_->width, data_->width);
    }
    std::span<const std::uint64_t> masks() const {
        if (data_->width != 1)
            throw std::invalid_argument("spin string requires multiple occupation words");
        return storage();
    }
    std::uint64_t mask(std::size_t row) const {
        return masks()[row];
    }
    int electron_count(std::size_t row = 0) const noexcept {
        int count = 0;
        for (const auto word : words(row))
            count += std::popcount(word);
        return count;
    }
    void push_words(std::span<const std::uint64_t> row) {
        own();
        auto required = row.size();
        while (required > 1 && row[required - 1] == 0)
            --required;
        if (required > data_->width) {
            const auto count = size();
            std::vector<std::uint64_t> widened(count * required, 0);
            for (std::size_t i = 0; i < count; ++i)
                std::copy_n(data_->words.data() + i * data_->width,
                            data_->width,
                            widened.data() + i * required);
            data_->words = std::move(widened);
            data_->width = required;
        }
        const auto begin = data_->words.size();
        data_->words.resize(begin + data_->width, 0);
        std::copy_n(row.data(), required, data_->words.data() + begin);
    }
    void push_mask(std::uint64_t mask) {
        push_words({&mask, 1});
    }
    void push_back(const std::vector<int>& occupied) {
        std::vector<std::uint64_t> row(1, 0);
        int previous = -1;
        for (const int p : occupied) {
            if (p < 0 || p <= previous)
                throw std::invalid_argument("invalid spin occupation ordering");
            if (row.size() <= static_cast<std::size_t>(p / 64))
                row.resize(p / 64 + 1, 0);
            row[p / 64] |= std::uint64_t{1} << (p % 64);
            previous = p;
        }
        push_words(row);
    }
    /** @brief Decode indices only for a consumer that extracts an orbital submatrix. */
    std::vector<int> operator[](std::size_t row) const {
        std::vector<int> indices;
        indices.reserve(electron_count(row));
        std::size_t w = 0;
        for (auto bits : words(row)) {
            while (bits) {
                indices.push_back(64 * w + std::countr_zero(bits));
                bits &= bits - 1;
            }
            ++w;
        }
        return indices;
    }
    std::vector<int> front() const {
        return (*this)[0];
    }
    bool same_row(std::size_t a, const StringTable& other, std::size_t b) const noexcept {
        const auto left = words(a), right = other.words(b);
        for (std::size_t i = 0; i < std::max(left.size(), right.size()); ++i)
            if ((i < left.size() ? left[i] : 0) != (i < right.size() ? right[i] : 0))
                return false;
        return true;
    }
    /** @brief Lexicographic orbital-list order without decoding either row. */
    bool less(std::size_t a, std::size_t b) const noexcept {
        const auto left = words(a), right = words(b);
        for (std::size_t w = 0; w < left.size(); ++w) {
            const auto difference = left[w] ^ right[w];
            if (!difference)
                continue;
            const auto first = std::countr_zero(difference);
            const bool left_has = (left[w] >> first) & 1;
            const auto remaining = left_has ? right : left;
            bool continues = (remaining[w] >> first) != 0;
            for (std::size_t k = w + 1; k < remaining.size() && !continues; ++k)
                continues = remaining[k] != 0;
            return left_has ? continues : !continues;
        }
        return false;
    }
    bool operator==(const StringTable& other) const noexcept {
        if (data_ == other.data_)
            return true;
        if (size() != other.size())
            return false;
        if (data_->width == other.data_->width)
            return std::equal(storage().begin(), storage().end(), other.storage().begin());
        for (std::size_t i = 0; i < size(); ++i)
            if (!same_row(i, other, i))
                return false;
        return true;
    }
    SpinStrings basis(int orbitals) const {
        if (!data_->frozen.empty() && data_->basis.orbitals() == orbitals)
            return data_->basis;
        return SpinStrings(orbitals, masks());
    }
    class Iterator {
      public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::vector<int>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = void;
        Iterator(const StringTable* table, std::size_t row) : table_(table), row_(row) {}
        value_type operator*() const {
            return (*table_)[row_];
        }
        Iterator& operator++() {
            ++row_;
            return *this;
        }
        Iterator operator++(int) {
            auto previous = *this;
            ++*this;
            return previous;
        }
        bool operator==(const Iterator&) const = default;

      private:
        const StringTable* table_;
        std::size_t row_;
    };
    Iterator begin() const {
        return {this, 0};
    }
    Iterator end() const {
        return {this, size()};
    }

  private:
    struct Data {
        std::vector<std::uint64_t> words;
        SpinStrings basis;
        std::span<const std::uint64_t> frozen;
        std::size_t width = 1;
    };
    std::shared_ptr<Data> data_ = std::make_shared<Data>();
    void own() {
        if (!data_.unique())
            data_ = std::make_shared<Data>(*data_);
        if (!data_->frozen.empty()) {
            const auto& masks = data_->basis.masks();
            data_->words.assign(masks.begin(), masks.end());
            data_->basis = SpinStrings();
            data_->frozen = {};
        }
    }
    std::span<const std::uint64_t> storage() const noexcept {
        return !data_->frozen.empty() ? data_->frozen
                                      : std::span<const std::uint64_t>(data_->words);
    }
};

} // namespace xmvb::mcscf
