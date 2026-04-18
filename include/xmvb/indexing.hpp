#pragma once

#include <cstddef>
#include <type_traits>

namespace xmvb {

template <typename Integer, std::enable_if_t<std::is_integral_v<Integer> || std::is_enum_v<Integer>, int> = 0>
constexpr std::size_t to_size(Integer value) noexcept {
  return static_cast<std::size_t>(value);
}

template <typename Left, typename Right>
constexpr std::size_t product_size(Left left, Right right) noexcept {
  return to_size(left) * to_size(right);
}

template <typename Row, typename Col, typename ColCount>
constexpr std::size_t row_major_index(Row row, Col col, ColCount n_columns) noexcept {
  return to_size(row) * to_size(n_columns) + to_size(col);
}

template <typename Row, typename Col, typename RowCount>
constexpr std::size_t col_major_index(Row row, Col col, RowCount n_rows) noexcept {
  return to_size(col) * to_size(n_rows) + to_size(row);
}

template <typename Container, typename Integer>
constexpr decltype(auto) index_at(Container& container, Integer index) {
  return container[to_size(index)];
}

template <typename Container, typename Integer>
constexpr decltype(auto) index_at(const Container& container, Integer index) {
  return container[to_size(index)];
}

}  // namespace xmvb
