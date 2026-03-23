#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace xmvb {

/**
 * @brief Copy-on-write vector wrapper for large mostly-immutable buffers.
 *
 * Copies share the backing storage until a mutable operation is requested.
 * This keeps `CppVbInput` copies cheap while preserving a `std::vector`-like
 * interface for existing numerical kernels.
 */
template <typename T>
class SharedVector {
public:
  using value_type = T;
  using size_type = typename std::vector<T>::size_type;
  using difference_type = typename std::vector<T>::difference_type;
  using reference = typename std::vector<T>::reference;
  using const_reference = typename std::vector<T>::const_reference;
  using pointer = typename std::vector<T>::pointer;
  using const_pointer = typename std::vector<T>::const_pointer;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  SharedVector()
      : data_(std::make_shared<std::vector<T>>()) {}

  SharedVector(std::vector<T> values)
      : data_(std::make_shared<std::vector<T>>(std::move(values))) {}

  SharedVector(std::initializer_list<T> values)
      : data_(std::make_shared<std::vector<T>>(values)) {}

  SharedVector& operator=(std::vector<T> values) {
    data_ = std::make_shared<std::vector<T>>(std::move(values));
    return *this;
  }

  SharedVector& operator=(std::initializer_list<T> values) {
    data_ = std::make_shared<std::vector<T>>(values);
    return *this;
  }

  operator const std::vector<T>&() const { return *data_; }
  operator std::vector<T>&() { return mutable_vector(); }

  const std::vector<T>& vector() const { return *data_; }

  std::vector<T>& mutable_vector() {
    ensure_unique();
    return *data_;
  }

  bool empty() const noexcept { return data_->empty(); }
  size_type size() const noexcept { return data_->size(); }

  const_reference operator[](size_type index) const { return (*data_)[index]; }
  reference operator[](size_type index) {
    ensure_unique();
    return (*data_)[index];
  }

  const_reference at(size_type index) const { return data_->at(index); }
  reference at(size_type index) {
    ensure_unique();
    return data_->at(index);
  }

  const_reference front() const { return data_->front(); }
  reference front() {
    ensure_unique();
    return data_->front();
  }

  const_reference back() const { return data_->back(); }
  reference back() {
    ensure_unique();
    return data_->back();
  }

  const_pointer data() const noexcept { return data_->data(); }
  pointer data() {
    ensure_unique();
    return data_->data();
  }

  const_iterator begin() const noexcept { return data_->begin(); }
  const_iterator end() const noexcept { return data_->end(); }
  const_iterator cbegin() const noexcept { return data_->cbegin(); }
  const_iterator cend() const noexcept { return data_->cend(); }

  iterator begin() {
    ensure_unique();
    return data_->begin();
  }
  iterator end() {
    ensure_unique();
    return data_->end();
  }

  void clear() { mutable_vector().clear(); }
  void reserve(size_type count) { mutable_vector().reserve(count); }

  void resize(size_type count) { mutable_vector().resize(count); }
  void resize(size_type count, const T& value) { mutable_vector().resize(count, value); }

  template <typename InputIt>
  void assign(InputIt first, InputIt last) {
    mutable_vector().assign(first, last);
  }

  void assign(size_type count, const T& value) {
    mutable_vector().assign(count, value);
  }

  void assign(std::initializer_list<T> values) {
    *this = values;
  }

  void push_back(const T& value) { mutable_vector().push_back(value); }
  void push_back(T&& value) { mutable_vector().push_back(std::move(value)); }

  template <typename... Args>
  reference emplace_back(Args&&... args) {
    auto& values = mutable_vector();
    values.emplace_back(std::forward<Args>(args)...);
    return values.back();
  }

  friend bool operator==(const SharedVector& left, const SharedVector& right) {
    return left.vector() == right.vector();
  }

  friend bool operator!=(const SharedVector& left, const SharedVector& right) {
    return !(left == right);
  }

  friend bool operator==(const SharedVector& left, const std::vector<T>& right) {
    return left.vector() == right;
  }

  friend bool operator!=(const SharedVector& left, const std::vector<T>& right) {
    return !(left == right);
  }

  friend bool operator==(const std::vector<T>& left, const SharedVector& right) {
    return left == right.vector();
  }

  friend bool operator!=(const std::vector<T>& left, const SharedVector& right) {
    return !(left == right);
  }

private:
  void ensure_unique() {
    if (!data_.unique()) {
      data_ = std::make_shared<std::vector<T>>(*data_);
    }
  }

  std::shared_ptr<std::vector<T>> data_;
};

}  // namespace xmvb
