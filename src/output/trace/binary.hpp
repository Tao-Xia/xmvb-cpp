#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace xmvb::output {

template <typename T>
void write_binary_buffer(
    const std::filesystem::path& path,
    const T* data,
    std::size_t count) {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error(
        "failed to open binary output file: " + path.string());
  }
  if (count > 0) {
    output.write(
        reinterpret_cast<const char*>(data),
        static_cast<std::streamsize>(sizeof(T) * count));
  }
  if (!output) {
    throw std::runtime_error(
        "failed to write binary output file: " + path.string());
  }
}

template <typename Container>
void write_binary_container(
    const std::filesystem::path& path,
    const Container& values) {
  using ValueType = typename Container::value_type;
  write_binary_buffer<ValueType>(path, values.data(), values.size());
}

}  // namespace xmvb::output
