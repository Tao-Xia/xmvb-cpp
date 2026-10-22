#pragma once

#include <cstdint>

namespace xmvb::cuda {

/** Identity of immutable orbital-frame contents, including copies of them.
 * Generate a new identity whenever the contents change. The empty identity
 * means that the caller cannot certify reuse and forces a frame update.
 */
class OrbitalFrameIdentity {
 public:
  OrbitalFrameIdentity() = default;
  static OrbitalFrameIdentity fresh();
  explicit operator bool() const noexcept { return value_ != 0; }

  friend bool operator==(OrbitalFrameIdentity lhs,
                         OrbitalFrameIdentity rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }
  friend bool operator!=(OrbitalFrameIdentity lhs,
                         OrbitalFrameIdentity rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  explicit OrbitalFrameIdentity(std::uint64_t value) : value_(value) {}
  std::uint64_t value_ = 0;
};

} // namespace xmvb::cuda
