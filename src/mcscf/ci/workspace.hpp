#pragma once

#include <cstddef>
#include <memory>

#include "core/workspace.hpp"

namespace xmvb::mcscf {

/** One action owns its working allowance and leases the shared sigma slab.
 * Nested actions borrow the same reservation. Accepted panels and the idle
 * slab keep their charge after the action returns; no numeric data is copied.
 */
class CiWorkspace {
  public:
    explicit CiWorkspace(std::size_t bytes = workspace::available(), bool use_sigma = true);
    ~CiWorkspace();
    CiWorkspace(const CiWorkspace&) = delete;
    CiWorkspace& operator=(const CiWorkspace&) = delete;
    std::size_t bytes() const noexcept;
    /** Bound idle sigma residency before allocating the other partitions. */
    void sigma_budget(std::size_t bytes) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::mcscf
