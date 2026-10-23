#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace xmvb::memory {

/** A paired resource limit and usage, measured in bytes. */
struct Constraint {
  std::string source;
  std::size_t limit = 0;
  std::size_t used = 0;
};

struct Snapshot {
  std::size_t rss = 0;
  std::vector<Constraint> constraints;
};

/** Read Linux task/node headroom; proc_root also permits isolated fixtures.
 * All finite cgroup ancestors participate, including mount-root offsets.
 */
Snapshot inspect(const std::filesystem::path& proc_root = "/proc");

struct Budget {
  std::string source;
  std::size_t limit = 0;
  std::size_t used = 0;
  std::size_t pending = 0;
  std::size_t bytes = 0;
};

/** Headroom after allocations not owned by the numerical workspace.
 * No credit is inferred from logical reservations or virtual address space.
 */
Budget plan(const Snapshot&, std::size_t pending = 0,
            std::optional<std::size_t> upper = {});

/** Default for library callers: discover resources once, without a fixed cap. */
std::size_t default_workspace();

/** Configure the CLI preparation phase. Explicit input is an upper bound. */
void configure(std::optional<std::size_t> upper);

/** Refine once when the first CI point and its solver dimensions are known.
 * This is not called inside numerical actions. Later caches retain their
 * allowance, rather than repeatedly subtracting their own resident storage.
 */
void prepare_ci(std::size_t pending);

}  // namespace xmvb::memory
