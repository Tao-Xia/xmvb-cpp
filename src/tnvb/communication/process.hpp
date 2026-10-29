#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include "tnvb/communication/response.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "vbscf/integrals/active/space.hpp"
#include "vbscf/derivatives/gradient/active_space/backend.hpp"

namespace xmvb::tnvb {

/** Request-private directory; child leases keep their calculation root alive. */
class StateDirectory {
 public:
  ~StateDirectory();
  StateDirectory(const StateDirectory&) = delete;
  StateDirectory& operator=(const StateDirectory&) = delete;
  const std::filesystem::path& path() const { return path_; }
 private:
  friend class WorkerProcess;
  StateDirectory(std::filesystem::path path,
                 std::shared_ptr<const StateDirectory> parent = {})
      : path_(std::move(path)), parent_(std::move(parent)) {}
  std::filesystem::path path_;
  std::shared_ptr<const StateDirectory> parent_;
};

/** Unique calculation/request scratch owner; never uses the old XMVB worker. */
class WorkerProcess {
 public:
  WorkerProcess();
  ~WorkerProcess();
  WorkerProcess(const WorkerProcess&) = delete;
  WorkerProcess& operator=(const WorkerProcess&) = delete;
  WorkerResponse evaluate(const vb::VbScfInput&,
                          const vb::PreparedActiveSpaceContext&,
                          const vb::ElectronicStateRequest&,
                          bool gradient, bool retain_forward = false,
                          std::shared_ptr<const StateDirectory> resume = {}) const;
  WorkerResponse complete_gradient(const vb::VbScfInput&,
                          const vb::ElectronicStateRequest&,
                          std::shared_ptr<const StateDirectory>) const;
 private:
  std::shared_ptr<const StateDirectory> scratch_;
};

}  // namespace xmvb::tnvb
