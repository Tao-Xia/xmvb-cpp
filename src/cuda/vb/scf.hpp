#pragma once

#include <functional>
#include <cstdint>
#include <string>
#include <vector>

#include "cuda/memory/buffer.hpp"
#include "vbscf/core/contracts/input.hpp"
#include "vbscf/optimization/driver/options.hpp"

namespace xmvb::cuda {

struct ScfIteration {
  int iteration=0,hvp=0,ci_actions=0,keyframes=0;
  double energy=0.0,energy_change=0.0,gradient_inf=0.0,seconds=0.0;
};

struct ScfResult {
  Matrix<double> orbitals,coefficients;
  bool converged=false;
  std::string reason;
  double energy=0.0,gradient_inf=0.0,seconds=0.0;
  double ao_upload_seconds=0.0,host_integral_release_seconds=0.0;
  int iterations=0,hvp=0,ci_actions=0;
  std::uint64_t ci_prepared_points=0,selected_metric_builds=0;
  std::vector<ScfIteration> trace;
};

/** Exact, real single-state GPU VBSCF. Host input supplies AO integrals and
 * sparse topology; all accepted/trial physical calculations, CI roots and
 * CIAH vectors remain on the device. Uses the common CIAH micro/macro policy.
 * Consumes and releases the host ERI payload after uploading its one device
 * representation. Other input topology is borrowed. The Context must outlive
 * the returned orbital/structure vectors. AO setup remains in seconds.
 */
ScfResult solve_scf(Context& context,const vb::VbScfInput& input,
    integrals::AoPairGraph&& integral_stream,double nuclear_repulsion,
    const vb::VbScfOptimizerOptions& options,
    const std::function<void(const ScfIteration&)>& observe={});

}  // namespace xmvb::cuda
