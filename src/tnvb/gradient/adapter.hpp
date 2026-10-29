#pragma once

#include "tnvb/communication/response.hpp"
#include "vbscf/derivatives/gradient/active_space/result.hpp"

namespace xmvb::tnvb {

/** Map orthogonal P/Pi/D3G to raw As/Ah and framed packed Ag. */
void populate_active_space_adjoints(
    const WorkerResponse&, vb::ActiveSpaceGradientResult*);

}  // namespace xmvb::tnvb
