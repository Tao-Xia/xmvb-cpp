#pragma once

namespace xmvb::cuda {

/** CPU product-tile batch policy, evaluated by an OpenMP-enabled host TU. */
int product_panel_width();

}  // namespace xmvb::cuda
