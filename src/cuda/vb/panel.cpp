#include "cuda/vb/panel.hpp"

#include "parallel/openmp.hpp"

namespace xmvb::cuda {

int product_panel_width() {
  return 8 * cooperative_openmp_thread_count();
}

}  // namespace xmvb::cuda
