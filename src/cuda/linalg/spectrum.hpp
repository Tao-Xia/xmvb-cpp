#pragma once

#include <vector>
#include <cusolverDn.h>

#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {

struct SymmetricSpectrum {
    Matrix<double> vectors, values;
    std::vector<double> host_values;
};

SymmetricSpectrum symmetric_spectrum(Context&, cusolverDnHandle_t, linalg::ConstMatrixView);
Matrix<double> spectral_power(Context&, const SymmetricSpectrum&, double exponent);

/** Immutable AO metric and its symmetric roots, shared for an entire SCF.
 * Only construction uploads host data; all factorizations run on the device.
 */
class AoMetric {
  public:
    AoMetric(Context&, const Eigen::Ref<const Eigen::MatrixXd>& overlap);
    AoMetric(const AoMetric&) = delete;
    AoMetric& operator=(const AoMetric&) = delete;
    const Context& context() const noexcept {
        return *context_;
    }
    linalg::ConstMatrixView overlap() const noexcept {
        return overlap_.view();
    }
    linalg::ConstMatrixView root() const noexcept {
        return root_.view();
    }
    linalg::ConstMatrixView inverse_root() const noexcept {
        return inverse_root_.view();
    }

  private:
    Context* context_;
    Matrix<double> overlap_, root_, inverse_root_;
};

} // namespace xmvb::cuda
