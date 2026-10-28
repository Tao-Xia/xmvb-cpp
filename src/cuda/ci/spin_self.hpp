#pragma once

#include <cstdint>
#include <span>

#include "cuda/memory/buffer.hpp"

namespace xmvb::cuda {

/** Nonorthogonal spin-string self overlaps and normalized channel moments.
 * For occupied columns B of the orbital frame: G=B^T B, s=det(G),
 * D=B G^-1 B^T. Batched vendor LU/GEMM acts on bounded string tiles.
 * No compound transform, spin-pair square or product-space scratch is stored.
 */
class SpinSelf {
  public:
    SpinSelf(Context& context, int orbitals, std::span<const std::uint64_t> masks);
    void update(linalg::ConstMatrixView frame, linalg::ConstMatrixView channel_kernel);
    linalg::ConstMatrixView moments() const noexcept {
        return moments_.view();
    }
    linalg::ConstMatrixView energy() const noexcept {
        return energy_.view();
    }
    linalg::ConstMatrixView overlap() const noexcept {
        return overlap_.view();
    }

  private:
    Context* context_;
    int orbitals_, occupied_, channels_;
    std::int64_t strings_;
    Buffer<int> occupations_, first_, second_;
    Matrix<double> moments_, energy_, overlap_;
};

} // namespace xmvb::cuda
