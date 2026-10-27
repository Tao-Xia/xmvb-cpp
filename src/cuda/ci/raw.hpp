#pragma once

#include <cstdint>

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {
struct RawCiLane {
    linalg::ConstMatrixView one, two, source, seed;
    linalg::MatrixView image, one_adjoint, two_adjoint;
};
struct RawCiBatch {
    RawCiLane lanes[2];
    int count;
};

/** Borrowed original-source topology. No transformed product carrier is stored. */
struct RawCiTopology {
    const std::int64_t* alpha;
    const std::int64_t* beta;
    std::int64_t groups;
    const std::uint64_t *alpha_masks, *beta_masks, *sorted_masks;
    const std::int64_t* sorted_ids;
    std::int64_t rows;
    int orbitals, alpha_electrons, beta_electrons, multiprocessors;
    bool covalent;
};

/** Shared canonical edge sweep for all RHS and up to two analytic directions. */
void apply_raw_ci(Context&,
                  const RawCiTopology&,
                  linalg::ConstMatrixView source,
                  const std::int64_t* columns,
                  int count,
                  linalg::ConstMatrixView one,
                  linalg::ConstMatrixView two,
                  linalg::MatrixView image,
                  linalg::ConstMatrixView seed = {},
                  linalg::MatrixView one_adjoint = {},
                  linalg::MatrixView two_adjoint = {},
                  RawCiBatch batch = {});
} // namespace xmvb::cuda
