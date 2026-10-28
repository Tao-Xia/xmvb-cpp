#pragma once
#include "cuda/memory/buffer.hpp"
namespace xmvb::cuda {
/** @brief Device canonical integrals T^-1 H T and analytic integral pullbacks.
 * Numerical storage is orbital-sized. Four-index canonical entries are ordered,
 * not eightfold packed. Standard matrix products use cuBLAS.
 */
class SimilarityHamiltonian {
  public:
    SimilarityHamiltonian(Context&, int orbitals);
    void update(linalg::ConstMatrixView one,
                linalg::ConstMatrixView pair,
                linalg::ConstMatrixView upper);
    struct Direction {
        Matrix<double> one, two, inverse, change;
    };
    Direction direction(linalg::ConstMatrixView one,
                        linalg::ConstMatrixView pair,
                        linalg::ConstMatrixView upper) const;
    /** Overwrites symmetric physical cotangents; a direction requests their tangent. */
    void pullback(linalg::ConstMatrixView one,
                  linalg::ConstMatrixView two,
                  linalg::MatrixView physical_one,
                  linalg::MatrixView physical_pair,
                  const Direction* direction = nullptr,
                  linalg::ConstMatrixView upper_direction = {},
                  linalg::ConstMatrixView one_direction = {},
                  linalg::ConstMatrixView two_direction = {}) const;
    linalg::ConstMatrixView one() const {
        return one_.view();
    }
    linalg::ConstMatrixView two() const {
        return two_.view();
    }
    linalg::ConstMatrixView inverse() const {
        return inverse_.view();
    }
    linalg::ConstMatrixView upper() const {
        return upper_.view();
    }

  private:
    Context& context_;
    int n_;
    Matrix<double> one_, two_, upper_, inverse_, physical_one_, physical_two_, change_;
};
} // namespace xmvb::cuda
