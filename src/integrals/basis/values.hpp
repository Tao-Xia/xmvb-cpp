#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

#include "integrals/ao/libcint_input.hpp"

namespace xmvb::integrals {

struct BasisValues {
    Eigen::MatrixXd value;
    std::array<Eigen::MatrixXd, 3> gradient;
};

/** Evaluates normalized Cartesian AOs at points in bohr.
 *
 * The input and its arrays must remain alive and unmodified for the evaluator's
 * lifetime. AO columns follow libcint's shell, contraction, Cartesian order.
 */
class BasisEvaluator {
  public:
    explicit BasisEvaluator(const LibcintInput& input);

    int n_bf() const {
        return n_bf_;
    }

    /** derivative_order is 0 or 1; order 0 clears all gradient matrices. */
    void evaluate(const Eigen::Ref<const Eigen::MatrixXd>& points,
                  int derivative_order,
                  BasisValues* output) const;

  private:
    struct Monomial {
        int lx;
        int ly;
        int lz;
    };

    struct Shell {
        const double* center = nullptr;
        const double* exponents = nullptr;
        const double* coefficients = nullptr; // contraction-major, nprim entries each
        int nprim = 0;
        int nctr = 0;
        int ao_offset = 0;
        double common_factor = 1.0;
        std::vector<Monomial> monomials;
    };

    const LibcintInput& input_;
    int n_bf_ = 0;
    std::vector<Shell> shells_;
    std::vector<double> ao_normalization_;
};

} // namespace xmvb::integrals
