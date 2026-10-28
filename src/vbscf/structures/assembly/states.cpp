#include "vbscf/structures/assembly/states.hpp"
#include "vbscf/structures/overlap/metric.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace xmvb::vb {

FixedStructureStates evaluate_fixed_states(const StructureAction& action,
                                           const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                                           const std::vector<double>& weights,
                                           const SelectedMetric* metric) {
    const Eigen::Index count = coefficients.cols();
    if (count == 0 || coefficients.rows() != action.n_structures() ||
        static_cast<Eigen::Index>(weights.size()) != count)
        throw std::invalid_argument("invalid fixed-state dimensions");
    for (const double weight : weights)
        if (!std::isfinite(weight) || weight <= 0 ||
            std::abs(weight - weights.front()) > 64 * std::numeric_limits<double>::epsilon())
            throw std::invalid_argument("fixed-state keyframes require equal positive weights");

    // Normalization must precede the adjoint, but needs only M C, not sigma.
    // E = w Tr[G^-1 C^T H C], G = C^T M C. No closure of C's VB span is assumed.
    StructureActionResult images;
    Eigen::MatrixXd coordinates;
    if (action.supports_integral_direction() && metric) {
        coordinates = metric->upper.triangularView<Eigen::Upper>() * coefficients;
        images.overlap = metric->upper.transpose().triangularView<Eigen::Lower>() * coordinates;
    } else if (action.supports_integral_direction())
        images.overlap = action.apply_overlap(coefficients);
    else
        images = action.apply(coefficients);
    const Eigen::MatrixXd gram = metric && coordinates.size()
                                     ? Eigen::MatrixXd(coordinates.transpose() * coordinates)
                                     : Eigen::MatrixXd(coefficients.transpose() * images.overlap);
    const Eigen::LLT<Eigen::MatrixXd> factor(gram);
    if (factor.info() != Eigen::Success)
        throw std::runtime_error("fixed structure keyframe has a singular state Gram matrix");
    const Eigen::MatrixXd white = factor.matrixU().solve(Eigen::MatrixXd::Identity(count, count));
    const Eigen::MatrixXd normalized = coefficients * white;
    if (coordinates.size())
        coordinates = (coordinates * white).eval();
    std::optional<StructureAdjointState> adjoint;
    if (action.supports_integral_direction()) {
        // Accumulate unshifted integral/generator cotangents along with H C.
        // The full state-space multiplier is known only after this same sweep.
        // Linearity: M*(C*white)=(M*C)*white, with exactly the same state gauge.
        // No second orbital transform or overlap projection is needed.
        const Eigen::MatrixXd normalized_overlap =
            coordinates.size()
                ? Eigen::MatrixXd(metric->upper.transpose().triangularView<Eigen::Lower>() *
                                  coordinates)
                : Eigen::MatrixXd(images.overlap * white);
        adjoint = action.prepare_active_adjoint(
            normalized, weights, std::vector<double>(count, 0.0), &images, &normalized_overlap);
    } else {
        images.hamiltonian = (images.hamiltonian * white).eval();
        images.overlap = (images.overlap * white).eval();
    }
    const Eigen::MatrixXd projected_h =
        coordinates.size()
            ? Eigen::MatrixXd(coordinates.transpose() *
                              metric->upper.transpose().triangularView<Eigen::Lower>().solve(
                                  images.hamiltonian))
            : Eigen::MatrixXd(normalized.transpose() * images.hamiltonian);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(projected_h);
    if (spectrum.info() != Eigen::Success)
        throw std::runtime_error("fixed structure keyframe state diagonalization failed");
    // Match the self-adjoint state model used by the small Ritz solve.
    const Eigen::MatrixXd multiplier = projected_h.selfadjointView<Eigen::Lower>();

    FixedStructureStates result;
    if (adjoint) {
        StructureResponseSweep sweep{{},
                                     std::move(adjoint->one_electron_gradient),
                                     std::move(adjoint->pair_kernel_gradient),
                                     std::move(adjoint->generator_gradient)};
        result.integral_adjoint = action.finish_response(*adjoint, std::move(sweep), -multiplier);
    }
    const Eigen::MatrixXd transform = white * spectrum.eigenvectors();
    result.coefficients = coefficients * transform;
    images.hamiltonian = (images.hamiltonian * spectrum.eigenvectors()).eval();
    images.overlap = (images.overlap * spectrum.eigenvectors()).eval();
    result.energies.resize(count);
    for (Eigen::Index state = 0; state < count; ++state)
        result.energies[state] = spectrum.eigenvalues()[state];
    result.residuals = std::move(images.hamiltonian);
    result.residuals.noalias() -=
        images.overlap *
        Eigen::Map<const Eigen::VectorXd>(result.energies.data(), count).asDiagonal();
    result.coefficient_gradient = (2 * weights.front()) * result.residuals * transform.transpose();
    return result;
}

} // namespace xmvb::vb
