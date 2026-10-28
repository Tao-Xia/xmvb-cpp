#include "vbscf/derivatives/hessian/coupled/structure.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/QR>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/metric.hpp"
#include "vbscf/structures/overlap/metric.hpp"
#include "vbscf/structures/overlap/spin_metric.hpp"
#include "linalg/metric_solve.hpp"
#include "mcscf/optimization/trace.hpp"

namespace xmvb::vb {
namespace {

bool weights_are_equal(const std::vector<double>& weights) {
    if (weights.empty())
        return false;
    const double reference = weights.front();
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(reference));
    return std::all_of(weights.begin(), weights.end(), [&](double weight) {
        return std::abs(weight - reference) <= tolerance;
    });
}

} // namespace

StructureTangentOperator::StructureTangentOperator(
    std::shared_ptr<const AcceptedPointContext> accepted_point, const StructureAction& action)
    : accepted_point_(std::move(accepted_point)), action_(&action) {
    if (accepted_point_ == nullptr) {
        throw std::invalid_argument("coupled structure operator requires an accepted point");
    }
    n_structures_ = accepted_point_->n_structures;
    n_states_ = static_cast<int>(accepted_point_->selected_state_indices.size());
    if (n_structures_ <= 0 || n_states_ <= 0 || action_->n_structures() != n_structures_ ||
        accepted_point_->selected_state_eigenvectors.rows() != n_structures_ ||
        accepted_point_->selected_state_eigenvectors.cols() != n_states_ ||
        accepted_point_->selected_state_energies.size() != static_cast<std::size_t>(n_states_) ||
        accepted_point_->normalized_state_weights.size() != static_cast<std::size_t>(n_states_)) {
        throw std::invalid_argument("coupled structure operator dimensions are inconsistent");
    }
    if (n_states_ > 1 && !weights_are_equal(accepted_point_->normalized_state_weights)) {
        throw std::invalid_argument(
            "coupled structure operator supports only equal-weight state averages");
    }

    energies_ = Eigen::Map<const Eigen::VectorXd>(accepted_point_->selected_state_energies.data(),
                                                  n_states_);
    coordinate_scales_.resize(n_states_);
    for (int state = 0; state < n_states_; ++state) {
        const double weight =
            accepted_point_->normalized_state_weights[static_cast<std::size_t>(state)];
        if (!(weight > 0.0) || !std::isfinite(weight) || !std::isfinite(energies_[state])) {
            throw std::invalid_argument(
                "coupled structure operator requires positive finite state weights and energies");
        }
        coordinate_scales_[state] = std::sqrt(2.0 * weight);
    }

    const StructureDiagonal& diagonal = action_->preconditioner_diagonal();
    if (diagonal.hamiltonian.size() != n_structures_ || diagonal.overlap.size() != n_structures_ ||
        !diagonal.hamiltonian.allFinite() || !diagonal.overlap.allFinite() ||
        (diagonal.overlap.array() <= 0.0).any()) {
        throw std::invalid_argument(
            "coupled structure operator requires finite positive H/S diagonals");
    }
    metric_diagonal_ = diagonal.overlap;
    hessian_diagonal_.resize(n_structures_, n_states_);
    absolute_hessian_diagonal_.resize(n_structures_, n_states_);
    for (int state = 0; state < n_states_; ++state) {
        hessian_diagonal_.col(state) = diagonal.hamiltonian - energies_[state] * diagonal.overlap;
        absolute_hessian_diagonal_.col(state) = hessian_diagonal_.col(state).cwiseAbs();
    }

    selected_ = accepted_point_->selected_state_eigenvectors;
    if (!accepted_point_->selected_state_structure_images.has_value()) {
        throw std::invalid_argument(
            "coupled structure operator requires cached selected-state structure images");
    }
    const StructureActionResult& selected_images =
        accepted_point_->selected_state_structure_images.value();
    if (selected_images.hamiltonian.rows() != n_structures_ ||
        selected_images.hamiltonian.cols() != n_states_ ||
        !selected_images.hamiltonian.allFinite()) {
        throw std::runtime_error(
            "coupled structure operator received invalid selected Hamiltonian images");
    }
    if (selected_images.overlap.rows() != n_structures_ ||
        selected_images.overlap.cols() != n_states_ || !selected_images.overlap.allFinite()) {
        throw std::runtime_error(
            "coupled structure operator received invalid selected overlap images");
    }
    const Eigen::MatrixXd selected_metric = selected_.transpose() * selected_images.overlap;
    const double metric_error =
        (selected_metric - Eigen::MatrixXd::Identity(n_states_, n_states_)).norm();
    const double metric_tolerance =
        1.0e3 * std::numeric_limits<double>::epsilon() * std::max<Eigen::Index>(1, n_structures_);
    if (!std::isfinite(metric_error) || metric_error > metric_tolerance) {
        throw std::invalid_argument(
            "coupled structure operator requires S-orthonormal selected states");
    }
    selected_metric_inverse_ = selected_metric.inverse();
    if (!selected_metric_inverse_.allFinite()) {
        throw std::runtime_error("coupled structure selected metric inverse is not finite");
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> factor(selected_images.overlap);
    factor.setThreshold(std::numeric_limits<double>::epsilon() *
                        std::max<Eigen::Index>(n_structures_, n_states_));
    if (factor.rank() != n_states_) {
        throw std::invalid_argument("selected-state metric images are linearly dependent");
    }
    constraint_qr_.compute(selected_images.overlap);
    metric_inverse_ = accepted_point_->metric_inverse(action);
    if (metric_inverse_)
        mcscf::diagnostics::ciah_trace("ciah_metric_factors",
                                       "bytes=%zu,blocks=%zu,max_block=%d,carrier_diagonal=%d",
                                       metric_inverse_->factor_bytes(),
                                       metric_inverse_->block_count(),
                                       metric_inverse_->max_block_size(),
                                       static_cast<int>(action_->supports_carrier_diagonal()));
}

Eigen::Index StructureTangentOperator::n_structures() const noexcept {
    return n_structures_;
}

int StructureTangentOperator::n_states() const noexcept {
    return n_states_;
}

Eigen::Index StructureTangentOperator::tangent_size() const noexcept {
    return (n_structures_ - n_states_) * n_states_;
}

void StructureTangentOperator::validate_shape(const Eigen::MatrixXd& coefficients) const {
    if (coefficients.rows() != n_structures_ || coefficients.cols() != n_states_ ||
        !coefficients.allFinite()) {
        throw std::invalid_argument(
            "structure tangent has incompatible dimensions or non-finite values");
    }
}

void StructureTangentOperator::project_in_place(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr) {
        throw std::invalid_argument("structure tangent output must not be null");
    }
    validate_shape(*coefficients);
    Eigen::MatrixXd transformed = constraint_qr_.householderQ().adjoint() * (*coefficients);
    transformed.topRows(n_states_).setZero();
    *coefficients = constraint_qr_.householderQ() * transformed;
}

Eigen::VectorXd StructureTangentOperator::coordinates(const StructureTangent& tangent) const {
    validate_shape(tangent.scaled_coefficients);
    const Eigen::MatrixXd transformed =
        constraint_qr_.householderQ().adjoint() * tangent.scaled_coefficients;
    const double tolerance = 1024.0 * std::numeric_limits<double>::epsilon() *
                             std::max<Eigen::Index>(1, n_structures_ * n_states_) *
                             std::max(1.0, tangent.scaled_coefficients.norm());
    if (transformed.topRows(n_states_).norm() > tolerance) {
        throw std::invalid_argument("structure tangent is not horizontal");
    }
    const Eigen::MatrixXd independent = transformed.bottomRows(n_structures_ - n_states_);
    return Eigen::Map<const Eigen::VectorXd>(independent.data(), independent.size());
}

Eigen::VectorXd StructureTangentOperator::project_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& scaled_coefficients) const {
    validate_shape(scaled_coefficients);
    const Eigen::MatrixXd transformed =
        constraint_qr_.householderQ().adjoint() * scaled_coefficients;
    const Eigen::MatrixXd independent = transformed.bottomRows(n_structures_ - n_states_);
    return Eigen::Map<const Eigen::VectorXd>(independent.data(), independent.size());
}

StructureTangent StructureTangentOperator::expand(const Eigen::VectorXd& coordinates) const {
    if (coordinates.size() != tangent_size() || !coordinates.allFinite()) {
        throw std::invalid_argument("structure coordinates have incompatible dimensions or values");
    }
    Eigen::MatrixXd transformed = Eigen::MatrixXd::Zero(n_structures_, n_states_);
    const Eigen::Index horizontal_rows = n_structures_ - n_states_;
    const Eigen::Map<const Eigen::MatrixXd> independent(
        coordinates.data(), horizontal_rows, n_states_);
    transformed.bottomRows(horizontal_rows) = independent;
    return StructureTangent{constraint_qr_.householderQ() * transformed};
}

StructureTangent StructureTangentOperator::project(
    const Eigen::Ref<const Eigen::MatrixXd>& scaled_coefficients) const {
    StructureTangent result{scaled_coefficients};
    project_in_place(&result.scaled_coefficients);
    return result;
}

StructureTangent StructureTangentOperator::from_coefficient_response(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficient_response) const {
    validate_shape(coefficient_response);
    Eigen::MatrixXd scaled = coefficient_response;
    for (int state = 0; state < n_states_; ++state) {
        scaled.col(state) *= coordinate_scales_[state];
    }
    return project(scaled);
}

Eigen::MatrixXd
StructureTangentOperator::coefficient_response(const StructureTangent& tangent) const {
    StructureTangent horizontal = project(tangent.scaled_coefficients);
    for (int state = 0; state < n_states_; ++state) {
        horizontal.scaled_coefficients.col(state) /= coordinate_scales_[state];
    }
    return horizontal.scaled_coefficients;
}

Eigen::MatrixXd StructureTangentOperator::coefficient_response_coordinates(
    const Eigen::VectorXd& coordinates) const {
    Eigen::MatrixXd response = expand(coordinates).scaled_coefficients;
    for (int state = 0; state < n_states_; ++state) {
        response.col(state) /= coordinate_scales_[state];
    }
    return response;
}

double
StructureTangentOperator::maximum_coefficient_component(const StructureTangent& tangent) const {
    const Eigen::MatrixXd response = coefficient_response(tangent);
    if (response.size() == 0)
        return 0.0;
    if (!action_->supports_integral_direction()) {
        return response.cwiseAbs().maxCoeff();
    }
    const Eigen::MatrixXd orthogonal = action_->orthogonalize_structure_block(response);
    return orthogonal.size() == 0 ? 0.0 : orthogonal.cwiseAbs().maxCoeff();
}

double StructureTangentOperator::maximum_coefficient_component_coordinates(
    const Eigen::VectorXd& coordinates) const {
    const Eigen::MatrixXd response = coefficient_response_coordinates(coordinates);
    if (response.size() == 0)
        return 0.0;
    if (!action_->supports_integral_direction()) {
        return response.cwiseAbs().maxCoeff();
    }
    const Eigen::MatrixXd orthogonal = action_->orthogonalize_structure_block(response);
    return orthogonal.size() == 0 ? 0.0 : orthogonal.cwiseAbs().maxCoeff();
}

StructureTangent StructureTangentOperator::apply_hessian(const StructureTangent& tangent) const {
    return apply_coupling(tangent).hessian;
}

Eigen::VectorXd
StructureTangentOperator::apply_hessian_coordinates(const Eigen::VectorXd& coordinates) const {
    return apply_coupling_coordinates(coordinates).hessian_coordinates;
}

StructureCouplingAction
StructureTangentOperator::apply_coupling(const StructureTangent& tangent) const {
    const Eigen::VectorXd tangent_coordinates = coordinates(tangent);
    StructureCoordinateCouplingAction coordinate_action =
        apply_coupling_coordinates(tangent_coordinates);
    return StructureCouplingAction{expand(coordinate_action.hessian_coordinates),
                                   std::move(coordinate_action.coefficient_response),
                                   std::move(coordinate_action.adjoint_multipliers)};
}

StructureCoordinateCouplingAction StructureTangentOperator::apply_coupling_coordinates(
    const Eigen::VectorXd& coordinates, StructureActiveIntegralAdjoint* adjoint) const {
    const StructureCoordinateCouplingBlock block =
        apply_coupling_coordinate_block(coordinates, adjoint);
    return StructureCoordinateCouplingAction{block.hessian_coordinates.col(0),
                                             block.metric_coordinates.col(0),
                                             block.coefficient_responses.front(),
                                             block.adjoint_multipliers.front()};
}

StructureCoordinateCouplingAction StructureTangentOperator::finish_coupling_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    StructureResponseSweep sweep,
    StructureActiveIntegralAdjoint* adjoint,
    const Eigen::MatrixXd* multiplier_correction) const {
    if (coefficients.rows() != n_structures_ || coefficients.cols() != n_states_)
        throw std::invalid_argument("joint structure response has incompatible coordinates");
    const auto block = finish_coupling_block(
        coefficients, std::move(sweep.images), &sweep, adjoint, multiplier_correction);
    return {block.hessian_coordinates.col(0),
            block.metric_coordinates.col(0),
            block.coefficient_responses.front(),
            block.adjoint_multipliers.front()};
}

StructureCoordinateCouplingBlock StructureTangentOperator::apply_coupling_coordinate_block(
    const Eigen::Ref<const Eigen::MatrixXd>& coordinates,
    StructureActiveIntegralAdjoint* adjoint) const {
    if (coordinates.rows() != tangent_size() || coordinates.cols() == 0 ||
        !coordinates.allFinite()) {
        throw std::invalid_argument(
            "structure coordinate block has incompatible dimensions or values");
    }
    const Eigen::Index width = coordinates.cols();
    if (adjoint && (width != 1 || !action_->supports_integral_direction()))
        throw std::invalid_argument("joint response requires one direct-CI direction");
    const Eigen::Index horizontal_rows = n_structures_ - n_states_;
    Eigen::MatrixXd transformed = Eigen::MatrixXd::Zero(n_structures_, n_states_ * width);
    for (Eigen::Index block = 0; block < width; ++block) {
        const Eigen::Map<const Eigen::MatrixXd> independent(
            coordinates.col(block).data(), horizontal_rows, n_states_);
        transformed.middleCols(block * n_states_, n_states_).bottomRows(horizontal_rows) =
            independent;
    }
    Eigen::MatrixXd raw = constraint_qr_.householderQ() * transformed;
    for (Eigen::Index block = 0; block < width; ++block) {
        for (int state = 0; state < n_states_; ++state) {
            raw.col(block * n_states_ + state) /= coordinate_scales_[state];
        }
    }

    StructureResponseSweep sweep;
    StructureActionResult images;
    if (adjoint) {
        if (!accepted_point_->structure_adjoint_state)
            accepted_point_->structure_adjoint_state =
                action_->prepare_active_adjoint(accepted_point_->selected_state_eigenvectors,
                                                accepted_point_->normalized_state_weights,
                                                accepted_point_->selected_state_energies);
        sweep = action_->apply_response(*accepted_point_->structure_adjoint_state, raw);
        images = std::move(sweep.images);
    } else {
        images = action_->apply(raw);
    }
    return finish_coupling_block(raw, std::move(images), adjoint ? &sweep : nullptr, adjoint);
}

StructureCoordinateCouplingBlock StructureTangentOperator::finish_coupling_block(
    const Eigen::Ref<const Eigen::MatrixXd>& raw,
    StructureActionResult images,
    StructureResponseSweep* sweep,
    StructureActiveIntegralAdjoint* adjoint,
    const Eigen::MatrixXd* multiplier_correction) const {
    const Eigen::Index width = raw.cols() / n_states_;
    const Eigen::Index horizontal_rows = n_structures_ - n_states_;
    Eigen::MatrixXd shifted = images.hamiltonian;
    for (Eigen::Index block = 0; block < width; ++block) {
        for (int state = 0; state < n_states_; ++state) {
            const Eigen::Index column = block * n_states_ + state;
            shifted.col(column).noalias() -= energies_[state] * images.overlap.col(column);
        }
    }

    StructureCoordinateCouplingBlock result;
    result.coefficient_responses.reserve(width);
    result.adjoint_multipliers.reserve(width);
    for (Eigen::Index block = 0; block < width; ++block) {
        result.coefficient_responses.push_back(raw.middleCols(block * n_states_, n_states_));
        result.adjoint_multipliers.push_back(
            -selected_metric_inverse_ *
            (selected_.transpose() * shifted.middleCols(block * n_states_, n_states_)));
    }

    Eigen::MatrixXd scaled_shifted = std::move(shifted);
    if (multiplier_correction)
        result.adjoint_multipliers.front() += *multiplier_correction;
    if (adjoint)
        *adjoint = action_->finish_response(*accepted_point_->structure_adjoint_state,
                                            std::move(*sweep),
                                            result.adjoint_multipliers.front());
    Eigen::MatrixXd scaled_overlap = images.overlap;
    for (Eigen::Index block = 0; block < width; ++block) {
        for (int state = 0; state < n_states_; ++state) {
            const Eigen::Index column = block * n_states_ + state;
            scaled_shifted.col(column) *= coordinate_scales_[state];
            scaled_overlap.col(column) *= coordinate_scales_[state];
        }
    }
    const Eigen::MatrixXd hessian_transformed =
        constraint_qr_.householderQ().adjoint() * scaled_shifted;
    const Eigen::MatrixXd metric_transformed =
        constraint_qr_.householderQ().adjoint() * scaled_overlap;
    result.hessian_coordinates.resize(tangent_size(), width);
    result.metric_coordinates.resize(tangent_size(), width);
    for (Eigen::Index block = 0; block < width; ++block) {
        const Eigen::MatrixXd hessian_independent =
            hessian_transformed.middleCols(block * n_states_, n_states_)
                .bottomRows(horizontal_rows);
        const Eigen::MatrixXd metric_independent =
            metric_transformed.middleCols(block * n_states_, n_states_).bottomRows(horizontal_rows);
        result.hessian_coordinates.col(block) = Eigen::Map<const Eigen::VectorXd>(
            hessian_independent.data(), hessian_independent.size());
        result.metric_coordinates.col(block) =
            Eigen::Map<const Eigen::VectorXd>(metric_independent.data(), metric_independent.size());
    }
    return result;
}

StructureTangent StructureTangentOperator::apply_metric(const StructureTangent& tangent) const {
    return expand(apply_metric_coordinates(coordinates(tangent)));
}

Eigen::VectorXd
StructureTangentOperator::apply_metric_coordinates(const Eigen::VectorXd& coordinates) const {
    const StructureTangent horizontal = expand(coordinates);
    // The accepted Gram is exactly F^T F at this chart. Reusing it avoids
    // replaying F and F^T just to apply the same structure tangent metric.
    const Eigen::MatrixXd overlap =
        accepted_point_->selected_metric
            ? accepted_point_->selected_metric->apply(horizontal.scaled_coefficients)
            : action_->apply_overlap(horizontal.scaled_coefficients);
    validate_shape(overlap);
    return project_coordinates(overlap);
}

double StructureTangentOperator::metric_inner_product(const Eigen::VectorXd& left,
                                                      const Eigen::VectorXd& right) const {
    if (left.size() != tangent_size() || right.size() != tangent_size() || !left.allFinite() ||
        !right.allFinite()) {
        throw std::invalid_argument(
            "structure metric operands have incompatible dimensions or values");
    }
    const double product = left.dot(apply_metric_coordinates(right));
    if (!std::isfinite(product)) {
        throw std::runtime_error("structure metric inner product is not finite");
    }
    return product;
}

Eigen::VectorXd
StructureTangentOperator::apply_inverse_metric_diagonal(const Eigen::VectorXd& covector) const {
    if (covector.size() != tangent_size() || !covector.allFinite()) {
        throw std::invalid_argument(
            "structure dual-norm covector has incompatible dimensions or values");
    }
    Eigen::MatrixXd ambient = expand(covector).scaled_coefficients;
    for (int state = 0; state < n_states_; ++state) {
        ambient.col(state).array() /= metric_diagonal_.array();
    }
    return project_coordinates(ambient);
}

double StructureTangentOperator::diagonal_dual_norm(const Eigen::VectorXd& covector) const {
    const Eigen::VectorXd inverse_image = apply_inverse_metric_diagonal(covector);
    const double squared_norm = covector.dot(inverse_image);
    const double tolerance =
        1024.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, covector.squaredNorm());
    if (!std::isfinite(squared_norm) || squared_norm < -tolerance) {
        throw std::runtime_error("structure diagonal dual norm is not positive semidefinite");
    }
    return std::sqrt(std::max(0.0, squared_norm));
}

Eigen::VectorXd
StructureTangentOperator::apply_inverse_metric_model(const Eigen::VectorXd& covector) const {
    const auto& selected_metric = accepted_point_->selected_metric;
    if (!metric_inverse_ && !selected_metric)
        return apply_inverse_metric_diagonal(covector);
    const Eigen::MatrixXd rhs = expand(covector).scaled_coefficients;
    Eigen::MatrixXd image;
    if (metric_inverse_)
        image = metric_inverse_->apply(rhs);
    else
        image = selected_metric->solve(rhs);
    // Z spans ker[(M C)^T]. The exact horizontal inverse is
    // Z^T [M^-1 - C (C^T M C)^-1 C^T] Z, not Z^T M^-1 Z.
    image.noalias() -= selected_ * (selected_metric_inverse_ * (selected_.transpose() * rhs));
    return project_coordinates(image);
}

double StructureTangentOperator::dual_norm(const Eigen::VectorXd& covector,
                                           linalg::MetricSolveWorkspace* workspace) const {
    return metric_image(covector, workspace).norm(covector);
}

linalg::MetricImage
StructureTangentOperator::metric_image(const Eigen::VectorXd& covector,
                                       linalg::MetricSolveWorkspace* workspace,
                                       const linalg::MetricImage* initial) const {
    if (covector.size() != tangent_size() || !covector.allFinite())
        throw std::invalid_argument("invalid structure metric covector");
    const bool trace = mcscf::diagnostics::ciah_event_trace_enabled();
    const auto start =
        trace ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    linalg::MetricSolveStats stats;
    const auto action = [this](const Eigen::VectorXd& x) { return apply_metric_coordinates(x); };
    const auto precondition = [this](const Eigen::VectorXd& x) {
        return apply_inverse_metric_model(x);
    };
    // Exact complete-space or resident selected factors need no recycling.
    // Recycle only streamed selected solves that repeat long Krylov traversals.
    linalg::MetricImage result;
    if (workspace) {
        result = workspace->image(covector,
                                  action,
                                  precondition,
                                  trace ? &stats : nullptr,
                                  initial,
                                  !metric_inverse_ && !accepted_point_->selected_metric);
    } else {
        Eigen::VectorXd image;
        result.inverse = linalg::solve_metric(covector,
                                              action,
                                              precondition,
                                              trace ? &stats : nullptr,
                                              initial ? &initial->inverse : nullptr,
                                              &image);
        result.residual = covector - image;
    }
    if (trace)
        mcscf::diagnostics::ciah_trace(
            "ciah_structure_norm",
            "seconds=%.17g,iterations=%d,actions=%d,certifications=%d,relative_residual=%.17g,"
            "cache_hit=%d,seeded=%d,propagated_residual=%.17g",
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
            stats.iterations,
            stats.actions,
            stats.certifications,
            stats.relative_residual,
            static_cast<int>(stats.cache_hit),
            static_cast<int>(initial != nullptr),
            initial ? initial->residual.norm() : 0.0);
    return result;
}

Eigen::VectorXd
StructureTangentOperator::apply_inverse_shifted_preconditioner(const Eigen::VectorXd& covector,
                                                               double shift) const {
    if (covector.size() != tangent_size() || !covector.allFinite()) {
        throw std::invalid_argument(
            "structure preconditioner covector has incompatible dimensions or values");
    }
    if (!std::isfinite(shift) || shift < 0.0) {
        throw std::invalid_argument(
            "structure preconditioner shift must be finite and nonnegative");
    }

    Eigen::MatrixXd ambient = expand(covector).scaled_coefficients;
    for (int state = 0; state < n_states_; ++state) {
        Eigen::VectorXd denominator =
            absolute_hessian_diagonal_.col(state) + shift * metric_diagonal_;
        const double scale = denominator.maxCoeff();
        if (scale == 0.0) {
            continue;
        }
        const double floor =
            std::max(std::numeric_limits<double>::min(),
                     std::numeric_limits<double>::epsilon() *
                         static_cast<double>(std::max<Eigen::Index>(1, n_structures_)) * scale);
        denominator = denominator.cwiseMax(floor);
        ambient.col(state).array() /= denominator.array();
    }
    return project_coordinates(ambient);
}

Eigen::VectorXd
StructureTangentOperator::apply_inverse_augmented_hessian_diagonal(const Eigen::VectorXd& covector,
                                                                   double eigenvalue) const {
    if (covector.size() != tangent_size() || !covector.allFinite()) {
        throw std::invalid_argument(
            "structure AH preconditioner covector has incompatible dimensions or values");
    }
    if (!std::isfinite(eigenvalue)) {
        throw std::invalid_argument("structure AH preconditioner eigenvalue must be finite");
    }

    if (metric_inverse_ && action_->supports_carrier_diagonal()) {
        // A=M^-1-C(C^T M C)^-1 C^T is the constrained inverse.
        // A F^T D^-1 F A = P_L [M^-1 F^T D^-1 F M^-1] P_R,
        // P_R=I-M C(C^T M C)^-1 C^T and P_L=P_R^T.
        // Use the cached M C; no metric action or inverse solve remains.
        const auto& mc = accepted_point_->selected_state_structure_images->overlap;
        Eigen::MatrixXd rhs = expand(covector).scaled_coefficients;
        Eigen::MatrixXd weights = selected_metric_inverse_ * (selected_.transpose() * rhs);
        rhs.noalias() -= mc * weights;
        Eigen::MatrixXd image = metric_inverse_->apply_carrier_inverse(rhs, energies_, eigenvalue);
        weights = selected_metric_inverse_ * (mc.transpose() * image);
        image.noalias() -= selected_ * weights;
        return project_coordinates(image);
    }

    Eigen::MatrixXd ambient = expand(covector).scaled_coefficients;
    constexpr double kDenominatorFloor = 1.0e-8;
    if (accepted_point_->spin_metric) {
        const auto& constraint = accepted_point_->selected_state_structure_images->overlap;
        Eigen::MatrixXd rhs(n_structures_, n_states_ + 1);
        rhs.rightCols(n_states_) = constraint;
        for (int state = 0; state < n_states_; ++state) {
            const Eigen::VectorXd diagonal =
                (hessian_diagonal_.col(state) - eigenvalue * metric_diagonal_)
                    .cwiseAbs()
                    .cwiseMax(kDenominatorFloor);
            rhs.col(0) = ambient.col(state);
            const Eigen::MatrixXd inverse =
                accepted_point_->spin_metric->solve_diagonal_model(rhs, diagonal);
            const auto inverse_constraint = inverse.rightCols(n_states_);
            const Eigen::MatrixXd gram = constraint.transpose() * inverse_constraint;
            const Eigen::LLT<Eigen::MatrixXd> factor(gram);
            if (factor.info() != Eigen::Success)
                throw std::runtime_error("spin AH constraint model is not positive definite");
            // Invert the restricted model, not merely project its ambient inverse:
            // R_h=R-R A(A^T R A)^-1 A^T R, with A=M C and R=(K G K)^-1.
            ambient.col(state) =
                inverse.col(0) -
                inverse_constraint * factor.solve(constraint.transpose() * inverse.col(0));
        }
        return project_coordinates(ambient);
    }
    for (int state = 0; state < n_states_; ++state) {
        Eigen::VectorXd denominator = hessian_diagonal_.col(state) - eigenvalue * metric_diagonal_;
        for (Eigen::Index i = 0; i < denominator.size(); ++i) {
            if (std::abs(denominator[i]) < kDenominatorFloor) {
                denominator[i] = kDenominatorFloor;
            }
        }
        ambient.col(state).array() /= denominator.array();
    }
    return project_coordinates(ambient);
}

double StructureTangentOperator::squared_norm(const StructureTangent& tangent) const {
    const Eigen::VectorXd horizontal = coordinates(tangent);
    const double norm = metric_inner_product(horizontal, horizontal);
    const double tolerance =
        1.0e3 * std::numeric_limits<double>::epsilon() * std::max(1.0, horizontal.squaredNorm());
    if (!std::isfinite(norm) || norm < -tolerance) {
        throw std::runtime_error("structure tangent metric is not positive semidefinite");
    }
    return std::max(0.0, norm);
}

} // namespace xmvb::vb
