#include "vbscf/structures/orthogonal_ci/metric.hpp"

#include "mcscf/ci/string_table.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

#include "mcscf/optimization/trace.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {
namespace {

std::size_t carrier_column_bytes(Eigen::Index structures, std::size_t alpha, std::size_t beta) {
    return core::checked_multiply(
        sizeof(double),
        core::checked_add(core::checked_multiply(alpha, beta),
                          core::checked_multiply(4, core::checked_index<std::size_t>(structures))));
}

} // namespace

bool StructureMetricInverse::fits(const StructureAction& action,
                                  const xmvb::mcscf::StringTable& alpha_strings,
                                  const xmvb::mcscf::StringTable& beta_strings,
                                  std::size_t factor_byte_limit,
                                  std::size_t workspace_byte_limit) {
    if (!action.storage().orthogonal_direct_ci)
        return false;
    const auto column_bytes =
        carrier_column_bytes(action.n_structures_, alpha_strings.size(), beta_strings.size());
    if (column_bytes == 0 || column_bytes > workspace_byte_limit)
        return false;
    return SpinMetric::fits(action, alpha_strings, beta_strings, factor_byte_limit);
}

std::unique_ptr<StructureMetricInverse>
StructureMetricInverse::try_build(const StructureAction& action,
                                  const xmvb::mcscf::StringTable& alpha_strings,
                                  const xmvb::mcscf::StringTable& beta_strings,
                                  std::size_t other_work_bytes,
                                  const StructureMetricInverse* accepted) {
    const auto column =
        carrier_column_bytes(action.n_structures_, alpha_strings.size(), beta_strings.size());
    // The Hamiltonian application holds one input and one output carrier.
    const auto scratch = core::checked_multiply(2, column);
    const auto available = workspace::available();
    if (other_work_bytes >= available || scratch >= available - other_work_bytes)
        return {};
    const auto factors = available - other_work_bytes - scratch;
    std::shared_ptr<const SpinMetric> spin;
    if (accepted && action.topology_ == accepted->action_->topology_)
        spin = accepted->spin_;
    if (!spin && !fits(action, alpha_strings, beta_strings, factors, scratch))
        return {};
    mcscf::diagnostics::ciah_trace("ciah_ci_spin_factors", "shared=%d", spin ? 1 : 0);
    if (!spin) {
        spin = std::make_shared<SpinMetric>(action, alpha_strings, beta_strings, factors);
        mcscf::diagnostics::ciah_trace("ciah_ci_spin_layout",
                                     "blocks=%zu,factors=%zu,contiguous=%zu,factor_bytes=%zu",
                                     spin->block_count(),
                                     spin->factor_count(),
                                     spin->contiguous_block_count(),
                                     spin->factor_bytes());
    }
    return std::unique_ptr<StructureMetricInverse>(
        new StructureMetricInverse(action, std::move(spin), true, column, scratch));
}

StructureMetricInverse::StructureMetricInverse(const StructureAction& action,
                                               const xmvb::mcscf::StringTable& alpha_strings,
                                               const xmvb::mcscf::StringTable& beta_strings,
                                               bool complete_space,
                                               std::size_t factor_byte_limit,
                                               std::size_t workspace_byte_limit)
    : StructureMetricInverse(
          action,
          std::make_shared<SpinMetric>(action, alpha_strings, beta_strings, factor_byte_limit),
          complete_space,
          carrier_column_bytes(action.n_structures_, alpha_strings.size(), beta_strings.size()),
          workspace_byte_limit) {}

StructureMetricInverse::StructureMetricInverse(const StructureAction& action,
                                               std::shared_ptr<const SpinMetric> spin,
                                               bool complete_space,
                                               std::size_t column_bytes,
                                               std::size_t workspace_byte_limit)
    : action_(&action), spin_(std::move(spin)) {
    if (!complete_space || !action.storage().orthogonal_direct_ci)
        throw std::invalid_argument("factorized metric inverse requires a complete spin sector");
    if (column_bytes == 0 || workspace_byte_limit < column_bytes)
        throw std::length_error("structure metric workspace cannot hold one carrier column");
    chunk_width_ = core::checked_index<Eigen::Index>(
        std::min<std::size_t>(workspace_byte_limit / column_bytes,
                              std::numeric_limits<Eigen::Index>::max() / action.n_unique_beta_));
    if (!carrier_workspace_.try_resize(core::checked_multiply(column_bytes, chunk_width_)))
        throw std::length_error("structure metric carrier exceeds shared workspace");
}

Eigen::MatrixXd StructureMetricInverse::apply(const Eigen::Ref<const Eigen::MatrixXd>& rhs) const {
    if (rhs.rows() != action_->n_structures_ || rhs.cols() <= 0 || !rhs.allFinite())
        throw std::invalid_argument("structure metric inverse RHS is invalid");
    Eigen::MatrixXd result(rhs.rows(), rhs.cols());
    for (Eigen::Index first = 0; first < rhs.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, rhs.cols() - first);
        Eigen::MatrixXd carrier =
            action_->expand_structure_block(spin_->solve(rhs.middleCols(first, count)));
        action_->inverse_adjoint_orbital_transform(&carrier);
        action_->inverse_orbital_transform(&carrier);
        result.middleCols(first, count) =
            spin_->solve(action_->contract_spin_product_block(carrier));
    }
    if (!result.allFinite())
        throw std::runtime_error("structure metric inverse produced nonfinite coefficients");
    return result;
}

Eigen::MatrixXd
StructureMetricInverse::apply_carrier_inverse(const Eigen::Ref<const Eigen::MatrixXd>& rhs,
                                              const Eigen::Ref<const Eigen::VectorXd>& energies,
                                              double eigenvalue) const {
    if (rhs.rows() != action_->n_structures_ || rhs.cols() <= 0 || rhs.cols() != energies.size() ||
        !rhs.allFinite())
        throw std::invalid_argument("invalid complete-sector carrier inverse RHS");
    Eigen::MatrixXd result(rhs.rows(), rhs.cols());
    for (Eigen::Index first = 0; first < rhs.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, rhs.cols() - first);
        Eigen::MatrixXd carrier =
            action_->expand_structure_block(spin_->solve(rhs.middleCols(first, count)));
        action_->inverse_adjoint_orbital_transform(&carrier);
        action_->divide_carrier_diagonal(carrier, energies.segment(first, count), eigenvalue);
        action_->inverse_orbital_transform(&carrier);
        result.middleCols(first, count) =
            spin_->solve(action_->contract_spin_product_block(carrier));
    }
    return result;
}

Eigen::MatrixXd StructureMetricInverse::to_ci_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients) const {
    Eigen::MatrixXd result(coefficients.rows(), coefficients.cols());
    for (Eigen::Index first = 0; first < coefficients.cols(); first += chunk_width_) {
        const Eigen::Index count =
            std::min<Eigen::Index>(chunk_width_, coefficients.cols() - first);
        const auto carrier =
            action_->orthogonalize_structure_block(coefficients.middleCols(first, count));
        result.middleCols(first, count) =
            spin_->solve(action_->contract_spin_product_block(carrier), SpinMetric::Part::Lower);
    }
    return result;
}

Eigen::MatrixXd StructureMetricInverse::from_ci_coordinates(
    const Eigen::Ref<const Eigen::MatrixXd>& coordinates) const {
    Eigen::MatrixXd result(coordinates.rows(), coordinates.cols());
    for (Eigen::Index first = 0; first < coordinates.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, coordinates.cols() - first);
        auto carrier = action_->expand_structure_block(
            spin_->solve(coordinates.middleCols(first, count), SpinMetric::Part::Upper));
        action_->inverse_orbital_transform(&carrier);
        result.middleCols(first, count) =
            spin_->solve(action_->contract_spin_product_block(carrier));
    }
    return result;
}

Eigen::MatrixXd StructureMetricInverse::ci_covector_to_structure(
    const Eigen::Ref<const Eigen::MatrixXd>& covector) const {
    Eigen::MatrixXd result(covector.rows(), covector.cols());
    for (Eigen::Index first = 0; first < covector.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, covector.cols() - first);
        auto carrier = action_->expand_structure_block(
            spin_->solve(covector.middleCols(first, count), SpinMetric::Part::Upper));
        action_->adjoint_orbital_transform(&carrier);
        result.middleCols(first, count) = action_->contract_spin_product_block(carrier);
    }
    return result;
}

Eigen::MatrixXd StructureMetricInverse::apply_ci_hamiltonian(
    const Eigen::Ref<const Eigen::MatrixXd>& coordinates) const {
    if (coordinates.cols() == 0)
        return Eigen::MatrixXd(coordinates.rows(), 0);
    // Each lane holds both an input and an output carrier. The metric owns
    // the minimum scratch for this action; lease extra lanes only while
    // Davidson actually has them, leaving working space for the sigma tiles.
    const auto lane_bytes = core::checked_multiply(2,
                                                   carrier_column_bytes(action_->n_structures_,
                                                                        action_->n_unique_alpha_,
                                                                        action_->n_unique_beta_));
    const auto minimum_tile = core::checked_multiply(
        2 * sizeof(double),
        core::checked_index<std::size_t>(action_->orthogonal_integrals().pair_kernel.rows()));
    const auto spare = std::max(lane_bytes, minimum_tile);
    const auto free = workspace::available();
    const auto extra_capacity = (free - std::min(free, spare)) / lane_bytes;
    const Eigen::Index resident = std::max<Eigen::Index>(1, chunk_width_ / 2);
    const Eigen::Index requested = std::min(
        coordinates.cols(), std::numeric_limits<Eigen::Index>::max() / action_->n_unique_beta_);
    const auto extra_lanes = std::min(
        extra_capacity, static_cast<std::size_t>(std::max<Eigen::Index>(0, requested - resident)));
    workspace::Reservation extra;
    const Eigen::Index admitted =
        extra.try_resize(extra_lanes * lane_bytes) ? static_cast<Eigen::Index>(extra_lanes) : 0;
    const Eigen::Index width = std::min(requested, resident + admitted);
    Eigen::MatrixXd result;
    if (width < coordinates.cols())
        result.resize(coordinates.rows(), coordinates.cols());
    mcscf::diagnostics::ciah_trace("ciah_ci_sigma_batch",
                                   "rhs=%ld,width=%ld,extra_bytes=%zu",
                                   static_cast<long>(coordinates.cols()),
                                   static_cast<long>(width),
                                   extra.bytes());
    using Clock = std::chrono::steady_clock;
    const bool trace = mcscf::diagnostics::ciah_event_trace_enabled();
    const auto stamp = [&] { return trace ? Clock::now() : Clock::time_point{}; };
    Clock::duration upper_time{}, lower_time{};
    for (Eigen::Index first = 0; first < coordinates.cols(); first += width) {
        const Eigen::Index count = std::min<Eigen::Index>(width, coordinates.cols() - first);
        Eigen::MatrixXd image;
        {
            const auto start = stamp();
            const auto coefficients =
                spin_->solve(coordinates.middleCols(first, count), SpinMetric::Part::Upper);
            upper_time += stamp() - start;
            image = action_->apply_recoupled_hamiltonian(coefficients);
        }
        const auto start = stamp();
        spin_->solve_in_place(image, SpinMetric::Part::Lower);
        lower_time += stamp() - start;
        if (count == coordinates.cols())
            result = std::move(image);
        else
            result.middleCols(first, count) = image;
    }
    if (trace)
        mcscf::diagnostics::ciah_trace("ciah_ci_spin_solve_cost",
                                     "upper=%.9f,lower=%.9f",
                                     std::chrono::duration<double>(upper_time).count(),
                                     std::chrono::duration<double>(lower_time).count());
    return result;
}

Eigen::VectorXd StructureMetricInverse::ci_occupation_diagonal() const {
    return spin_->occupation_diagonal(action_->orthogonal_integrals());
}

Eigen::MatrixXd
StructureMetricInverse::apply_ci_preconditioner(const Eigen::Ref<const Eigen::MatrixXd>& covector,
                                                const Eigen::Ref<const Eigen::VectorXd>& energies,
                                                double eigenvalue) const {
    Eigen::MatrixXd result(covector.rows(), covector.cols());
    for (Eigen::Index first = 0; first < covector.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, covector.cols() - first);
        auto carrier = action_->expand_structure_block(
            spin_->solve(covector.middleCols(first, count), SpinMetric::Part::Upper));
        action_->divide_carrier_diagonal(carrier, energies.segment(first, count), eigenvalue);
        result.middleCols(first, count) =
            spin_->solve(action_->contract_spin_product_block(carrier), SpinMetric::Part::Lower);
    }
    return result;
}

Eigen::MatrixXd StructureMetricInverse::transport_coefficients(
    const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
    const ExteriorOrbitalTransform& alpha,
    const ExteriorOrbitalTransform& beta) const {
    if (coefficients.rows() != action_->n_structures_ || coefficients.cols() <= 0 ||
        !coefficients.allFinite())
        throw std::invalid_argument("invalid complete-sector transport coefficients");
    Eigen::MatrixXd result(coefficients.rows(), coefficients.cols());
    for (Eigen::Index first = 0; first < coefficients.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, coefficients.cols() - first);
        Eigen::MatrixXd carrier = action_->expand_structure_block(coefficients.middleCols(first, count));
        alpha.apply_left(&carrier);
        for (Eigen::Index block = 0; block < count; ++block)
            beta.apply_right_block(carrier.middleCols(block * beta.dimension(), beta.dimension()));
        result.middleCols(first, count) = spin_->solve(action_->contract_spin_product_block(carrier));
    }
    return result;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> StructureMetricInverse::transport_pullback(
    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
    const Eigen::Ref<const Eigen::VectorXd>& covector,
    const ExteriorOrbitalTransform& alpha,
    const ExteriorOrbitalTransform& beta) const {
    if (!supports_transport_pullback() || coefficients.size() != action_->n_structures_ ||
        covector.size() != coefficients.size())
        throw std::invalid_argument("invalid complete-sector transport pullback");
    // Lift with W G^-1, the adjoint of the closed-sector coefficient projection.
    Eigen::MatrixXd left = action_->expand_structure_block(spin_->solve(covector));
    Eigen::MatrixXd generator = action_->carrier_generator_adjoint(
        left, action_->expand_structure_block(coefficients));
    alpha.apply_adjoint_left(&left);
    beta.apply_adjoint_right(&left);
    Eigen::VectorXd base = action_->contract_spin_product_block(left).col(0);
    return {std::move(generator), std::move(base)};
}

Eigen::MatrixXd
StructureMetricInverse::apply_at_transform(const Eigen::Ref<const Eigen::MatrixXd>& rhs,
                                           const ExteriorOrbitalTransform& alpha,
                                           const ExteriorOrbitalTransform& beta) const {
    if (rhs.rows() != action_->n_structures_ || rhs.cols() <= 0 || !rhs.allFinite())
        throw std::invalid_argument("trial metric inverse RHS is invalid");
    Eigen::MatrixXd result(rhs.rows(), rhs.cols());
    for (Eigen::Index first = 0; first < rhs.cols(); first += chunk_width_) {
        const Eigen::Index count = std::min<Eigen::Index>(chunk_width_, rhs.cols() - first);
        Eigen::MatrixXd carrier =
            action_->expand_structure_block(spin_->solve(rhs.middleCols(first, count)));
        alpha.apply_inverse_adjoint_left(&carrier);
        for (Eigen::Index block = 0; block < count; ++block)
            beta.apply_inverse_adjoint_right_block(
                carrier.middleCols(block * beta.dimension(), beta.dimension()));
        alpha.apply_inverse_left(&carrier);
        for (Eigen::Index block = 0; block < count; ++block)
            beta.apply_inverse_right_block(
                carrier.middleCols(block * beta.dimension(), beta.dimension()));
        result.middleCols(first, count) =
            spin_->solve(action_->contract_spin_product_block(carrier));
    }
    return result;
}

void StructureMetricInverse::rebind(const StructureAction& action) {
    if (action.n_structures_ != action_->n_structures_ ||
        action.n_unique_alpha_ != action_->n_unique_alpha_ ||
        action.n_unique_beta_ != action_->n_unique_beta_)
        throw std::invalid_argument("moved structure metric action has different dimensions");
    action_ = &action;
}

} // namespace xmvb::vb
