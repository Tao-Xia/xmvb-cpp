#include <cstdint>
#include "vbscf/structures/assembly/operator_first_internal.hpp"

#include "core/workspace.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::vb {
namespace {
/** @brief A shared primal panel and independent derivative lanes. */
struct Panel {
    Eigen::MatrixXd value;
    std::vector<Eigen::MatrixXd> directions;
    Panel(int rows, int cols, int lanes)
        : value(Eigen::MatrixXd::Zero(rows, cols)), directions(lanes, value) {}
};
/** @brief Apply the single-spin Gram map once, differentiating only the moving frame. */
void gram(Panel& panel,
          const ExteriorOrbitalTransform& transform,
          const ExteriorTransformDirection* moving,
          bool frame_only = false) {
    if (moving) {
        transform.apply_directional_left(*moving, &panel.value, &panel.directions.front());
        transform.apply_directional_adjoint_left(*moving, &panel.value, &panel.directions.front());
    } else {
        transform.apply_left(&panel.value);
        transform.apply_adjoint_left(&panel.value);
    }
    for (std::size_t l = moving ? 1 : 0; !frame_only && l < panel.directions.size(); ++l) {
        transform.apply_left(&panel.directions[l]);
        transform.apply_adjoint_left(&panel.directions[l]);
    }
}
struct Lane {
    Eigen::MatrixXd source, one, two, moment, h_image, m_image;
    StructureResponseSweep result;
    bool images = false;
};
} // namespace

StructureResponseSweep OperatorFirstAction::jet(const StructureAdjointState& state,
                                                const mcscf::ActiveIntegrals* integral_direction,
                                                const Eigen::MatrixXd* orbital_direction,
                                                const Eigen::MatrixXd* coefficient_direction,
                                                bool collect_images,
                                                std::size_t workspace_bytes,
                                                std::vector<Eigen::MatrixXd>* densities,
                                                OperatorFirstState* accepted,
                                                StructureResponseRequest* ci_response) const {
    const auto& data = *data_;
    const auto& view = data.view;
    const std::int64_t na = view.alpha.dimension(), nb = view.beta.dimension();
    const int n = view.alpha.n_orbitals(), pairs = n * (n + 1) / 2, states = state.weights.size();
    if (states <= 0 || state.structure_coefficients.rows() != data.n_structures ||
        state.structure_coefficients.cols() != states || state.energies.size() != states)
        throw std::invalid_argument("invalid operator-first jet state");
    const bool tangent = integral_direction || orbital_direction || coefficient_direction;
    if (ci_response &&
        (!tangent || ci_response->coefficients.rows() != data.n_structures ||
         ci_response->coefficients.cols() != states || !ci_response->coefficients.allFinite()))
        throw std::invalid_argument("invalid joint operator-first CI direction");
    const bool moving = orbital_direction != nullptr;
    const auto* fixed = state.operator_state.get();
    const bool reuse = tangent && fixed && fixed->point == data.point;
    const bool collect_value = !tangent || (moving && !reuse);
    const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(n, n);
    const Eigen::MatrixXd& dr = moving ? *orbital_direction : zero;
    mcscf::ActiveIntegrals empty;
    empty.one_electron = zero;
    empty.pair_kernel = Eigen::MatrixXd::Zero(pairs, pairs);
    const auto dh = integral_direction || moving
                        ? std::optional<mcscf::SimilarityHamiltonian>(data.hamiltonian.direction(
                              integral_direction ? *integral_direction : empty, dr))
                        : std::nullopt;
    const auto da = moving ? std::optional(view.alpha.direction(dr)) : std::nullopt;
    const auto db = moving ? std::optional(view.beta.direction(dr)) : std::nullopt;
    const int lanes = (tangent ? 1 : 0) + (ci_response ? 1 : 0);
    const int outputs = (collect_images ? 1 : 0) + (ci_response ? 1 : 0);
    const auto targets = data.source.beta_columns();
    const std::size_t per_column =
        sizeof(double) * (6 + 5 * lanes) * (static_cast<std::size_t>(na) + nb);
    const std::size_t small = static_cast<std::size_t>(n) * n;
    const std::size_t adjoints =
        sizeof(double) *
        (small * small * (24 + 4 * lanes) + small * (32 + 4 * lanes) +
         (1 + lanes) * cooperative_openmp_thread_count() * (small * small + small));
    const std::size_t intrinsic =
        sizeof(double) * view.products.size() * states *
        (1 + (coefficient_direction ? 1 : 0) + (ci_response ? 1 : 0) + 2 * outputs);
    if (workspace_bytes <= intrinsic || workspace_bytes - intrinsic < per_column)
        throw std::length_error("operator-first jet budget cannot hold one panel");
    const int width =
        std::min<std::size_t>(targets.size(),
                              std::min<std::size_t>(8 * cooperative_openmp_thread_count(),
                                                    (workspace_bytes - intrinsic) / per_column));
    const workspace::Temporary charge(adjoints + intrinsic + width * per_column);
    const auto result = [&](bool images) {
        return StructureResponseSweep{
            {Eigen::MatrixXd::Zero(images ? data.n_structures : 0, states),
             Eigen::MatrixXd::Zero(images ? data.n_structures : 0, states)},
            zero,
            Eigen::MatrixXd::Zero(pairs, pairs),
            zero};
    };
    std::vector<Lane> directions(lanes);
    for (int l = 0; l < lanes; ++l) {
        auto& lane = directions[l];
        lane.images = l == 0 ? collect_images : true;
        lane.result = result(lane.images);
        lane.one = zero;
        lane.two = Eigen::MatrixXd::Zero(n * n, n * n);
        lane.moment = zero;
        lane.h_image = Eigen::MatrixXd::Zero(lane.images ? view.products.size() : 0, states);
        lane.m_image = lane.h_image;
        if ((l == 0 && coefficient_direction) || l > 0)
            lane.source = Eigen::MatrixXd::Zero(view.products.size(), states);
    }
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(view.products.size(), states);
#pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < view.products.size(); ++p)
        for (auto t = view.offsets[p]; t < view.offsets[p + 1]; ++t) {
            const auto& term = view.terms[t];
            x.row(p).noalias() +=
                term.coefficient * state.structure_coefficients.row(term.structure);
            if (coefficient_direction)
                directions[0].source.row(p).noalias() +=
                    term.coefficient * coefficient_direction->row(term.structure);
            if (ci_response)
                directions.back().source.row(p).noalias() +=
                    term.coefficient * ci_response->coefficients.row(term.structure);
        }
    StructureResponseSweep out = result(collect_images);
    Eigen::MatrixXd h_image =
        Eigen::MatrixXd::Zero(!tangent && collect_images ? view.products.size() : 0, states);
    Eigen::MatrixXd m_image = h_image;
    if (densities)
        densities->assign(states * (states + 1) / 2, zero);
    Eigen::MatrixXd raw_h = zero, raw_g = Eigen::MatrixXd::Zero(n * n, n * n), raw_moment = zero;
    if (reuse && moving) {
        raw_h = fixed->one;
        raw_g = fixed->two;
        raw_moment = fixed->moment;
    }
    for (std::size_t first = 0; first < targets.size(); first += width) {
        const std::int64_t count = std::min<std::size_t>(width, targets.size() - first);
        const auto betas = targets.subspan(first, count);
        Panel right(nb, count, lanes);
        for (int j = 0; j < count; ++j)
            right.value(betas[j], j) = 1;
        gram(right, view.beta, db ? &*db : nullptr, true);
        for (int k = 0; k < states; ++k) {
            const double w = state.weights[k], energy = state.energies[k];
            Panel z(na, count, lanes), source(na, count, lanes);
            data.source.source_columns(x.middleCols(k, 1), betas, source.value);
            for (int l = 0; l < lanes; ++l)
                if (directions[l].source.size())
                    data.source.source_columns(
                        directions[l].source.middleCols(k, 1), betas, source.directions[l]);
            Panel left_seed(na, count, lanes), beta_seed(nb, count, lanes);
            data.source.multiply_source(x.col(k), right.value, left_seed.value);
            Eigen::MatrixXd temp(na, count);
            for (int l = 0; l < lanes; ++l) {
                if (directions[l].source.size())
                    data.source.multiply_source(
                        directions[l].source.col(k), right.value, left_seed.directions[l]);
                if (l == 0 && moving) {
                    data.source.multiply_source(x.col(k), right.directions[l], temp);
                    left_seed.directions[l] += temp;
                }
                left_seed.directions[l] *= w;
            }
            left_seed.value *= w;
            {
                Panel canonical = left_seed;
                gram(canonical, view.alpha, da ? &*da : nullptr);
                const mcscf::RawCiAdjoint adjoint{
                    canonical.value,
                    {collect_value ? &raw_h : nullptr, collect_value ? &raw_g : nullptr}};
                std::vector<mcscf::RawCiDirection> raw;
                raw.reserve(lanes);
                for (int l = 0; l < lanes; ++l)
                    raw.push_back({l == 0 && dh ? &*dh : nullptr,
                                   directions[l].source.size() ? &directions[l].source : nullptr,
                                   k,
                                   z.directions[l],
                                   canonical.directions[l],
                                   {&directions[l].one, &directions[l].two}});
                data.source.apply_columns(
                    data.hamiltonian, x.middleCols(k, 1), betas, z.value, &adjoint, raw);
                if (densities && w != 0)
                    for (int j = k; j < states; ++j)
                        if (state.weights[j] != 0)
                            (*densities)[j * (j + 1) / 2 + k] +=
                                data.source.one_body_moment(
                                    x.middleCols(j, 1), betas, canonical.value) /
                                w;
            }
            z.value -= energy * source.value;
            for (int l = 0; l < lanes; ++l)
                z.directions[l] -= energy * source.directions[l];
            gram(z, view.alpha, da ? &*da : nullptr);
            if (outputs)
                gram(source, view.alpha, da ? &*da : nullptr);
            data.source.multiply_source(x.col(k), z.value, beta_seed.value, true);
            temp.resize(nb, count);
            for (int l = 0; l < lanes; ++l) {
                if (directions[l].source.size())
                    data.source.multiply_source(
                        directions[l].source.col(k), z.value, beta_seed.directions[l], true);
                data.source.multiply_source(x.col(k), z.directions[l], temp, true);
                beta_seed.directions[l] += temp;
                beta_seed.directions[l] *= w;
            }
            beta_seed.value *= w;
            if (outputs) {
#pragma omp parallel for schedule(static)
                for (std::size_t p = 0; p < view.products.size(); ++p) {
                    const std::int64_t a = view.products[p] / nb, b = view.products[p] % nb;
                    if (!tangent && collect_images) {
                        const double m = source.value.row(a).dot(right.value.row(b));
                        h_image(p, k) += z.value.row(a).dot(right.value.row(b)) + energy * m;
                        m_image(p, k) += m;
                    }
                    for (int l = 0; l < lanes; ++l) {
                        auto& lane = directions[l];
                        if (!lane.images)
                            continue;
                        double dm = source.directions[l].row(a).dot(right.value.row(b));
                        double dhv = z.directions[l].row(a).dot(right.value.row(b));
                        if (l == 0 && moving) {
                            dm += source.value.row(a).dot(right.directions[l].row(b));
                            dhv += z.value.row(a).dot(right.directions[l].row(b));
                        }
                        lane.h_image(p, k) += dhv + energy * dm;
                        lane.m_image(p, k) += dm;
                    }
                }
            }
            if (collect_value)
                raw_moment += view.sigma.spin_generator_adjoint(false, z.value, left_seed.value) +
                              view.sigma.spin_generator_adjoint(true, right.value, beta_seed.value);
            for (int l = 0; l < lanes; ++l) {
                directions[l].moment +=
                    view.sigma.spin_generator_adjoint(false, z.directions[l], left_seed.value) +
                    view.sigma.spin_generator_adjoint(false, z.value, left_seed.directions[l]) +
                    view.sigma.spin_generator_adjoint(true, right.value, beta_seed.directions[l]);
                if (l == 0 && moving)
                    directions[l].moment += view.sigma.spin_generator_adjoint(
                        true, right.directions[l], beta_seed.value);
            }
        }
    }
    if (accepted) {
        accepted->point = data.point;
        accepted->one = raw_h;
        accepted->two = raw_g;
        accepted->moment = raw_moment;
    }
    const auto& r = view.alpha.orbital_transform();
    const Eigen::MatrixXd inverse =
        r.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(n, n));
    if (densities)
        for (auto& density : *densities)
            density = inverse.transpose() * density * r.transpose();
    const auto finish = [&](StructureResponseSweep& sweep,
                            const Eigen::MatrixXd& h,
                            const Eigen::MatrixXd& m,
                            bool images) {
        auto& tiles = sweep.images.tiles;
        tiles.expansions = states * (1 + (coefficient_direction ? 1 : 0) + (ci_response ? 1 : 0));
        tiles.structure_projections = images ? 2 * states : 0;
        tiles.source_tiles = states * ((targets.size() + width - 1) / width);
        tiles.source_columns = states * targets.size();
        tiles.input_panels = tiles.integral_adjoint_sweeps = tiles.source_tiles;
        tiles.hamiltonian_projections = tiles.overlap_projections = images ? states : 0;
        tiles.source_workspace_bound = intrinsic + width * per_column;
        if (!images)
            return;
#pragma omp parallel for schedule(static)
        for (Eigen::Index s = 0; s < data.n_structures; ++s)
            for (auto t = view.structure_offsets[s]; t < view.structure_offsets[s + 1]; ++t) {
                const auto& term = view.terms[view.structure_terms[t]];
                sweep.images.hamiltonian.row(s).noalias() += term.coefficient * h.row(term.group);
                sweep.images.overlap.row(s).noalias() += term.coefficient * m.row(term.group);
            }
    };
    if (!tangent) {
        const auto adj = data.hamiltonian.pullback(raw_h, raw_g);
        out.one_electron = adj.one_electron;
        out.pair_kernel = adj.pair_kernel;
        out.generator = 2 * inverse.transpose() * raw_moment * r.transpose();
        finish(out, h_image, m_image, collect_images);
    }
    for (int l = 0; l < lanes; ++l) {
        auto& lane = directions[l];
        const Eigen::MatrixXd& delta = l == 0 ? dr : zero;
        const auto adj = data.hamiltonian.pullback(raw_h, raw_g, &delta, &lane.one, &lane.two);
        lane.result.one_electron = adj.one_electron;
        lane.result.pair_kernel = adj.pair_kernel;
        lane.result.generator = 2 * inverse.transpose() * lane.moment * r.transpose();
        if (l == 0 && moving) {
            const Eigen::MatrixXd dinverse = -inverse * dr * inverse;
            lane.result.generator += 2 * (dinverse.transpose() * raw_moment * r.transpose() +
                                          inverse.transpose() * raw_moment * dr.transpose());
        }
        finish(lane.result, lane.h_image, lane.m_image, lane.images);
    }
    if (ci_response)
        ci_response->result = std::move(directions.back().result);
    return tangent ? std::move(directions.front().result) : std::move(out);
}
} // namespace xmvb::vb
