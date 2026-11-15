#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>

#include "core/index.hpp"
#include "mcscf/ci/sigma.hpp"
#include "mcscf/ci/workspace.hpp"
#include "mcscf/optimization/trace.hpp"
#include "parallel/openmp.hpp"
#include "vbscf/structures/assembly/input_columns.hpp"

namespace xmvb::vb {

StructureActiveIntegralAdjoint
StructureAction::recoupled_state_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& lanes) const {
    if (lanes.rows() != n_structures_ || lanes.cols() != 2)
        throw std::invalid_argument("recoupled adjoint requires state and residual lanes");
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto view = orthogonal_view();
    const auto na = n_unique_alpha_, nb = n_unique_beta_;
    const auto column_bytes = sizeof(double) * static_cast<std::size_t>(na);
    const auto budget = workspace::available() / 4;
    const auto extent = std::max<Eigen::Index>(
        1,
        std::min<Eigen::Index>(
            nb,
            std::min<std::size_t>(8 * cooperative_openmp_thread_count(),
                                  budget / std::max<std::size_t>(1, 4 * column_bytes))));
    const mcscf::CiWorkspace operation;
    operation.sigma_budget(budget);
    auto pattern = std::make_shared<detail::ColumnPattern>();
    pattern->full = true;
    pattern->offsets.resize(2);
    pattern->offsets << 0, na;
    const std::vector<std::uint8_t> support(nb, 1);
    const bool dense =
        topology_->products.size() ==
        core::checked_multiply(static_cast<std::size_t>(na), static_cast<std::size_t>(nb));
    const auto price = [&](Eigen::Index) { return column_bytes + pattern->bytes(); };
    const auto build = [&](int lane) -> detail::InputColumns::Build {
        return [&, lane](std::span<const Eigen::Index> keys) {
            std::vector<detail::ColumnTile> tiles;
            tiles.reserve(keys.size());
            for (std::size_t j = 0; j < keys.size(); ++j)
                tiles.emplace_back(pattern, Eigen::MatrixXd(na, 1));
            cooperative_openmp_for(
                na * keys.size(), cooperative_openmp_thread_count(), [&](std::ptrdiff_t item) {
                    const auto a = item % na, j = item / na;
                    const auto product = a * nb + keys[j];
                    const auto found = dense ? topology_->products.begin() + product
                                             : std::lower_bound(topology_->products.begin(),
                                                                topology_->products.end(),
                                                                product);
                    double value = 0;
                    if (found != topology_->products.end() && *found == product) {
                        const auto g = found - topology_->products.begin();
                        for (auto k = topology_->offsets[g]; k < topology_->offsets[g + 1]; ++k) {
                            const auto& term = topology_->terms[k];
                            value += term.coefficient * lanes(term.structure_index, lane);
                        }
                    }
                    tiles[j].values(a, 0) = value;
                });
            return tiles;
        };
    };
    detail::InputColumns state(na, 1, extent, budget, support, build(0), price);
    detail::InputColumns residual(na, 1, extent, budget, support, build(1), price);
    const auto reader = [](detail::InputColumns& columns) -> mcscf::DirectCiTileReader {
        return [&columns](const auto& betas,
                          const auto& rows,
                          Eigen::Ref<Eigen::MatrixXd> values,
                          mcscf::DirectCiRead mode) {
            columns.read(betas, rows, values, mode == mcscf::DirectCiRead::Dense);
        };
    };
    const auto y = reader(state), r = reader(residual);
    // No sigma writer: H*x is already known from the retained Ritz images.
    const auto integrals = view.sigma.apply_tiled_with_adjoint(
        1, y, {}, {}, nullptr, view.exchange_parity, {}, budget);
    const auto density_end = Clock::now();
    const mcscf::DirectCiColumnReader r_column = [&](Eigen::Index b) {
        const auto& tile = residual.column(b);
        return mcscf::DirectCiColumnView{tile.values.data(), nullptr, na};
    };
    const auto n = view.sigma.n_orbitals();
    Eigen::MatrixXd generator = Eigen::MatrixXd::Zero(n, n);
    std::vector<Eigen::Index> rows(na);
    std::iota(rows.begin(), rows.end(), 0);
    for (Eigen::Index first = 0; first < nb; first += extent) {
        const auto count = std::min<Eigen::Index>(extent, nb - first);
        std::vector<Eigen::Index> betas(count);
        std::iota(betas.begin(), betas.end(), first);
        workspace::Temporary storage(column_bytes * count);
        Eigen::MatrixXd panel(na, count);
        y(betas, rows, panel, mcscf::DirectCiRead::Dense);
        generator.noalias() +=
            2 * view.sigma.generator_adjoint_streamed(
                    first, panel, {{r, r_column, 1.0, true}}, false, view.exchange_parity != 0);
    }
    mcscf::diagnostics::ciah_trace(
        "ciah_ci_reused_adjoint",
        "density=%.9f,generator=%.9f,state_bytes=%zu,residual_bytes=%zu",
        std::chrono::duration<double>(density_end - start).count(),
        std::chrono::duration<double>(Clock::now() - density_end).count(),
        state.cache().peak_bytes,
        residual.cache().peak_bytes);
    StructureAdjointState gauge;
    gauge.weights = {1.0};
    return finish_response(
        gauge,
        {{}, integrals.one_electron, integrals.pair_kernel, std::move(generator)},
        Eigen::MatrixXd::Zero(1, 1));
}

} // namespace xmvb::vb
