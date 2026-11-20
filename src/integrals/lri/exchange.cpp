#include "integrals/lri/exchange.hpp"

#include <RI/ri/LRI.h>
#include <climits>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>

namespace xmvb::integrals::lri {
namespace {
using Engine = RI::LRI<std::size_t, std::int64_t, 3, double>;
using Tensor = RI::Tensor<double>;
using Blocks = std::map<std::size_t, std::map<Engine::TAC, Tensor>>;
constexpr Engine::TC origin{};
const std::vector<RI::Label::ab_ab> channels{RI::Label::ab_ab::a0b0_a1b1,
                                             RI::Label::ab_ab::a0b0_a1b2,
                                             RI::Label::ab_ab::a0b0_a2b1,
                                             RI::Label::ab_ab::a0b0_a2b2};

void require(bool valid, const char* message) {
    if (!valid)
        throw std::invalid_argument(message);
}

// LibRI's BLAS interface uses LP64 dimensions, even though its tensor indices
// are size_t. Check flattened local panels before allocating or narrowing.
std::size_t panel_size(std::initializer_list<std::size_t> dimensions) {
    std::size_t size = 1;
    for (auto d : dimensions) {
        require(d > 0 && d <= static_cast<std::size_t>(INT_MAX) / size,
                "local-RI panel exceeds LP64 BLAS range");
        size *= d;
    }
    return size;
}

void matrix_shape(const Eigen::MatrixXd& matrix, std::size_t rows, std::size_t cols) {
    require(matrix.rows() == static_cast<Eigen::Index>(rows) &&
                matrix.cols() == static_cast<Eigen::Index>(cols) && matrix.allFinite(),
            "invalid local-RI matrix shape or values");
}
} // namespace

struct Exchange::Impl {
    Engine engine;
    std::vector<std::size_t> offsets{0};
    Blocks density;

    Impl(const std::vector<AtomBasis>& atoms,
         const std::vector<PairFit>& fits,
         const std::vector<CoulombBlock>& coulomb) {
        int initialized = 0, finalized = 0, thread_level = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        require(initialized && !finalized, "local-RI requires active MPI");
        MPI_Query_thread(&thread_level);
        require(thread_level >= MPI_THREAD_FUNNELED, "local-RI requires MPI_THREAD_FUNNELED");
        require(!atoms.empty(), "local-RI requires atoms");
        std::map<std::size_t, Engine::Tatom_pos> positions;
        for (std::size_t a = 0; a < atoms.size(); ++a) {
            panel_size({atoms[a].orbitals, atoms[a].auxiliaries});
            require(atoms[a].orbitals <=
                        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) -
                            offsets.back(),
                    "local-RI AO index overflow");
            offsets.push_back(offsets.back() + atoms[a].orbitals);
            positions[a] = {static_cast<double>(a), 0, 0};
        }
        engine.set_parallel(
            MPI_COMM_SELF, positions, {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}, {1, 1, 1}, channels);
        Blocks coefficients, metric;
        std::set<std::pair<std::size_t, std::size_t>> seen;
        for (const auto& fit : fits) {
            const auto [a, b] = std::pair{fit.left, fit.right};
            require(a <= b && b < atoms.size() && seen.emplace(a, b).second,
                    "invalid or duplicate local-RI fit pair");
            const auto na = atoms[a].orbitals, nb = atoms[b].orbitals;
            panel_size({atoms[a].auxiliaries, na, nb});
            matrix_shape(fit.left_fit, atoms[a].auxiliaries, panel_size({na, nb}));
            Tensor left({atoms[a].auxiliaries, na, nb});
            for (std::size_t p = 0; p < atoms[a].auxiliaries; ++p)
                for (std::size_t i = 0; i < na; ++i)
                    for (std::size_t j = 0; j < nb; ++j) {
                        if (a == b)
                            require(fit.left_fit(p, i * nb + j) == fit.left_fit(p, j * na + i),
                                    "on-site fit must be symmetric");
                        left(p, i, j) = fit.left_fit(p, i * nb + j) * (a == b ? 0.5 : 1.0);
                    }
            coefficients[a][{b, origin}] = std::move(left);
            if (a == b) {
                require(fit.right_fit.size() == 0, "on-site fit has only one endpoint");
            } else {
                panel_size({atoms[b].auxiliaries, nb, na});
                matrix_shape(fit.right_fit, atoms[b].auxiliaries, na * nb);
                Tensor right({atoms[b].auxiliaries, nb, na});
                for (std::size_t p = 0; p < atoms[b].auxiliaries; ++p)
                    for (std::size_t i = 0; i < na; ++i)
                        for (std::size_t j = 0; j < nb; ++j)
                            right(p, j, i) = fit.right_fit(p, i * nb + j);
                coefficients[b][{a, origin}] = std::move(right);
            }
        }
        seen.clear();
        for (const auto& block : coulomb) {
            const auto a = block.left, b = block.right;
            require(a <= b && b < atoms.size() && seen.emplace(a, b).second,
                    "invalid or duplicate local-RI Coulomb pair");
            const auto pa = atoms[a].auxiliaries, pb = atoms[b].auxiliaries;
            panel_size({pa, pb});
            matrix_shape(block.values, pa, pb);
            if (a == b)
                require(block.values.isApprox(block.values.transpose(), 1e-14),
                        "on-site Coulomb block must be symmetric");
            Tensor v({pa, pb});
            for (std::size_t p = 0; p < pa; ++p)
                for (std::size_t q = 0; q < pb; ++q)
                    v(p, q) = block.values(p, q);
            if (a != b)
                metric[b][{a, origin}] = v.transpose();
            metric[a][{b, origin}] = std::move(v);
        }
        for (std::size_t a = 0; a < atoms.size(); ++a)
            for (std::size_t b = 0; b < atoms.size(); ++b) {
                panel_size({atoms[a].orbitals, atoms[b].orbitals});
                density[a][{b, origin}] = Tensor({atoms[a].orbitals, atoms[b].orbitals});
            }
        const std::map<std::string, double> flags{
            {"flag_period", 0}, {"flag_comm", 0}, {"flag_filter", 0}};
        engine.set_tensors_map2(coefficients, {RI::Label::ab::a, RI::Label::ab::b}, flags, "C");
        engine.set_tensors_map2(metric, {RI::Label::ab::a0b0}, flags, "V");
        engine.set_tensors_map2(
            density,
            {RI::Label::ab::a1b1, RI::Label::ab::a1b2, RI::Label::ab::a2b1, RI::Label::ab::a2b2},
            flags,
            "D");
    }
};

Exchange::Exchange(const std::vector<AtomBasis>& atoms,
                   const std::vector<PairFit>& fits,
                   const std::vector<CoulombBlock>& coulomb)
    : impl_(std::make_unique<Impl>(atoms, fits, coulomb)) {}
Exchange::~Exchange() = default;

Eigen::MatrixXd Exchange::apply(const Eigen::Ref<const Eigen::MatrixXd>& input) {
    const auto n = static_cast<Eigen::Index>(impl_->offsets.back());
    require(input.rows() == n && input.cols() == n && input.allFinite() &&
                input.isApprox(input.transpose(), 1e-14),
            "local-RI density must be symmetric");
    // Tensor copies in the engine share these buffers. Updating values preserves
    // the fixed atom-pair index and never filters signed/zero response blocks.
    for (auto& [a, row] : impl_->density)
        for (auto& [key, block] : row)
            for (std::size_t i = 0; i < block.shape[0]; ++i)
                for (std::size_t j = 0; j < block.shape[1]; ++j)
                    block(i, j) = input(impl_->offsets[a] + i, impl_->offsets[key.first] + j);
    Blocks output;
    impl_->engine.cal_loop3(channels, output);
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, n);
    for (const auto& [a, row] : output)
        for (const auto& [key, block] : row)
            for (std::size_t i = 0; i < block.shape[0]; ++i)
                for (std::size_t j = 0; j < block.shape[1]; ++j)
                    result(impl_->offsets[a] + i, impl_->offsets[key.first] + j) += block(i, j);
    return result;
}
} // namespace xmvb::integrals::lri
