#pragma once
#include "dft/grid/atomic.hpp"
#include "integrals/lri/exchange.hpp"
#include "integrals/nao/radial.hpp"
#include <vector>

namespace xmvb::integrals::nao {
struct Function {
    std::shared_ptr<const Radial> radial;
    int m;
};
struct Center {
    Eigen::Vector3d position;
    std::vector<Function> orbitals;
    std::vector<Function> auxiliaries;
};
struct PreparationOptions {
    dft::GridOptions grid;
    // Required execution partition, not a physical truncation/cache threshold.
    std::size_t panel_points;
    double relative_rank_cutoff;
};
struct Prepared {
    Eigen::MatrixXd overlap, kinetic;
    std::vector<lri::AtomBasis> atoms;
    std::vector<lri::PairFit> fits;
    std::vector<lri::CoulombBlock> coulomb;
    std::vector<std::size_t> fit_ranks;
    double max_fit_residual = 0;
    double kinetic_asymmetry = 0;
    // Bytes in the four retained field panels, not total process RSS.
    std::size_t panel_bytes = 0;
};
// Finite-system reference preparation. Returns S, Hartree-unit T, local fits
// and Coulomb blocks; no nuclear/pseudopotential or periodic sums are implied.
// AO values and auxiliary potentials are evaluated in shared panels; no
// full grid x AO or global grid x AO-pair arrays are retained.
Prepared prepare(const std::vector<Center>& centers, const PreparationOptions& options);
} // namespace xmvb::integrals::nao
