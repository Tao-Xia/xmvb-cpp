#include "output/text/normalization.hpp"

#include <cmath>
#include <stdexcept>

#include "core/timing.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
#include "vbscf/structures/assembly/hamiltonian_overlap.hpp"

namespace xmvb::output {

StructureNorms structure_norms(const vb::VbScfOptimizerResult& result) {
    const auto& scf = result.scf_result;
    const int n_str = scf.n_structures;
    if (n_str <= 0)
        throw std::invalid_argument("structure report requires a nonempty space");
    Eigen::VectorXd diagonal;
    if (scf.structure_overlap_diagonal_exact) {
        if (scf.structure_overlap_diagonal.size() != static_cast<std::size_t>(n_str))
            throw std::invalid_argument("exact structure norms have inconsistent dimensions");
        diagonal = Eigen::Map<const Eigen::VectorXd>(scf.structure_overlap_diagonal.data(), n_str);
    } else if (scf.structure_matrices.overlap_matrix.size() ==
               static_cast<std::size_t>(n_str) * n_str) {
        diagonal = Eigen::Map<const Eigen::MatrixXd>(
                       scf.structure_matrices.overlap_matrix.data(), n_str, n_str)
                       .diagonal();
    } else {
        profiling::ScopedPhase phase(profiling::Phase::StructureNormalization, true);
        const auto& input = result.optimized_input;
        const int n_active = input.orbital_preparation_input.n_active_orbitals;
        if (result.active_orbital_overlap_matrix.size() !=
            static_cast<std::size_t>(n_active) * n_active)
            throw std::invalid_argument("structure report requires the accepted active overlap");
        // Only topology is prepared, once per report; no physical pair cache is
        // rebuilt or retained, and no exported orbital frame is re-evaluated.
        const auto topology = vb::build_same_spin_pair_topology(
            input.structure_data.alpha(), input.structure_data.beta(), n_active);
        diagonal =
            vb::FullDeterminantStructureHamiltonianOverlapBuilder{}.build_exact_overlap_diagonal(
                input.structure_data.alpha(),
                input.structure_data.beta(),
                input.structure_data.incidence(),
                result.active_orbital_overlap_matrix,
                n_active,
                n_str,
                topology);
    }
    StructureNorms norms;
    norms.mean_overlap = diagonal.mean();
    norms.values.resize(n_str);
    for (int i = 0; i < n_str; ++i) {
        if (!std::isfinite(diagonal[i]) || diagonal[i] <= 0.0)
            throw std::domain_error("a final VB structure has a non-positive or non-finite norm");
        norms.values[i] = std::sqrt(diagonal[i]);
    }
    return norms;
}

} // namespace xmvb::output
