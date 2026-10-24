#include "casscf/integrals/active.hpp"

#include <cmath>
#include <stdexcept>
#include "core/pair_index.hpp"
#include "integrals/ao/exact.hpp"
#include "integrals/active/exact.hpp"
#include "integrals/active/one_electron.hpp"
#include "integrals/active/two_electron.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::casscf {

CasIntegrals make_active_integrals(const Eigen::Ref<const Eigen::MatrixXd>& h,
                                   const MoCoulomb& eri,
                                   int core,
                                   int active,
                                   double nuclear) {
    const int n = static_cast<int>(h.rows());
    if (core < 0 || active < 1 || core + active > n || h.cols() != n || !h.allFinite() || !eri ||
        !std::isfinite(nuclear))
        throw std::invalid_argument("invalid orthonormal MO integrals or CAS partition");
    CasIntegrals result;
    result.core_energy = nuclear;
    for (int i = 0; i < core; ++i) {
        result.core_energy += 2 * h(i, i);
        for (int j = 0; j < core; ++j)
            result.core_energy += 2 * eri(i, i, j, j) - eri(i, j, j, i);
    }
    result.active.one_electron = h.block(core, core, active, active);
    for (int a = 0; a < active; ++a)
        for (int b = 0; b < active; ++b)
            for (int i = 0; i < core; ++i)
                result.active.one_electron(a, b) +=
                    2 * eri(core + a, core + b, i, i) - eri(core + a, i, i, core + b);
    const int active_pairs = active * (active + 1) / 2;
    result.active.pair_kernel.resize(active_pairs, active_pairs);
    for (int a = 0; a < active; ++a)
        for (int b = 0; b <= a; ++b)
            for (int c = 0; c < active; ++c)
                for (int d = 0; d <= c; ++d)
                    result.active.pair_kernel(core::PairIndex::packed_pair_index(a, b),
                                              core::PairIndex::packed_pair_index(c, d)) =
                        eri(core + a, core + b, core + c, core + d);
    return result;
}

PreparedCasIntegrals prepare_active_integrals(const integrals::AoIntegralInput& ao,
                                              const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                                              int core,
                                              int active,
                                              double nuclear) {
    const int n_bf = ao.n_basis_functions;
    if (n_bf < 1 || core < 0 || active < 1 || core + active > n_bf || orbitals.rows() != n_bf ||
        orbitals.cols() != n_bf || ao.ao_core_hamiltonian_matrix.rows() != n_bf ||
        ao.ao_core_hamiltonian_matrix.cols() != n_bf || ao.pair_graph.empty() ||
        !orbitals.allFinite() || !ao.ao_core_hamiltonian_matrix.allFinite() ||
        !std::isfinite(nuclear))
        throw std::invalid_argument("invalid AO integrals, MO frame or CAS partition");

    PreparedCasIntegrals result;
    result.core_density = orbitals.leftCols(core) * orbitals.leftCols(core).transpose();
    const Eigen::Map<const Eigen::MatrixXd> source(
        result.core_density.data(), result.core_density.size(), 1);
    integrals::AoH1eBatchWorkspace workspace;
    Eigen::MatrixXd response;
    integrals::apply_symmetric_ao_h1e_batch(
        source, ao, effective_openmp_thread_count(), &workspace, &response);
    const Eigen::Map<const Eigen::MatrixXd> potential(response.data(), n_bf, n_bf);
    result.fock = ao.ao_core_hamiltonian_matrix + potential;
    result.integrals.core_energy =
        nuclear +
        (result.core_density.array() * (2 * ao.ao_core_hamiltonian_matrix + potential).array())
            .sum();
    result.integrals.active.one_electron = integrals::ActiveSpaceOneElectronBuilder{}
                                               .build(result.fock, orbitals, n_bf, core, active)
                                               .h1e_act;

    // Borrow sparse orbital support in the same numerical transform as VBSCF.
    std::vector<int> offsets(1, 0), indices;
    std::vector<double> coefficients;
    for (int mu = 0; mu < n_bf; ++mu) {
        for (int a = 0; a < active; ++a) {
            const double value = orbitals(mu, core + a);
            if (value == 0)
                continue;
            indices.push_back(a);
            coefficients.push_back(value);
        }
        offsets.push_back(static_cast<int>(indices.size()));
    }
    result.two_electron =
        integrals::transform_exact_active_integrals(ao, offsets, indices, coefficients, active);
    const auto view = integrals::make_active_space_two_electron_view(result.two_electron);
    const int pairs = integrals::packed_active_pair_count(active);
    result.integrals.active.pair_kernel.resize(pairs, pairs);
    for (int p = 0; p < pairs; ++p)
        for (int q = 0; q < pairs; ++q)
            result.integrals.active.pair_kernel(p, q) =
                integrals::lookup_active_space_two_electron_kernel_value(view, p, q, active);
    return result;
}

CasIntegrals make_active_integrals(const integrals::AoIntegralInput& ao,
                                   const Eigen::Ref<const Eigen::MatrixXd>& orbitals,
                                   int core,
                                   int active,
                                   double nuclear) {
    return prepare_active_integrals(ao, orbitals, core, active, nuclear).integrals;
}

} // namespace xmvb::casscf
