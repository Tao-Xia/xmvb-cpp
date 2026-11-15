#include "vbscf/structures/assembly/action.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "parallel/openmp.hpp"
#include "mcscf/ci/sigma.hpp"

namespace xmvb::vb {
namespace {

/** Each structure has one owner; its original summation order is retained. */
template <class Topology, class Read>
Eigen::MatrixXd project_products(const Topology& topology,
                                 Eigen::Index structures,
                                 Eigen::Index beta_size,
                                 Eigen::Index width,
                                 int parity,
                                 const Read& read) {
    Eigen::MatrixXd result(structures, width);
    const int workers = bounded_openmp_thread_count(effective_openmp_thread_count(), structures);
#pragma omp parallel if (workers > 1) num_threads(workers)
    {
        std::size_t worker = 0, team = 1;
#ifdef _OPENMP
        worker = omp_get_thread_num();
        team = omp_get_num_threads();
#endif
        const auto terms = topology.structure_offsets.back();
        const auto boundary = [&](std::size_t rank) -> std::size_t {
            if (rank == 0)
                return 0;
            if (rank == team)
                return structures;
            const auto target = (terms / team) * rank + std::min(terms % team, rank);
            return std::lower_bound(topology.structure_offsets.begin(),
                                    topology.structure_offsets.end(),
                                    target) -
                   topology.structure_offsets.begin();
        };
        const auto first = boundary(worker), last = boundary(worker + 1);
        const auto rows = [&](auto& sum) {
            for (auto structure = first; structure < last; ++structure) {
                std::fill(sum.begin(), sum.end(), 0.0);
                for (auto position = topology.structure_offsets[structure];
                     position < topology.structure_offsets[structure + 1];
                     ++position) {
                    const auto& term = topology.terms[topology.structure_terms[position]];
                    const auto product = topology.products[term.group];
                    const auto alpha = product / beta_size, beta = product % beta_size;
                    if (parity != 0 && alpha > beta)
                        continue;
                    for (std::size_t rhs = 0; rhs < sum.size(); ++rhs) {
                        double value = read(alpha, beta, rhs);
                        if (parity != 0 && alpha < beta)
                            value += parity * read(beta, alpha, rhs);
                        sum[rhs] += term.coefficient * value;
                    }
                }
                for (std::size_t rhs = 0; rhs < sum.size(); ++rhs)
                    result(structure, rhs) = sum[rhs];
            }
        };
        if (width == 1) {
            std::array<double, 1> sum;
            rows(sum);
        } else {
            std::vector<double> sum(width);
            rows(sum);
        }
    }
    return result;
}

} // namespace

Eigen::MatrixXd StructureAction::contract_spin_product_block(
    const Eigen::Ref<const Eigen::MatrixXd>& images) const {
    return contract_spin_product_block(images, 0);
}

Eigen::MatrixXd
StructureAction::contract_spin_product_block(const Eigen::Ref<const Eigen::MatrixXd>& images,
                                             int parity) const {
    if (images.rows() != n_unique_alpha_ || images.cols() <= 0 ||
        images.cols() % n_unique_beta_ != 0)
        throw std::invalid_argument(
            "unique-string-product image block has incompatible dimensions");
    return project_products(*topology_,
                            n_structures_,
                            n_unique_beta_,
                            images.cols() / n_unique_beta_,
                            parity,
                            [&](auto alpha, auto beta, auto rhs) {
                                return images(alpha, rhs * n_unique_beta_ + beta);
                            });
}

} // namespace xmvb::vb
