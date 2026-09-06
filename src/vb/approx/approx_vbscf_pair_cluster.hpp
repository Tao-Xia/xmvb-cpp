#pragma once

#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vb/orbital/active_space_two_electron_result.hpp"

namespace xmvb::vb {

struct ApproxVbScfPairClusterMatrixElement {
  double overlap = 0.0;
  double one_electron_hamiltonian = 0.0;
  double total_hamiltonian = 0.0;
};

struct ApproxVbScfPairClusterHamiltonianInput {
  int n_active_orbitals = 0;
  const std::vector<double>* active_orbital_overlap_matrix = nullptr;
  const Eigen::MatrixXd* active_one_electron_integrals = nullptr;
  const ActiveSpaceTwoElectronResult* active_two_electron_result = nullptr;
};

int approx_vbscf_active_pair_count(int n_active_orbitals);

std::pair<int, int> unpack_approx_vbscf_active_pair(int pair_index);

bool approx_vbscf_pair_cluster_is_compatible(
    const std::vector<std::pair<int, int>>& pairs);

/**
 * @brief Evaluates the exact spin-adapted overlap of a tiny local VB pair cluster.
 *
 * The cluster is built from one or two active pair patterns and uses the same
 * determinant expansion convention as the exact VB structure-overlap code.
 */
double evaluate_approx_vbscf_pair_cluster_overlap(
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<double>& active_orbital_overlap_matrix,
    int n_active_orbitals);

/**
 * @brief Evaluates exact active-space Hamiltonian and overlap for one local cluster.
 *
 * This is the local numerator counterpart of the pair-cluster metric.  It
 * enumerates only the determinant terms of the tiny cluster and reuses the
 * exact determinant Hamiltonian evaluator; it never constructs the global VB
 * structure list.
 */
ApproxVbScfPairClusterMatrixElement
evaluate_approx_vbscf_pair_cluster_matrix_element(
    const std::vector<std::pair<int, int>>& pairs,
    const ApproxVbScfPairClusterHamiltonianInput& input);

}  // namespace xmvb::vb
