#pragma once

#include <vector>
#include <utility>

#include <Eigen/Core>

#include "vbscf/derivatives/hessian/responses/same_spin/backward.hpp"
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/pairs/accepted_tile.hpp"

namespace xmvb::vb::detail {

class SameSpinDirectionalPairTileView;

/**
 * @brief Directional data for one rectangular unique-spin pair tile.
 *
 * Storage is local to `[left_begin,left_end) x [right_begin,right_end)` and is
 * released after the current structure/backward consumers finish. No global
 * `n_unique x n_unique` directional object is required by this representation.
 */
struct SameSpinDirectionalPairTile {
  int left_begin = 0;
  int right_begin = 0;
  Eigen::MatrixXd delta_overlap;
  Eigen::MatrixXd delta_regular_hamiltonian;
  Eigen::MatrixXd delta_singular_hamiltonian;
  /** Pair-major packed images supplied directly by regular RI contractions. */
  Eigen::MatrixXd ri_projected_channels;
  std::vector<unsigned char> ri_projected_ready;
  std::vector<SameSpinPolynomialDirectionalPairData> pairs;

  int left_size() const noexcept {
    return static_cast<int>(delta_overlap.rows());
  }

  int right_size() const noexcept {
    return static_cast<int>(delta_overlap.cols());
  }

  const SameSpinPolynomialDirectionalPairData& pair(
      int left_local,
      int right_local) const;

  SameSpinDirectionalPairTileView view(bool transposed = false) const;
};

/** Zero-copy orientation view of one canonical same-spin tile. */
class SameSpinDirectionalPairTileView {
public:
  using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
  using ConstMatrixMap = Eigen::Map<
      const Eigen::MatrixXd, Eigen::Unaligned, Stride>;

  SameSpinDirectionalPairTileView(
      const SameSpinDirectionalPairTile& storage,
      bool transposed) noexcept
      : storage_(&storage), transposed_(transposed) {}

  int left_begin() const noexcept;
  int right_begin() const noexcept;
  int left_size() const noexcept;
  int right_size() const noexcept;
  bool transposed() const noexcept { return transposed_; }

  ConstMatrixMap delta_overlap() const;
  ConstMatrixMap delta_regular_hamiltonian() const;
  ConstMatrixMap delta_singular_hamiltonian() const;

  template <typename Consumer>
  void with_pair(
      int left_local,
      int right_local,
      Consumer&& consume) const {
    if (!transposed_) {
      consume(storage_->pair(left_local, right_local));
      return;
    }
    SameSpinPolynomialDirectionalPairData pair =
        storage_->pair(right_local, left_local);
    pair.delta_cofactor_1st.transposeInPlace();
    pair.delta_same_spin_overlap_hamiltonian_gradient.transposeInPlace();
    consume(pair);
  }

private:
  ConstMatrixMap matrix_view(const Eigen::MatrixXd& matrix) const;

  const SameSpinDirectionalPairTile* storage_ = nullptr;
  bool transposed_ = false;
};

SameSpinDirectionalPairTile build_directional_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const std::vector<SpinDeterminantPairEvaluation>& ordered_pair_cache,
    int n_unique_determinants,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    int left_begin,
    int left_end,
    int right_begin,
    int right_end,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool build_ri_projected_channels = false);

SameSpinDirectionalPairTile build_directional_pair_tile(
    const std::vector<std::vector<int>>& unique_determinants,
    const AcceptedSpinPairTile& accepted_pair_tile,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    const Eigen::MatrixXd* accepted_active_one_electron,
    const Eigen::MatrixXd* accepted_ri_active_pair_factors,
    const Eigen::MatrixXd* directional_ri_active_pair_factors,
    bool build_ri_projected_channels = false);

void accumulate_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    double weight,
    Eigen::MatrixXd* active_one_electron_gradient);

void accumulate_overlap_block_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& overlap_block_gradient,
    double weight,
    int n_active_orbitals,
    std::vector<double>* active_orbital_overlap_gradient);

void accumulate_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    double weight,
    std::vector<double>* packed_active_two_electron_gradient);

SameSpinPolynomialDirectionalPairData build_polynomial_spin_directional_data(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const SpinDeterminantPairEvaluation& pair_evaluation,
    int n_active_orbitals,
    const ActiveSpaceIntegralDirectionView& direction,
    bool need_overlap_gradient = true);

void accumulate_directional_one_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const Eigen::MatrixXd& cofactor_1st,
    const Eigen::MatrixXd& delta_cofactor_1st,
    double weight,
    double delta_weight,
    Eigen::MatrixXd* active_one_electron_gradient);

void accumulate_directional_deleted_minor_same_spin_two_electron_gradient_contribution_local(
    const std::vector<int>& occ_L,
    const std::vector<int>& occ_R,
    const CofactorDifferential& cofactor,
    const std::vector<double>& delta_ao_overlap_matrix,
    int n_active_orbitals,
    double weight,
    double delta_weight,
    std::vector<double>* packed_active_two_electron_gradient);

}  // namespace xmvb::vb::detail
