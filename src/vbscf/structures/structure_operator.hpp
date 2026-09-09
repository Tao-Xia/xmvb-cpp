#pragma once

#include <vector>

#include <Eigen/Core>

namespace xmvb::vb {

struct StructureCoefficientBlock;

/**
 * @brief Abstract matrix-vector product for the structure Hamiltonian and overlap.
 *
 * Davidson needs H*x and S*x without materializing the full N x N matrices.
 * DenseStructureMatrixOperator wraps already-materialized matrices for LAPACK
 * dsygvd.  TiledStructureMatrixOperator computes matrix-free products using
 * the same unique-spin-string tile infrastructure as the forward assembly.
 */
class StructureMatrixOperator {
public:
  virtual ~StructureMatrixOperator() = default;
  virtual int dimension() const = 0;

  /// y = H * x  (Hamiltonian matrix-vector product)
  virtual void apply_hamiltonian(const double* x, double* y) const = 0;

  /// y = S * x  (overlap matrix-vector product)
  virtual void apply_overlap(const double* x, double* y) const = 0;

  /// Diagonal of (H - shift * S)^-1 for Davidson preconditioning.
  /// Returns diag_H[i] and diag_S[i] separately; caller computes shift.
  virtual Eigen::VectorXd diagonal_hamiltonian() const = 0;
  virtual Eigen::VectorXd diagonal_overlap() const = 0;
};

// ---------------------------------------------------------------------------
// Dense (materialized) operator — wraps already-built H and S matrices.
// Used by LAPACK dsygvd through GeneralizedEigensolver::solve().
// ---------------------------------------------------------------------------

class DenseStructureMatrixOperator : public StructureMatrixOperator {
public:
  DenseStructureMatrixOperator(
      const std::vector<double>& hamiltonian_matrix,
      const std::vector<double>& overlap_matrix,
      int dimension);

  int dimension() const override;
  void apply_hamiltonian(const double* x, double* y) const override;
  void apply_overlap(const double* x, double* y) const override;
  Eigen::VectorXd diagonal_hamiltonian() const override;
  Eigen::VectorXd diagonal_overlap() const override;

  const std::vector<double>& hamiltonian_matrix() const { return H_; }
  const std::vector<double>& overlap_matrix() const { return S_; }

private:
  std::vector<double> H_;
  std::vector<double> S_;
  int dim_ = 0;
};

// ---------------------------------------------------------------------------
// Tiled (matrix-free) operator — uses unique-spin-string tile infrastructure.
// Used by Davidson through GeneralizedEigensolver::solve_davidson().
//
// The forward assembly already builds coefficient blocks and spin-pair tile
// providers.  This operator references that infrastructure and computes
// H*x and S*x without materializing the full N x N matrices.
//
// Structure matrix element:
//   H[I,J] = sum_{a,a',b,b'} C_I[a,b] * H_spin(a,b;a',b') * C_J[a',b']
// where C_I is the local coefficient matrix (alpha_support x beta_support)
// and H_spin is the spin-pair entry from the ForwardSpinPairTileProvider.
// ---------------------------------------------------------------------------

template <typename TwoElectronInputT>
class ForwardSpinPairTileProvider;

class TiledStructureMatrixOperator : public StructureMatrixOperator {
public:
  /// @param coefficient_blocks  Structure-to-unique-spin mapping (borrowed)
  /// @param det_overlap_cache    Determinant diagonal overlaps (borrowed)
  /// @param alpha_provider       Alpha-spin pair tiles (borrowed)
  /// @param beta_provider        Beta-spin pair tiles (borrowed)
  /// @param n_structures         Number of VB structures (= dimension)
  /// @param n_unique_alpha       Number of unique alpha spin strings
  /// @param n_unique_beta        Number of unique beta spin strings
  /// @param close_shell_same_spin  True if alpha and beta share identical strings
  TiledStructureMatrixOperator(
      const std::vector<StructureCoefficientBlock>& coefficient_blocks,
      const ForwardSpinPairTileProvider<std::vector<double>>& alpha_provider,
      const ForwardSpinPairTileProvider<std::vector<double>>& beta_provider,
      const std::vector<double>& det_overlap_cache,
      int n_structures,
      int n_unique_alpha,
      int n_unique_beta,
      bool close_shell_same_spin);

  int dimension() const override;
  void apply_hamiltonian(const double* x, double* y) const override;
  void apply_overlap(const double* x, double* y) const override;
  Eigen::VectorXd diagonal_hamiltonian() const override;
  Eigen::VectorXd diagonal_overlap() const override;

private:
  /// Compute C_I · H_tile · C_J^T · x[J] and accumulate into y[I].
  void accumulate_structure_pair(
      const StructureCoefficientBlock& block_I,
      const StructureCoefficientBlock& block_J,
      double x_J,
      double* y_I) const;

  const std::vector<StructureCoefficientBlock>& blocks_;
  const ForwardSpinPairTileProvider<std::vector<double>>& alpha_provider_;
  const ForwardSpinPairTileProvider<std::vector<double>>& beta_provider_;
  const std::vector<double>& det_overlap_cache_;
  int n_structures_ = 0;
  int n_unique_alpha_ = 0;
  int n_unique_beta_ = 0;
  bool close_shell_same_spin_ = false;
};

}  // namespace xmvb::vb
