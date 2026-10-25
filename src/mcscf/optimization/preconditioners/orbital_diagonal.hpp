#pragma once

#include <cstddef>
#include <vector>
#include <Eigen/Core>

namespace xmvb::mcscf {

/** MO slices sufficient for an analytic fixed-state rotation diagonal. */
struct AnalyticDiagonalIntermediates {
    int n_core = 0;
    int n_active = 0;
    Eigen::MatrixXd h_core;
    Eigen::MatrixXd v_core;
    Eigen::MatrixXd j_pc;
    Eigen::MatrixXd k_pc;
    std::vector<double> j_active;
    std::vector<double> k_active;
    std::vector<double> active_eri;
};

struct DiagonalContractions {
    Eigen::VectorXd potential;
    Eigen::VectorXd gradient;
    Eigen::MatrixXd one_rdm;
    Eigen::MatrixXd two_rdm;
};

std::size_t orbital_diagonal_tensor4_index(int p, int q, int r, int s, int dimension) noexcept;
std::size_t orbital_diagonal_slice_index(int p, int a, int b, int n_active) noexcept;
std::size_t orbital_diagonal_packed_eri_count(int n_active) noexcept;

/** Symmetrized active 2-RDM represented by a packed integral cotangent. */
std::vector<double> unpack_symmetric_active_two_rdm(const std::vector<double>& packed_eri_gradient,
                                                    int n_active);

Eigen::VectorXd diagonal_gradient(const AnalyticDiagonalIntermediates& integrals,
                                  const Eigen::Ref<const Eigen::MatrixXd>& active_one_rdm,
                                  const std::vector<double>& active_two_rdm,
                                  const Eigen::VectorXd& potential);
Eigen::MatrixXd assemble_rotation_diagonal(const AnalyticDiagonalIntermediates& integrals,
                                           const Eigen::Ref<const Eigen::MatrixXd>& active_one_rdm,
                                           const DiagonalContractions& contractions);

/** Signed diagonal of real antisymmetric orbital rotations, all MO pairs. */
Eigen::MatrixXd
build_analytic_rotation_hessian_diagonal(const AnalyticDiagonalIntermediates& integrals,
                                         const Eigen::Ref<const Eigen::MatrixXd>& active_one_rdm,
                                         const std::vector<double>& packed_active_eri_gradient);

} // namespace xmvb::mcscf
