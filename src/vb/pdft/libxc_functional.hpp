#pragma once

#include <memory>
#include <Eigen/Core>

namespace xmvb::vb::pdft {

/**
 * @brief Wrapper for libxc exchange-correlation functionals.
 *
 * This class provides a simplified interface to libxc for evaluating
 * XC energy densities at grid points. It handles the libxc initialization,
 * memory management, and calling conventions.
 *
 * The wrapper supports both LDA and GGA functionals, automatically
 * detecting which type based on the functional ID.
 */
class LibxcFunctional {
public:
  /**
   * @brief Constructs functional from libxc identifier.
   *
   * Common functional IDs:
   * - XC_LDA_X (1): LDA exchange (Slater)
   * - XC_LDA_C_VWN (7): LDA correlation (VWN)
   * - XC_GGA_X_PBE (101): GGA exchange (PBE)
   * - XC_GGA_C_PBE (130): GGA correlation (PBE)
   *
   * @param functional_id Libxc functional ID.
   * @throws std::invalid_argument if functional ID is not supported.
   */
  explicit LibxcFunctional(int functional_id);

  /**
   * @brief Destructor (handles libxc cleanup).
   */
  ~LibxcFunctional();

  // Disable copy (libxc context is not copyable)
  LibxcFunctional(const LibxcFunctional&) = delete;
  LibxcFunctional& operator=(const LibxcFunctional&) = delete;

  // Enable move
  LibxcFunctional(LibxcFunctional&&) noexcept;
  LibxcFunctional& operator=(LibxcFunctional&&) noexcept;

  /**
   * @brief Evaluates XC energy density at grid points (LDA).
   *
   * Computes eps_xc(r) such that:
   *   E_xc = integral rho(r) * eps_xc(r) dr
   *
   * @param rho_alpha Alpha spin density at each point.
   * @param rho_beta Beta spin density at each point.
   * @return XC energy density eps_xc(r) at each point.
   * @throws std::runtime_error if functional requires gradients (GGA).
   */
  Eigen::VectorXd evaluate_energy_density(
      const Eigen::VectorXd& rho_alpha,
      const Eigen::VectorXd& rho_beta) const;

  /**
   * @brief Evaluates XC energy density with gradients (GGA).
   *
   * @param rho_alpha Alpha spin density.
   * @param rho_beta Beta spin density.
   * @param grad_rho_alpha Gradient of alpha density (n_points x 3).
   * @param grad_rho_beta Gradient of beta density (n_points x 3).
   * @return XC energy density eps_xc(r) at each point.
   */
  Eigen::VectorXd evaluate_energy_density_gga(
      const Eigen::VectorXd& rho_alpha,
      const Eigen::VectorXd& rho_beta,
      const Eigen::MatrixXd& grad_rho_alpha,
      const Eigen::MatrixXd& grad_rho_beta) const;

  /**
   * @brief Returns the functional ID.
   */
  int functional_id() const { return functional_id_; }

  /**
   * @brief Returns whether this is a GGA functional.
   */
  bool is_gga() const;

  /**
   * @brief Returns the functional name.
   */
  std::string name() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int functional_id_;
};

}  // namespace xmvb::vb::pdft
