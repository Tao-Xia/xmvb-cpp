#include "vb/pdft/libxc_functional.hpp"

#include <stdexcept>
#include <cstring>
#include <vector>

// Libxc headers
#ifdef XMVB_CPP_HAS_LIBXC
#include <xc.h>
#endif

namespace xmvb::vb::pdft {

#ifdef XMVB_CPP_HAS_LIBXC

// Implementation details hidden in pimpl
struct LibxcFunctional::Impl {
  xc_func_type func;
  bool initialized = false;

  ~Impl() {
    if (initialized) {
      xc_func_end(&func);
    }
  }
};

LibxcFunctional::LibxcFunctional(int functional_id)
    : impl_(std::make_unique<Impl>()),
      functional_id_(functional_id) {
  // Initialize libxc functional
  if (xc_func_init(&impl_->func, functional_id, XC_POLARIZED) != 0) {
    throw std::invalid_argument("Failed to initialize libxc functional");
  }
  impl_->initialized = true;
}

LibxcFunctional::~LibxcFunctional() = default;

LibxcFunctional::LibxcFunctional(LibxcFunctional&&) noexcept = default;
LibxcFunctional& LibxcFunctional::operator=(LibxcFunctional&&) noexcept = default;

bool LibxcFunctional::is_gga() const {
  return impl_->func.info->family == XC_FAMILY_GGA ||
         impl_->func.info->family == XC_FAMILY_HYB_GGA;
}

std::string LibxcFunctional::name() const {
  return std::string(impl_->func.info->name);
}

Eigen::VectorXd LibxcFunctional::evaluate_energy_density(
    const Eigen::VectorXd& rho_alpha,
    const Eigen::VectorXd& rho_beta) const {
  const int n_points = static_cast<int>(rho_alpha.size());

  if (rho_beta.size() != n_points) {
    throw std::invalid_argument("rho_alpha and rho_beta size mismatch");
  }

  if (is_gga()) {
    throw std::runtime_error("GGA functional requires gradients");
  }

  // Prepare input arrays for libxc
  std::vector<double> rho_input(2 * n_points);
  for (int i = 0; i < n_points; ++i) {
    rho_input[2 * i] = rho_alpha(i);
    rho_input[2 * i + 1] = rho_beta(i);
  }

  // Allocate output
  std::vector<double> zk(n_points);

  // Call libxc
  xc_lda_exc(&impl_->func, n_points, rho_input.data(), zk.data());

  // Convert to Eigen
  Eigen::VectorXd result(n_points);
  for (int i = 0; i < n_points; ++i) {
    result(i) = zk[i];
  }

  return result;
}

Eigen::VectorXd LibxcFunctional::evaluate_energy_density_gga(
    const Eigen::VectorXd& rho_alpha,
    const Eigen::VectorXd& rho_beta,
    const Eigen::MatrixXd& grad_rho_alpha,
    const Eigen::MatrixXd& grad_rho_beta) const {
  const int n_points = static_cast<int>(rho_alpha.size());

  if (!is_gga()) {
    // For LDA, just ignore gradients
    return evaluate_energy_density(rho_alpha, rho_beta);
  }

  // Prepare input for GGA
  std::vector<double> rho_input(2 * n_points);
  std::vector<double> sigma_input(3 * n_points);

  for (int i = 0; i < n_points; ++i) {
    rho_input[2 * i] = rho_alpha(i);
    rho_input[2 * i + 1] = rho_beta(i);

    // Compute sigma = |grad rho|^2
    const double gx_a = grad_rho_alpha(i, 0);
    const double gy_a = grad_rho_alpha(i, 1);
    const double gz_a = grad_rho_alpha(i, 2);
    const double gx_b = grad_rho_beta(i, 0);
    const double gy_b = grad_rho_beta(i, 1);
    const double gz_b = grad_rho_beta(i, 2);

    sigma_input[3 * i] = gx_a * gx_a + gy_a * gy_a + gz_a * gz_a;  // sigma_aa
    sigma_input[3 * i + 1] = gx_a * gx_b + gy_a * gy_b + gz_a * gz_b;  // sigma_ab
    sigma_input[3 * i + 2] = gx_b * gx_b + gy_b * gy_b + gz_b * gz_b;  // sigma_bb
  }

  // Allocate output
  std::vector<double> zk(n_points);

  // Call libxc GGA
  xc_gga_exc(&impl_->func, n_points, rho_input.data(), sigma_input.data(), zk.data());

  // Convert to Eigen
  Eigen::VectorXd result(n_points);
  for (int i = 0; i < n_points; ++i) {
    result(i) = zk[i];
  }

  return result;
}

#else  // !XMVB_CPP_HAS_LIBXC

// Stub implementation when libxc is not available
struct LibxcFunctional::Impl {};

LibxcFunctional::LibxcFunctional(int functional_id)
    : impl_(std::make_unique<Impl>()),
      functional_id_(functional_id) {
  throw std::runtime_error("LibxcFunctional requires libxc support");
}

LibxcFunctional::~LibxcFunctional() = default;
LibxcFunctional::LibxcFunctional(LibxcFunctional&&) noexcept = default;
LibxcFunctional& LibxcFunctional::operator=(LibxcFunctional&&) noexcept = default;

bool LibxcFunctional::is_gga() const {
  throw std::runtime_error("LibxcFunctional requires libxc support");
}

std::string LibxcFunctional::name() const {
  throw std::runtime_error("LibxcFunctional requires libxc support");
}

Eigen::VectorXd LibxcFunctional::evaluate_energy_density(
    const Eigen::VectorXd&,
    const Eigen::VectorXd&) const {
  throw std::runtime_error("LibxcFunctional requires libxc support");
}

Eigen::VectorXd LibxcFunctional::evaluate_energy_density_gga(
    const Eigen::VectorXd&,
    const Eigen::VectorXd&,
    const Eigen::MatrixXd&,
    const Eigen::MatrixXd&) const {
  throw std::runtime_error("LibxcFunctional requires libxc support");
}

#endif  // XMVB_CPP_HAS_LIBXC

}  // namespace xmvb::vb::pdft
