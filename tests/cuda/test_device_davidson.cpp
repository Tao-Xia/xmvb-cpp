#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/linalg/blas.hpp"
#include "cuda/linalg/davidson.hpp"
#include "cuda/memory/buffer.hpp"
#include "linalg/eigensolver.hpp"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Eigen::MatrixXd download(xmvb::cuda::Context& context,
                         xmvb::linalg::ConstMatrixView source) {
  Eigen::MatrixXd host(source.rows, source.cols);
  xmvb::cuda::copy_to_host(context, host, source);
  return host;
}

void check_case(bool identity_overlap) {
  constexpr int n = 6;
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd s = Eigen::MatrixXd::Identity(n, n);
  for (int i = 0; i < n; ++i) {
    h(i, i) = -1.3 + 0.43 * i;
    for (int j = 0; j < i; ++j) {
      h(i, j) = h(j, i) = 0.035 * std::cos((i + 1) * (j + 2));
      if (!identity_overlap) {
        s(i, j) = s(j, i) = 0.015 * std::sin((i + 2) * (j + 1));
      }
    }
  }
  const Eigen::VectorXd h_diagonal = h.diagonal();
  const Eigen::VectorXd s_diagonal = s.diagonal();
  auto options = xmvb::linalg::make_davidson_options(n, 2, 1e-9, 1e-8, 1e-9);
  options.identity_overlap = identity_overlap;

  xmvb::cuda::Context context(64ULL * 1024ULL * 1024ULL);
  xmvb::cuda::Matrix<double> device_h(context, n, n);
  xmvb::cuda::Matrix<double> device_s(context, n, n);
  xmvb::cuda::copy_to_device(context, device_h.view(), h);
  xmvb::cuda::copy_to_device(context, device_s.view(), s);
  const xmvb::cuda::DeviceEigenAction action = [&](xmvb::linalg::ConstMatrixView x,
      xmvb::linalg::MatrixView hx, xmvb::linalg::MatrixView sx) {
    xmvb::cuda::gemm(context, hx, device_h.view(), x);
    if (!identity_overlap) xmvb::cuda::gemm(context, sx, device_s.view(), x);
  };
  const xmvb::linalg::GeneralizedEigenAction cpu_action = [&]
      (const Eigen::Ref<const Eigen::MatrixXd>& x) {
    return xmvb::linalg::GeneralizedEigenActionResult{
        h * x, identity_overlap ? Eigen::MatrixXd{} : (s * x).eval()};
  };

  const auto cpu_result = xmvb::linalg::GeneralizedEigensolver().solve_davidson(
      cpu_action, h_diagonal, s_diagonal, options);
  const auto host_result = xmvb::cuda::solve_davidson(
      context, action, h_diagonal, s_diagonal, options);
  const auto device_result = xmvb::cuda::solve_device_davidson(
      context, action, h_diagonal, s_diagonal, options);
  require(device_result.eigenpairs.eigenvector_matrix.rows() == n &&
              device_result.eigenpairs.eigenvector_matrix.cols() == 2,
          "device Davidson roots have wrong dimensions");
  require(device_result.hamiltonian_eigenvectors.rows() == n &&
              device_result.hamiltonian_eigenvectors.cols() == 2,
          "device Davidson H images have wrong dimensions");
  require(device_result.overlap_view().rows == n &&
              device_result.overlap_view().cols == 2,
          "device Davidson overlap images have wrong dimensions");
  require(device_result.identity_overlap == identity_overlap,
          "device Davidson overlap mode changed");
  require(device_result.iterations == host_result.iterations &&
              device_result.block_actions == host_result.block_actions &&
              device_result.peak_subspace_dimension ==
                  host_result.peak_subspace_dimension,
          "device Davidson iteration metadata differs from host-output solve");

  const Eigen::MatrixXd roots = download(
      context, device_result.eigenpairs.eigenvector_matrix.view());
  const Eigen::MatrixXd h_roots = download(
      context, device_result.hamiltonian_eigenvectors.view());
  const Eigen::MatrixXd s_roots = download(context, device_result.overlap_view());
  for (int root = 0; root < 2; ++root) {
    const double eigenvalue = device_result.eigenpairs.eigenvalues.at(root);
    require(std::abs(eigenvalue - host_result.eigenpairs.eigenvalues.at(root)) <
                2e-10,
            "device Davidson Ritz energy differs from host-output solve");
    require(std::abs(eigenvalue - cpu_result.eigenpairs.eigenvalues.at(root)) <
                2e-9,
            "device Davidson Ritz energy differs from CPU Davidson");
    const Eigen::Map<const Eigen::VectorXd> host_root(
        host_result.eigenpairs.eigenvector_matrix.data() + n * root, n);
    const double sign = host_root.dot(roots.col(root)) < 0.0 ? -1.0 : 1.0;
    require((roots.col(root) - sign * host_root).norm() < 2e-9,
            "device Davidson root differs from host-output solve");
    require((h_roots.col(root) - h * roots.col(root)).norm() < 2e-9 &&
                (s_roots.col(root) - s * roots.col(root)).norm() < 2e-9,
            "device Davidson retained images differ from direct products");
    require(device_result.stationarity_residual_norms.at(root) < 1.01e-9,
            "device Davidson stationarity threshold was not met");
  }

  const auto restarted = xmvb::cuda::solve_device_davidson(
      context, action, h_diagonal, s_diagonal,
      device_result.eigenpairs.eigenvector_matrix.view(), options);
  for (int root = 0; root < 2; ++root) {
    require(std::abs(restarted.eigenpairs.eigenvalues.at(root) -
                     device_result.eigenpairs.eigenvalues.at(root)) < 2e-10,
            "device-initial Davidson changed the converged Ritz energy");
  }
}

}  // namespace

int main() {
  try {
    check_case(false);
    check_case(true);
    std::cout << "CUDA device-output Davidson parity: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
