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
  for (int count:{1,2,4,5}) {
    const Eigen::MatrixXd expected=xmvb::linalg::make_davidson_initial_vectors(
        h_diagonal,s_diagonal,count);
    Eigen::MatrixXd prefix=expected;
    prefix.rightCols(count-count/2).setZero();
    xmvb::cuda::Matrix<double> seeds(context,n,count);
    xmvb::cuda::copy_to_device(context,seeds.view(),prefix);
    xmvb::cuda::fill_davidson_sign_seeds(context,seeds.view(),count/2);
    require((download(context,seeds.view())-expected).norm()==0.0,
        "device Davidson sign sequence differs from CPU initial subspace");
  }
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

void check_projected_correction_scale() {
  using namespace xmvb;
  cuda::Context context(64ULL * 1024 * 1024);
  Eigen::Matrix3d h = Eigen::Matrix3d::Zero();
  h(0, 1) = h(1, 0) = 1.0;
  h(2, 2) = 2.0;
  Eigen::Matrix3d inverse;
  inverse << 1e16, 1e8, 0, 1e8, 2, 0, 0, 0, 1;
  Eigen::MatrixXd seed = Eigen::MatrixXd::Zero(3, 1);
  seed(0, 0) = 1.0;
  cuda::Matrix<double> dh(context, 3, 3), dp(context, 3, 3), initial(context, 3, 1);
  cuda::copy_to_device(context, dh.view(), h);
  cuda::copy_to_device(context, dp.view(), inverse);
  cuda::copy_to_device(context, initial.view(), seed);
  for (bool identity : {false, true}) {
    const cuda::DeviceEigenAction action = [&](linalg::ConstMatrixView x,
        linalg::MatrixView hx, linalg::MatrixView mx) {
      cuda::gemm(context, hx, dh.view(), x);
      if (!identity) cuda::check_cuda(cudaMemcpyAsync(mx.data, x.data,
          3 * sizeof(double), cudaMemcpyDeviceToDevice, context.stream()),
          "copy identity overlap image");
    };
    for (double scale : {1e-4, 1.0, 1e4}) {
      const Eigen::MatrixXd scaled = scale * inverse;
      cuda::copy_to_device(context, dp.view(), scaled);
      const cuda::DeviceEigenPreconditioner precondition =
          [&](linalg::ConstMatrixView r, double, linalg::MatrixView p) {
            cuda::gemm(context, p, dp.view(), r);
          };
      auto options = linalg::make_davidson_options(3, 1, 1e-7, 1e-3, 1e-7);
      options.identity_overlap = identity;
      const auto result = cuda::solve_device_davidson(context, action,
          Eigen::VectorXd(h.diagonal()), Eigen::VectorXd::Ones(3),
          initial.view(), options, precondition);
      require(std::abs(result.eigenpairs.eigenvalues[0] + 1.0) < 1e-12 &&
              result.stationarity_residual_norms[0] < 1e-12 &&
              result.block_actions == 2,
          "independent correction rejected after removing large basis component");
    }
  }
  std::cout << "CUDA projected correction scale: PASS\n";
}

}  // namespace

int main() {
  try {
    check_case(false);
    check_case(true);
    check_projected_correction_scale();
    std::cout << "CUDA device-output Davidson parity: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
