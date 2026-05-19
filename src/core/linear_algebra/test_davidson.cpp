#include "core/linear_algebra/generalized_eigensolver.hpp"
#include <cmath>
#include <iostream>
#include <Eigen/Core>

namespace {
// Tridiagonal: H = tridiag(-1,2,-1), S = I.  Well-separated eigenvalues.
// This is a demanding test for Davidson because the eigenvalue density
// increases with n, but the gaps are known analytically.
std::pair<std::vector<double>, std::vector<double>> make_test(int n) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    H(i, i) = 2.0;
    if (i > 0) H(i, i-1) = H(i-1, i) = -1.0;
  }
  Eigen::MatrixXd S = Eigen::MatrixXd::Identity(n, n);
  std::vector<double> Hv(n*n), Sv(n*n);
  Eigen::Map<Eigen::MatrixXd>(Hv.data(), n, n) = H;
  Eigen::Map<Eigen::MatrixXd>(Sv.data(), n, n) = S;
  return {Hv, Sv};
}

double max_ev_err(const std::vector<double>& a, const std::vector<double>& b, int k) {
  double e = 0.0;
  for (int i = 0; i < k; ++i) e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}

double max_evec_err(const std::vector<double>& v1, const std::vector<double>& v2, int n, int k) {
  double worst = 1.0;
  for (int j = 0; j < k; ++j) {
    Eigen::Map<const Eigen::VectorXd> c1(v1.data() + static_cast<std::size_t>(j) * n, n);
    Eigen::Map<const Eigen::VectorXd> c2(v2.data() + static_cast<std::size_t>(j) * n, n);
    worst = std::min(worst, std::abs(c1.dot(c2)));
  }
  return 1.0 - worst;
}
}  // namespace

int main() {
  xmvb::core::GeneralizedEigensolver solver;
  bool ok = true;

  for (int n : {80, 120, 200, 400, 600}) {
    std::cout << "n=" << n << "..." << std::flush;
    auto [Hv, Sv] = make_test(n);
    auto ref = solver.solve(Hv, Sv, n);

    for (int n_roots : {1, 3, 5, 8}) {
      if (n_roots >= n) continue;
      auto dav = solver.solve_davidson(Hv, Sv, n, n_roots);
      double ev_err = max_ev_err(ref.eigenvalues, dav.eigenvalues, n_roots);
      double evec_err = max_evec_err(ref.eigenvector_matrix, dav.eigenvector_matrix, n, n_roots);
      if (ev_err > 1e-6 || evec_err > 1e-4) {
        std::cout << " FAIL(nr=" << n_roots << " ev=" << ev_err << " evec=" << evec_err << ")";
        ok = false;
      }
    }
    if (ok) std::cout << " PASS\n"; else std::cout << "\n";
  }
  std::cout << (ok ? "ALL PASSED" : "SOME FAILED") << std::endl;
  return ok ? 0 : 1;
}