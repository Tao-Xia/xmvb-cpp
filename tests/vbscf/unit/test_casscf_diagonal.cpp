#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/optimization/preconditioners/casscf_diagonal.hpp"

namespace {

std::size_t eri_index(int p, int q, int r, int s, int n) {
  return (((static_cast<std::size_t>(p) * n + q) * n + r) * n + s);
}

double eri(
    const std::vector<double>& values,
    int p, int q, int r, int s, int n) {
  return values[eri_index(p, q, r, s, n)];
}

std::vector<double> make_symmetric_eri(int n) {
  std::vector<double> values(static_cast<std::size_t>(n) * n * n * n, 0.0);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q <= p; ++q) {
      const int pq = p * (p + 1) / 2 + q;
      for (int r = 0; r < n; ++r) {
        for (int s = 0; s <= r; ++s) {
          const int rs = r * (r + 1) / 2 + s;
          if (rs > pq) continue;
          const double value =
              0.013 * (1 + pq) - 0.007 * (1 + rs) + 0.0003 * pq * rs;
          const int permutations[8][4] = {
              {p,q,r,s}, {q,p,r,s}, {p,q,s,r}, {q,p,s,r},
              {r,s,p,q}, {s,r,p,q}, {r,s,q,p}, {s,r,q,p}};
          for (const auto& x : permutations) {
            values[eri_index(x[0], x[1], x[2], x[3], n)] = value;
          }
        }
      }
    }
  }
  return values;
}

std::vector<double> rotate_eri(
    const std::vector<double>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& u) {
  const int n = static_cast<int>(u.rows());
  std::vector<double> rotated(static_cast<std::size_t>(n) * n * n * n, 0.0);
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q)
      for (int r = 0; r < n; ++r)
        for (int s = 0; s < n; ++s)
          for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
              for (int c = 0; c < n; ++c)
                for (int d = 0; d < n; ++d)
                  rotated[eri_index(p,q,r,s,n)] +=
                      u(a,p) * u(b,q) * u(c,r) * u(d,s) *
                      eri(values,a,b,c,d,n);
  return rotated;
}

double energy(
    const Eigen::Ref<const Eigen::MatrixXd>& h,
    const std::vector<double>& g,
    const Eigen::Ref<const Eigen::MatrixXd>& dm1,
    const std::vector<double>& dm2,
    int n_core) {
  const int n = static_cast<int>(h.rows());
  const int na = static_cast<int>(dm1.rows());
  double value = 0.0;
  for (int c = 0; c < n_core; ++c) value += 2.0 * h(c,c);
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < na; ++b)
      value += h(n_core+a,n_core+b) * dm1(a,b);
  for (int c = 0; c < n_core; ++c)
    for (int d = 0; d < n_core; ++d)
      value += 2.0 * eri(g,c,c,d,d,n) - eri(g,c,d,d,c,n);
  for (int c = 0; c < n_core; ++c)
    for (int a = 0; a < na; ++a)
      for (int b = 0; b < na; ++b)
        value += dm1(a,b) *
            (2.0 * eri(g,c,c,n_core+a,n_core+b,n) -
             eri(g,c,n_core+a,n_core+b,c,n));
  for (int a = 0; a < na; ++a)
    for (int b = 0; b < na; ++b)
      for (int c = 0; c < na; ++c)
        for (int d = 0; d < na; ++d)
          value += 0.5 * dm2[eri_index(a,b,c,d,na)] *
              eri(g,n_core+a,n_core+b,n_core+c,n_core+d,n);
  return value;
}

xmvb::vb::CasscfDiagonalIntermediates make_intermediates(
    const Eigen::Ref<const Eigen::MatrixXd>& h,
    const std::vector<double>& g,
    int n_core,
    int na) {
  const int n = static_cast<int>(h.rows());
  xmvb::vb::CasscfDiagonalIntermediates x;
  x.n_core = n_core;
  x.n_active = na;
  x.h_core = h;
  x.v_core = Eigen::MatrixXd::Zero(n,n);
  x.j_pc = Eigen::MatrixXd::Zero(n,n_core);
  x.k_pc = Eigen::MatrixXd::Zero(n,n_core);
  x.ppaa.resize(static_cast<std::size_t>(n)*n*na*na);
  x.papa.resize(static_cast<std::size_t>(n)*na*n*na);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      for (int c = 0; c < n_core; ++c) {
        x.v_core(p,q) +=
            2.0 * eri(g,p,q,c,c,n) - eri(g,p,c,c,q,n);
      }
      for (int a = 0; a < na; ++a)
        for (int b = 0; b < na; ++b) {
          x.ppaa[((p*n+q)*na+a)*na+b] =
              eri(g,p,q,n_core+a,n_core+b,n);
          x.papa[((p*na+a)*n+q)*na+b] =
              eri(g,p,n_core+a,q,n_core+b,n);
        }
    }
    for (int c = 0; c < n_core; ++c) {
      x.j_pc(p,c) = eri(g,p,p,c,c,n);
      x.k_pc(p,c) = eri(g,p,c,p,c,n);
    }
  }
  return x;
}

}  // namespace

int main() {
  constexpr int n = 5;
  constexpr int nc = 1;
  constexpr int na = 2;
  Eigen::MatrixXd h(n,n);
  for (int p = 0; p < n; ++p)
    for (int q = 0; q <= p; ++q)
      h(p,q) = h(q,p) = 0.03 * (p+1) - 0.017 * (q+1) + (p==q ? 0.4 : 0.0);
  const std::vector<double> g = make_symmetric_eri(n);
  Eigen::MatrixXd dm1(na,na);
  dm1 << 1.15, -0.21, -0.21, 0.85;
  std::vector<double> dm2 = make_symmetric_eri(na);
  for (double& value : dm2) value *= 2.7;

  const int n_pairs = na*(na+1)/2;
  std::vector<double> packed(static_cast<std::size_t>(n_pairs)*(n_pairs+1)/2);
  for (int p = 0; p < na; ++p)
    for (int q = 0; q <= p; ++q) {
      const int pq = xmvb::vb::TwoElectronIndexer::packed_pair_index(p,q);
      for (int r = 0; r < na; ++r)
        for (int s = 0; s <= r; ++s) {
          const int rs = xmvb::vb::TwoElectronIndexer::packed_pair_index(r,s);
          if (rs > pq) continue;
          const int multiplicity =
              (p==q ? 1 : 2) * (r==s ? 1 : 2) * (pq==rs ? 1 : 2);
          packed[xmvb::vb::TwoElectronIndexer::packed_pair_of_pairs_index(pq,rs)] =
              0.5 * multiplicity * dm2[eri_index(p,q,r,s,na)];
        }
    }

  const auto x = make_intermediates(h,g,nc,na);
  const Eigen::MatrixXd diagonal =
      xmvb::vb::build_casscf_orbital_hessian_diagonal(x,dm1,packed);
  const double step = 2.0e-4;
  const double e0 = energy(h,g,dm1,dm2,nc);
  double max_error = 0.0;
  for (int q = 0; q < nc+na; ++q) {
    for (int p = q+1; p < n; ++p) {
      if ((p < nc && q < nc) || (p < nc+na && q >= nc)) continue;
      Eigen::MatrixXd up = Eigen::MatrixXd::Identity(n,n);
      Eigen::MatrixXd um = up;
      const double c = std::cos(step);
      const double s = std::sin(step);
      up(p,p)=up(q,q)=c; up(p,q)=-s; up(q,p)=s;
      um(p,p)=um(q,q)=c; um(p,q)=s; um(q,p)=-s;
      const double ep = energy(up.transpose()*h*up,rotate_eri(g,up),dm1,dm2,nc);
      const double em = energy(um.transpose()*h*um,rotate_eri(g,um),dm1,dm2,nc);
      const double fd = (ep - 2.0*e0 + em)/(step*step);
      max_error = std::max(max_error,std::abs(fd-diagonal(p,q)));
    }
  }
  if (max_error > 2.0e-6) {
    std::cerr << "CASSCF diagonal finite-difference error = " << max_error << '\n';
    return 1;
  }
  std::cout << "CASSCF diagonal finite-difference error = " << max_error << '\n';
  return 0;
}
