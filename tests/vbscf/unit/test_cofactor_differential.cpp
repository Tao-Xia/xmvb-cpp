#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <Eigen/LU>
#include <Eigen/QR>
#include "vbscf/determinants/algebra/cofactor_differential.hpp"
#include "vbscf/determinants/algebra/overlap.hpp"

namespace {
Eigen::MatrixXd minor(const Eigen::MatrixXd& x, int row, int col) {
  Eigen::MatrixXd m(x.rows()-1, x.cols()-1);
  for (int i=0, r=0; i<x.rows(); ++i) if (i != row) {
    for (int j=0, c=0; j<x.cols(); ++j) if (j != col) m(r,c++)=x(i,j);
    ++r;
  }
  return m;
}

// Independent polynomial reference: multilinearity in columns, evaluated by
// determinants of replaced-column minors. No SVD or inverse identities.
Eigen::MatrixXd reference(const Eigen::MatrixXd& x, const Eigen::MatrixXd& a,
                          const Eigen::MatrixXd& b, int order) {
  Eigen::MatrixXd c=Eigen::MatrixXd::Zero(x.rows(),x.cols());
  for (int i=0;i<x.rows();++i) for (int j=0;j<x.cols();++j) {
    auto m=minor(x,i,j), da=minor(a,i,j), db=minor(b,i,j);
    double v=0;
    if (order==0) v=m.rows()==0 ? 1.0 : m.determinant();
    else for (int k=0;k<m.cols();++k) {
      Eigen::MatrixXd replaced=m; replaced.col(k)=da.col(k);
      if (order==1) v+=replaced.determinant();
      else for (int l=0;l<m.cols();++l) if (l!=k) {
        Eigen::MatrixXd twice=replaced; twice.col(l)=db.col(l);
        v+=twice.determinant();
      }
    }
    c(i,j)=((i+j)%2 ? -v:v);
  }
  return c;
}

void check(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b,
    const char* quantity,
    int n,
    double small,
    int deficient) {
  if ((a-b).norm() > 2e-11*std::max(1.0,b.norm()))
  {
    std::ostringstream message;
    message << std::scientific << quantity
            << " disagrees with polynomial reference for n=" << n
            << ", small=" << small
            << ", deficient=" << deficient
            << ", error=" << (a - b).norm()
            << ", reference_norm=" << b.norm();
    throw std::runtime_error(message.str());
  }
}
}

int main() {
  try {
    const xmvb::vb::DeterminantOverlapResolver overlap_resolver;
    for (int n=0;n<=7;++n) {
      Eigen::MatrixXd a=Eigen::MatrixXd::Random(n,n), b=Eigen::MatrixXd::Random(n,n);
      Eigen::MatrixXd u=Eigen::MatrixXd::Identity(n,n), v=u;
      if (n) {
        u=Eigen::MatrixXd(Eigen::MatrixXd::Random(n,n).householderQr().householderQ());
        v=Eigen::MatrixXd(Eigen::MatrixXd::Random(n,n).householderQr().householderQ());
      }
      for (double small : {1.0, 1e-4, 1e-12, 0.0}) for (int deficient=1;deficient<=4;++deficient) {
        Eigen::VectorXd s=Eigen::VectorXd::Ones(n);
        for (int i=0;i<std::min(n,deficient);++i) s(i)=small;
        const Eigen::MatrixXd x=u*s.asDiagonal()*v.transpose();
        xmvb::vb::CofactorDifferential c(
            overlap_resolver.resolve_matrix(x));
        check(c.value(),reference(x,a,b,0), "cofactor", n, small, deficient);
        check(c.first(a),reference(x,a,b,1), "first", n, small, deficient);
        check(c.mixed(a,b),reference(x,a,b,2), "mixed", n, small, deficient);
        check(c.first(a+b),c.first(a)+c.first(b), "first linearity", n, small, deficient);
        check(c.mixed(a,b),c.mixed(b,a), "mixed symmetry", n, small, deficient);
        const int m=n*(n-1)/2;
        const Eigen::MatrixXd w=Eigen::MatrixXd::Random(m,m), dw=Eigen::MatrixXd::Random(m,m);
        Eigen::MatrixXd second_reference(m,m), gradient_reference=Eigen::MatrixXd::Zero(n,n);
        for (int r2=1;r2<n;++r2) for (int r1=0;r1<r2;++r1)
          for (int c2=1;c2<n;++c2) for (int c1=0;c1<c2;++c1) {
            const Eigen::MatrixXd sub=minor(minor(x,r2,c2),r1,c1);
            const double sign=(r1+r2+c1+c2)%2 ? -1.0:1.0;
            second_reference(r2*(r2-1)/2+r1,c2*(c2-1)/2+c1)=sign*(n==2 ? 1.0:sub.determinant());
            const Eigen::MatrixXd sub_cofactor=reference(sub,sub,sub,0);
            for (int r=0,ri=0;r<n;++r) if(r!=r1 && r!=r2) {
              for (int col=0,ci=0;col<n;++col) if(col!=c1 && col!=c2)
                gradient_reference(r,col)+=sign*w(r2*(r2-1)/2+r1,c2*(c2-1)/2+c1)*sub_cofactor(ri,ci++);
              ++ri;
            }
          }
        check(c.second(),second_reference, "second", n, small, deficient);
        const double contracted_second = c.second_contraction(w);
        const double contracted_second_reference =
            (second_reference.cwiseProduct(w)).sum();
        if (std::abs(contracted_second - contracted_second_reference) >
            2e-11 * std::max(1.0, std::abs(contracted_second_reference))) {
          throw std::runtime_error(
              "second cofactor contraction disagrees with explicit reference for n=" +
              std::to_string(n) + ", small=" + std::to_string(small) +
              ", deficient=" + std::to_string(deficient));
        }
        check(c.second_contraction_gradient(w),gradient_reference,
              "second gradient", n, small, deficient);
        const double step=1e-5;
        xmvb::vb::CofactorDifferential plus(
            overlap_resolver.resolve_matrix(x + step * a));
        xmvb::vb::CofactorDifferential minus(
            overlap_resolver.resolve_matrix(x - step * a));
        const auto dg=c.second_contraction_gradient_direction(a,w,dw);
        const Eigen::MatrixXd fd=(plus.second_contraction_gradient(w+step*dw)-
                                  minus.second_contraction_gradient(w-step*dw))/(2*step);
        if ((dg-fd).norm()>1e-7*std::max(1.0,dg.norm()))
          throw std::runtime_error(
              "contracted second-cofactor gradient direction fails finite differences for n=" +
              std::to_string(n) + ", small=" + std::to_string(small) +
              ", deficient=" + std::to_string(deficient) +
              ", error=" + std::to_string((dg - fd).norm()));
        check(c.second_contraction_gradient_direction(a,w,dw),
              c.second_contraction_gradient_direction(a,w,Eigen::MatrixXd::Zero(m,m))+
              c.second_contraction_gradient(dw),
              "second gradient direction linearity", n, small, deficient);
        const double slope=(c.second_first(a).cwiseProduct(w)).sum();
        const double adjoint=(c.second_contraction_gradient(w).cwiseProduct(a)).sum();
        if (std::abs(slope-adjoint)>2e-11*std::max(1.0,std::abs(slope)))
          throw std::runtime_error(
              "second cofactor pushforward/pullback mismatch for n=" +
              std::to_string(n) + ", small=" + std::to_string(small) +
              ", deficient=" + std::to_string(deficient));
        const Eigen::MatrixXd dc_fd=(plus.second()-minus.second())/(2*step);
        if ((dc_fd-c.second_first(a)).norm()>1e-7*std::max(1.0,dc_fd.norm()))
          throw std::runtime_error(
              "second cofactor direction fails finite differences for n=" +
              std::to_string(n) + ", small=" + std::to_string(small) +
              ", deficient=" + std::to_string(deficient));
      }
    }
    {
      constexpr int n = 6;
      const Eigen::MatrixXd x = Eigen::MatrixXd::Identity(n, n);
      const xmvb::vb::CofactorDifferential regular(
          overlap_resolver.resolve_matrix(x));
      const std::size_t expected_bytes =
          2 * n * n * sizeof(double);
      if (regular.dynamic_bytes() != expected_bytes) {
        throw std::runtime_error(
            "regular cofactor representation retained compound matrices");
      }
    }
    std::cout << "Cofactor value, first and mixed derivatives: polynomial reference, near-singular and rank-deficient tests passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
