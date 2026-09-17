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
int pair_index(int first, int second) {
  return second * (second - 1) / 2 + first;
}

Eigen::MatrixXd minor(const Eigen::MatrixXd& x, int row, int col) {
  Eigen::MatrixXd m(x.rows()-1, x.cols()-1);
  for (int i=0, r=0; i<x.rows(); ++i) if (i != row) {
    for (int j=0, c=0; j<x.cols(); ++j) if (j != col) m(r,c++)=x(i,j);
    ++r;
  }
  return m;
}

/** @brief Evaluates the directional derivative of a determinant by columns. */
double determinant_first_reference(
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& direction) {
  double result = 0.0;
  for (int column = 0; column < x.cols(); ++column) {
    Eigen::MatrixXd replaced = x;
    replaced.col(column) = direction.col(column);
    result += replaced.determinant();
  }
  return result;
}

/**
 * @brief Builds the second cofactor from independently evaluated deleted minors.
 */
Eigen::MatrixXd second_reference(const Eigen::MatrixXd& x) {
  const int n = x.rows();
  const int n_pairs = n * (n - 1) / 2;
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0; column_first < column_second;
             ++column_first) {
          const Eigen::MatrixXd submatrix = minor(
              minor(x, row_second, column_second),
              row_first, column_first);
          const double sign =
              (row_first + row_second + column_first + column_second) % 2
                  ? -1.0
                  : 1.0;
          result(pair_index(row_first, row_second),
                 pair_index(column_first, column_second)) =
              sign * (n == 2 ? 1.0 : submatrix.determinant());
        }
      }
    }
  }
  return result;
}

/**
 * @brief Differentiates every deleted second-cofactor minor independently.
 */
Eigen::MatrixXd second_first_reference(
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& direction) {
  const int n = x.rows();
  const int n_pairs = n * (n - 1) / 2;
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0; column_first < column_second;
             ++column_first) {
          const Eigen::MatrixXd submatrix = minor(
              minor(x, row_second, column_second),
              row_first, column_first);
          const Eigen::MatrixXd subdirection = minor(
              minor(direction, row_second, column_second),
              row_first, column_first);
          const double sign =
              (row_first + row_second + column_first + column_second) % 2
                  ? -1.0
                  : 1.0;
          result(pair_index(row_first, row_second),
                 pair_index(column_first, column_second)) =
              sign * determinant_first_reference(submatrix, subdirection);
        }
      }
    }
  }
  return result;
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

/**
 * @brief Pulls a second-cofactor contraction back to the overlap matrix.
 */
Eigen::MatrixXd second_gradient_reference(
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& weights) {
  const int n = x.rows();
  Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(n, n);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0; column_first < column_second;
             ++column_first) {
          const Eigen::MatrixXd submatrix = minor(
              minor(x, row_second, column_second),
              row_first, column_first);
          const Eigen::MatrixXd subcofactor =
              reference(submatrix, submatrix, submatrix, 0);
          const double sign =
              (row_first + row_second + column_first + column_second) % 2
                  ? -1.0
                  : 1.0;
          const double factor =
              sign * weights(pair_index(row_first, row_second),
                             pair_index(column_first, column_second));
          for (int row = 0, subrow = 0; row < n; ++row) {
            if (row == row_first || row == row_second) {
              continue;
            }
            for (int column = 0, subcolumn = 0; column < n; ++column) {
              if (column == column_first || column == column_second) {
                continue;
              }
              gradient(row, column) +=
                  factor * subcofactor(subrow, subcolumn++);
            }
            ++subrow;
          }
        }
      }
    }
  }
  return gradient;
}

/**
 * @brief Differentiates the pulled-back second-cofactor contraction exactly.
 */
Eigen::MatrixXd second_gradient_direction_reference(
    const Eigen::MatrixXd& x, const Eigen::MatrixXd& direction,
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& delta_weights) {
  const int n = x.rows();
  Eigen::MatrixXd gradient_direction = Eigen::MatrixXd::Zero(n, n);
  for (int row_second = 1; row_second < n; ++row_second) {
    for (int row_first = 0; row_first < row_second; ++row_first) {
      for (int column_second = 1; column_second < n; ++column_second) {
        for (int column_first = 0; column_first < column_second;
             ++column_first) {
          const Eigen::MatrixXd submatrix = minor(
              minor(x, row_second, column_second),
              row_first, column_first);
          const Eigen::MatrixXd subdirection = minor(
              minor(direction, row_second, column_second),
              row_first, column_first);
          const Eigen::MatrixXd subcofactor =
              reference(submatrix, submatrix, submatrix, 0);
          const Eigen::MatrixXd subcofactor_direction =
              reference(submatrix, subdirection, subdirection, 1);
          const double sign =
              (row_first + row_second + column_first + column_second) % 2
                  ? -1.0
                  : 1.0;
          const int row_pair = pair_index(row_first, row_second);
          const int column_pair = pair_index(column_first, column_second);
          for (int row = 0, subrow = 0; row < n; ++row) {
            if (row == row_first || row == row_second) {
              continue;
            }
            for (int column = 0, subcolumn = 0; column < n; ++column) {
              if (column == column_first || column == column_second) {
                continue;
              }
              gradient_direction(row, column) += sign *
                  (weights(row_pair, column_pair) *
                       subcofactor_direction(subrow, subcolumn) +
                   delta_weights(row_pair, column_pair) *
                       subcofactor(subrow, subcolumn));
              ++subcolumn;
            }
            ++subrow;
          }
        }
      }
    }
  }
  return gradient_direction;
}

/** @brief Produces a reproducible dense orthogonal matrix. */
Eigen::MatrixXd orthogonal_matrix(int n, double phase) {
  Eigen::MatrixXd result = Eigen::MatrixXd::Identity(n, n);
  for (int first = 0; first < n; ++first) {
    for (int second = first + 1; second < n; ++second) {
      const double angle = phase + 0.07 * (first + 1) * (second + 2);
      const double cosine = std::cos(angle);
      const double sine = std::sin(angle);
      const Eigen::VectorXd first_column = result.col(first);
      const Eigen::VectorXd second_column = result.col(second);
      result.col(first) = cosine * first_column + sine * second_column;
      result.col(second) = -sine * first_column + cosine * second_column;
    }
  }
  return result;
}

/** @brief Produces reproducible, nonsymmetric derivative test data. */
Eigen::MatrixXd patterned_matrix(int rows, int columns, double phase) {
  Eigen::MatrixXd result(rows, columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      result(row, column) =
          std::sin((row + 1) * (column + 2) + phase) +
          0.25 * std::cos((row + 3) * (column + 1) - phase);
    }
  }
  return result;
}

/** @brief Constructs a full-rank overlap with a controlled dangerous subspace. */
Eigen::MatrixXd ill_conditioned_matrix(
    int n, int dangerous_modes, double condition_number) {
  Eigen::VectorXd singular_values = Eigen::VectorXd::Ones(n);
  const double smallest = 1.0 / condition_number;
  for (int mode = 0; mode < dangerous_modes; ++mode) {
    singular_values(mode) = (mode + 1) * smallest;
  }
  return orthogonal_matrix(n, 0.13) * singular_values.asDiagonal() *
      orthogonal_matrix(n, -0.29).transpose();
}

/** @brief Constructs an overlap with the requested exact numerical nullity. */
Eigen::MatrixXd rank_deficient_matrix(int n, int nullity) {
  Eigen::VectorXd singular_values = Eigen::VectorXd::Ones(n);
  singular_values.head(nullity).setZero();
  return orthogonal_matrix(n, 0.23) * singular_values.asDiagonal() *
      orthogonal_matrix(n, -0.17).transpose();
}

void check_case(
    const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected,
    const char* quantity, const std::string& scenario) {
  const double error = (actual - expected).norm();
  const double tolerance = 2e-11 * std::max(1.0, expected.norm());
  if (error > tolerance) {
    std::ostringstream message;
    message << std::scientific << quantity
            << " disagrees with deleted-minor polynomial reference for "
            << scenario << ", error=" << error
            << ", tolerance=" << tolerance
            << ", reference_norm=" << expected.norm();
    throw std::runtime_error(message.str());
  }
}

void check_scalar(
    double actual, double expected, const char* quantity,
    const std::string& scenario) {
  const double error = std::abs(actual - expected);
  const double tolerance = 2e-11 * std::max(1.0, std::abs(expected));
  if (error > tolerance) {
    std::ostringstream message;
    message << std::scientific << quantity
            << " disagrees with deleted-minor polynomial reference for "
            << scenario << ", error=" << error
            << ", tolerance=" << tolerance
            << ", reference=" << expected;
    throw std::runtime_error(message.str());
  }
}

/**
 * @brief Verifies every cofactor differential operation against deleted minors.
 */
void check_differential_case(
    const xmvb::vb::DeterminantOverlapResolver& overlap_resolver,
    const Eigen::MatrixXd& x, int expected_nullity,
    int expected_dangerous_modes,
    const std::string& scenario) {
  const int n = x.rows();
  const int n_pairs = n * (n - 1) / 2;
  const Eigen::MatrixXd direction_a = patterned_matrix(n, n, 0.31);
  const Eigen::MatrixXd direction_b = patterned_matrix(n, n, -0.47);
  const Eigen::MatrixXd weights =
      patterned_matrix(n_pairs, n_pairs, 0.19);
  const Eigen::MatrixXd delta_weights =
      patterned_matrix(n_pairs, n_pairs, -0.37);
  const auto overlap = overlap_resolver.resolve_matrix(x);
  if (overlap.nullity != expected_nullity) {
    throw std::runtime_error(
        scenario + ": constructed overlap has nullity " +
        std::to_string(overlap.nullity) + ", expected " +
        std::to_string(expected_nullity));
  }
  const xmvb::vb::CofactorDifferential differential(overlap);
  if (!differential.uses_interpolated_form() ||
      differential.dangerous_mode_count() != expected_dangerous_modes) {
    throw std::runtime_error(
        scenario + ": overlap did not select the expected interpolated form");
  }

  check_case(differential.value(), reference(x, direction_a, direction_b, 0),
             "cofactor", scenario);
  check_case(differential.first(direction_a),
             reference(x, direction_a, direction_b, 1),
             "first cofactor direction", scenario);
  check_case(differential.mixed(direction_a, direction_b),
             reference(x, direction_a, direction_b, 2),
             "mixed cofactor direction", scenario);

  const Eigen::MatrixXd expected_second = second_reference(x);
  const Eigen::MatrixXd expected_second_first =
      second_first_reference(x, direction_a);
  const Eigen::MatrixXd expected_gradient =
      second_gradient_reference(x, weights);
  const Eigen::MatrixXd expected_gradient_direction =
      second_gradient_direction_reference(
          x, direction_a, weights, delta_weights);
  check_case(differential.second(), expected_second,
             "second cofactor", scenario);
  check_case(differential.second_first(direction_a), expected_second_first,
             "second cofactor direction", scenario);
  check_scalar(differential.second_contraction(weights),
               (weights.cwiseProduct(expected_second)).sum(),
               "second cofactor contraction", scenario);
  check_case(differential.second_contraction_gradient(weights),
             expected_gradient, "second contraction gradient", scenario);
  check_case(differential.second_contraction_gradient_direction(
                 direction_a, weights, delta_weights),
             expected_gradient_direction,
             "second contraction gradient direction", scenario);

  const double pushforward =
      (differential.second_first(direction_a).cwiseProduct(weights)).sum();
  const double pullback =
      (differential.second_contraction_gradient(weights)
           .cwiseProduct(direction_a))
          .sum();
  const double expected_adjoint =
      (expected_second_first.cwiseProduct(weights)).sum();
  check_scalar(pushforward, expected_adjoint,
               "second cofactor pushforward", scenario);
  check_scalar(pullback, expected_adjoint,
               "second cofactor pullback", scenario);
  check_scalar(pushforward, pullback,
               "second cofactor pushforward/pullback adjoint", scenario);
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
        Eigen::MatrixXd expected_second(m,m), expected_gradient=Eigen::MatrixXd::Zero(n,n);
        for (int r2=1;r2<n;++r2) for (int r1=0;r1<r2;++r1)
          for (int c2=1;c2<n;++c2) for (int c1=0;c1<c2;++c1) {
            const Eigen::MatrixXd sub=minor(minor(x,r2,c2),r1,c1);
            const double sign=(r1+r2+c1+c2)%2 ? -1.0:1.0;
            expected_second(r2*(r2-1)/2+r1,c2*(c2-1)/2+c1)=sign*(n==2 ? 1.0:sub.determinant());
            const Eigen::MatrixXd sub_cofactor=reference(sub,sub,sub,0);
            for (int r=0,ri=0;r<n;++r) if(r!=r1 && r!=r2) {
              for (int col=0,ci=0;col<n;++col) if(col!=c1 && col!=c2)
                expected_gradient(r,col)+=sign*w(r2*(r2-1)/2+r1,c2*(c2-1)/2+c1)*sub_cofactor(ri,ci++);
              ++ri;
            }
          }
        check(c.second(),expected_second, "second", n, small, deficient);
        const double contracted_second = c.second_contraction(w);
        const double contracted_second_reference =
            (expected_second.cwiseProduct(w)).sum();
        if (std::abs(contracted_second - contracted_second_reference) >
            2e-11 * std::max(1.0, std::abs(contracted_second_reference))) {
          throw std::runtime_error(
              "second cofactor contraction disagrees with explicit reference for n=" +
              std::to_string(n) + ", small=" + std::to_string(small) +
              ", deficient=" + std::to_string(deficient));
        }
        check(c.second_contraction_gradient(w),expected_gradient,
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
      for (const double condition_number :
           {1e2, 1e4, 1e6, 1e8, 1e10}) {
        for (int dangerous_modes = 1; dangerous_modes <= 4;
             ++dangerous_modes) {
          std::ostringstream scenario;
          scenario << std::scientific
                   << "full-rank overlap, condition=" << condition_number
                   << ", dangerous_modes=" << dangerous_modes;
          check_differential_case(
              overlap_resolver,
              ill_conditioned_matrix(n, dangerous_modes, condition_number),
              0, dangerous_modes, scenario.str());
        }
      }
      for (int nullity = 1; nullity <= 4; ++nullity) {
        check_differential_case(
            overlap_resolver, rank_deficient_matrix(n, nullity), nullity,
            nullity,
            "rank-deficient overlap, nullity=" + std::to_string(nullity));
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
