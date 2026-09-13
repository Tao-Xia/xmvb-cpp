#include "vbscf/integrals/ao/one_electron/direct_operator.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb {
namespace {

struct H1eTerms {
  int ij;
  int kl;
  int ik;
  int jl;
  int il;
  int jk;
  int lj;
  int ki;
  int kj;
  int li;
  double value;
};

std::size_t matrix_size(const AoIntegralInput& ao) {
  const int n_bf = ao.n_basis_functions;
  if (n_bf <= 0 ||
      ao.ao_two_electron_integral_indices.size() !=
          ao.ao_two_electron_integral_values.size() * 4) {
    throw std::invalid_argument("invalid AO-H1E integral input");
  }
  const std::size_t n = n_bf;
  return n * n;
}

int matrix_index(int row, int column, int n_bf) {
  return column * n_bf + row;
}

H1eTerms make_terms(const AoIntegralInput& ao, std::size_t eri_index) {
  const int* eri = ao.ao_two_electron_integral_indices.data() + 4 * eri_index;
  const int n_bf = ao.n_basis_functions;
  const int i = eri[0];
  const int j = eri[1];
  const int k = eri[2];
  const int l = eri[3];
  double value = ao.ao_two_electron_integral_values[eri_index];
  if (i == j) {
    value *= 0.5;
  }
  if (k == l) {
    value *= 0.5;
  }
  if (i == k && j == l) {
    value *= 0.5;
  }

  return {
      .ij = matrix_index(i, j, n_bf),
      .kl = matrix_index(k, l, n_bf),
      .ik = matrix_index(i, k, n_bf),
      .jl = matrix_index(j, l, n_bf),
      .il = matrix_index(i, l, n_bf),
      .jk = matrix_index(j, k, n_bf),
      .lj = matrix_index(l, j, n_bf),
      .ki = matrix_index(k, i, n_bf),
      .kj = matrix_index(k, j, n_bf),
      .li = matrix_index(l, i, n_bf),
      .value = value,
  };
}

inline void accumulate_forward(
    const H1eTerms& t,
    const double* source,
    double* target) {
  const double coulomb = 4.0 * t.value;
  target[t.ij] += coulomb * source[t.kl];
  target[t.kl] += coulomb * source[t.ij];
  target[t.ik] -= t.value * source[t.lj];
  target[t.jl] -= t.value * source[t.ki];
  target[t.il] -= t.value * source[t.kj];
  target[t.jk] -= t.value * source[t.li];
}

inline void accumulate_transpose(
    const H1eTerms& t,
    const double* adjoint,
    double* target) {
  const double coulomb = 4.0 * t.value;
  target[t.kl] += coulomb * adjoint[t.ij];
  target[t.ij] += coulomb * adjoint[t.kl];
  target[t.lj] -= t.value * adjoint[t.ik];
  target[t.ki] -= t.value * adjoint[t.jl];
  target[t.kj] -= t.value * adjoint[t.il];
  target[t.li] -= t.value * adjoint[t.jk];
}

int active_threads(int requested, std::size_t n_integrals) {
  if (requested <= 0) {
    throw std::invalid_argument("AO-H1E thread count must be positive");
  }
  if (n_integrals == 0) {
    return 1;
  }
  return std::min(requested, static_cast<int>(n_integrals));
}

void reduce(
    std::vector<std::vector<double>> partial,
    std::vector<double>* result) {
  if (partial.size() == 1) {
    *result = std::move(partial.front());
    return;
  }
  result->assign(partial.front().size(), 0.0);
  for (const auto& local : partial) {
    for (std::size_t index = 0; index < result->size(); ++index) {
      (*result)[index] += local[index];
    }
  }
}

}  // namespace

std::vector<double> apply_ao_h1e(
    const double* source,
    const AoIntegralInput& ao,
    int n_threads) {
  if (source == nullptr) {
    throw std::invalid_argument("AO-H1E source must not be null");
  }
  const std::size_t size = matrix_size(ao);
  const std::size_t n_integrals = ao.ao_two_electron_integral_values.size();
  n_threads = active_threads(n_threads, n_integrals);
  std::vector<std::vector<double>> partial(
      n_threads, std::vector<double>(size, 0.0));
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    auto& local = partial[thread];
#pragma omp for schedule(static)
    for (std::ptrdiff_t offset = 0;
         offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++offset) {
      accumulate_forward(make_terms(ao, offset), source, local.data());
    }
  }
  std::vector<double> result;
  reduce(std::move(partial), &result);
  return result;
}

std::vector<double> apply_ao_h1e_transpose(
    const double* adjoint,
    const AoIntegralInput& ao,
    int n_threads) {
  if (adjoint == nullptr) {
    throw std::invalid_argument("AO-H1E adjoint must not be null");
  }
  const std::size_t size = matrix_size(ao);
  const std::size_t n_integrals = ao.ao_two_electron_integral_values.size();
  n_threads = active_threads(n_threads, n_integrals);
  std::vector<std::vector<double>> partial(
      n_threads, std::vector<double>(size, 0.0));
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    auto& local = partial[thread];
#pragma omp for schedule(static)
    for (std::ptrdiff_t offset = 0;
         offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++offset) {
      accumulate_transpose(make_terms(ao, offset), adjoint, local.data());
    }
  }
  std::vector<double> result;
  reduce(std::move(partial), &result);
  return result;
}

void apply_ao_h1e_fused(
    const double* source,
    const double* adjoint,
    const AoIntegralInput& ao,
    int n_threads,
    AoH1eFusedWorkspace* workspace,
    std::vector<double>* forward,
    std::vector<double>* transpose) {
  if (source == nullptr || adjoint == nullptr ||
      workspace == nullptr || forward == nullptr || transpose == nullptr) {
    throw std::invalid_argument("AO-H1E fused buffers must not be null");
  }
  const std::size_t size = matrix_size(ao);
  const std::size_t n_integrals = ao.ao_two_electron_integral_values.size();
  n_threads = active_threads(n_threads, n_integrals);
  workspace->forward.resize(n_threads);
  workspace->transpose.resize(n_threads);
  for (int thread = 0; thread < n_threads; ++thread) {
    workspace->forward[thread].assign(size, 0.0);
    workspace->transpose[thread].assign(size, 0.0);
  }
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    auto& local_forward = workspace->forward[thread];
    auto& local_transpose = workspace->transpose[thread];
#pragma omp for schedule(static)
    for (std::ptrdiff_t offset = 0;
         offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++offset) {
      const H1eTerms terms = make_terms(ao, offset);
      accumulate_forward(terms, source, local_forward.data());
      accumulate_transpose(terms, adjoint, local_transpose.data());
    }
  }
  forward->assign(size, 0.0);
  transpose->assign(size, 0.0);
  for (int thread = 0; thread < n_threads; ++thread) {
    const auto& local_forward = workspace->forward[thread];
    const auto& local_transpose = workspace->transpose[thread];
    for (std::size_t index = 0; index < size; ++index) {
      (*forward)[index] += local_forward[index];
      (*transpose)[index] += local_transpose[index];
    }
  }
}

void apply_ao_h1e_fused_batch(
    const Eigen::Ref<const Eigen::MatrixXd>& sources,
    const Eigen::Ref<const Eigen::MatrixXd>& adjoints,
    const AoIntegralInput& ao,
    int n_threads,
    Eigen::MatrixXd* forward,
    Eigen::MatrixXd* transpose) {
  if (forward == nullptr || transpose == nullptr) {
    throw std::invalid_argument("AO-H1E batch outputs must not be null");
  }
  const std::size_t size = matrix_size(ao);
  if (sources.rows() != static_cast<Eigen::Index>(size) ||
      adjoints.rows() != static_cast<Eigen::Index>(size) ||
      sources.cols() != adjoints.cols()) {
    throw std::invalid_argument("AO-H1E batch input shape mismatch");
  }
  const std::size_t n_integrals = ao.ao_two_electron_integral_values.size();
  n_threads = active_threads(n_threads, n_integrals);
  const Eigen::Index n_directions = sources.cols();
  using RowMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const RowMatrix source_rows = sources;
  const RowMatrix adjoint_rows = adjoints;
  std::vector<RowMatrix> partial_forward(
      n_threads, RowMatrix::Zero(size, n_directions));
  std::vector<RowMatrix> partial_transpose(
      n_threads, RowMatrix::Zero(size, n_directions));
#pragma omp parallel num_threads(n_threads)
  {
    int thread = 0;
#ifdef _OPENMP
    thread = omp_get_thread_num();
#endif
    RowMatrix& local_forward = partial_forward[thread];
    RowMatrix& local_transpose = partial_transpose[thread];
#pragma omp for schedule(static)
    for (std::ptrdiff_t offset = 0;
         offset < static_cast<std::ptrdiff_t>(n_integrals);
         ++offset) {
      const H1eTerms t = make_terms(ao, offset);
      const double coulomb = 4.0 * t.value;
#pragma omp simd
      for (Eigen::Index direction = 0;
           direction < n_directions;
           ++direction) {
        local_forward(t.ij, direction) +=
            coulomb * source_rows(t.kl, direction);
        local_forward(t.kl, direction) +=
            coulomb * source_rows(t.ij, direction);
        local_forward(t.ik, direction) -=
            t.value * source_rows(t.lj, direction);
        local_forward(t.jl, direction) -=
            t.value * source_rows(t.ki, direction);
        local_forward(t.il, direction) -=
            t.value * source_rows(t.kj, direction);
        local_forward(t.jk, direction) -=
            t.value * source_rows(t.li, direction);
        local_transpose(t.kl, direction) +=
            coulomb * adjoint_rows(t.ij, direction);
        local_transpose(t.ij, direction) +=
            coulomb * adjoint_rows(t.kl, direction);
        local_transpose(t.lj, direction) -=
            t.value * adjoint_rows(t.ik, direction);
        local_transpose(t.ki, direction) -=
            t.value * adjoint_rows(t.jl, direction);
        local_transpose(t.kj, direction) -=
            t.value * adjoint_rows(t.il, direction);
        local_transpose(t.li, direction) -=
            t.value * adjoint_rows(t.jk, direction);
      }
    }
  }
  RowMatrix sum_forward = RowMatrix::Zero(size, n_directions);
  RowMatrix sum_transpose = RowMatrix::Zero(size, n_directions);
  for (int thread = 0; thread < n_threads; ++thread) {
    sum_forward += partial_forward[thread];
    sum_transpose += partial_transpose[thread];
  }
  *forward = sum_forward;
  *transpose = sum_transpose;
}

}  // namespace xmvb::vb
