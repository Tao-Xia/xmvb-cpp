#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "c12_fixture.hpp"
#include "cuda/vb/ci.hpp"

namespace xmvb::cuda::test {

/** Read an archived column-major FP64 matrix, not a complete carrier. */
inline Eigen::MatrixXd read_frozen_ci_matrix(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  std::int64_t rows = 0, columns = 0;
  input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char*>(&columns), sizeof(columns));
  if (!input || rows <= 0 || columns <= 0 || rows > 10000000 / columns)
    throw std::runtime_error("invalid frozen CI matrix dimensions: " + path);
  Eigen::MatrixXd matrix(rows, columns);
  input.read(reinterpret_cast<char*>(matrix.data()), matrix.size() * sizeof(double));
  if (!input || !matrix.allFinite())
    throw std::runtime_error("invalid frozen CI matrix payload: " + path);
  return matrix;
}

/** Replay a real frozen CI point with physically equivalent seed scalings. */
inline void check_frozen_ci(C12Fixture& fixture, const std::string& prefix) {
  const auto upper = read_frozen_ci_matrix(prefix + "-upper.bin");
  const auto one = read_frozen_ci_matrix(prefix + "-one.bin");
  const auto pair = read_frozen_ci_matrix(prefix + "-pair.bin");
  const auto guess = read_frozen_ci_matrix(prefix + "-guess.bin");
  Context context(4ULL * 1024 * 1024 * 1024);
  StructureAction action(context, *fixture.action);
  Matrix<double> u(context, upper.rows(), upper.cols());
  Matrix<double> h(context, one.rows(), one.cols());
  Matrix<double> g(context, pair.rows(), pair.cols());
  Matrix<double> seed(context, guess.rows(), guess.cols());
  copy_to_device(context, u.view(), upper);
  copy_to_device(context, h.view(), one);
  copy_to_device(context, g.view(), pair);
  action.update_integrals(h.view(), g.view(), u.view());
  SpinMetric metric(context, *fixture.action,
      fixture.topology.alpha_reuse_table.unique_determinants,
      fixture.topology.beta_reuse_table.unique_determinants,
      context.device_byte_budget());
  StructureCi solver(context, action, metric, fixture.loaded.input.complete_active_space);
  const auto options = linalg::make_davidson_options(action.size(), 1, 1e-7, 1e-3, 1e-7);
  double reference = 0.0;
  for (int trial = -1; trial <= 1; ++trial) {
    const Eigen::MatrixXd scaled = (1.0 + std::ldexp(double(trial), -20)) * guess;
    copy_to_device(context, seed.view(), scaled);
    const auto result = solver.solve_device(seed.view(), options);
    const double energy = result.eigenpairs.eigenvalues[0];
    if (trial == -1) reference = energy;
    if (std::abs(energy - reference) > 1e-10 ||
        result.stationarity_residual_norms[0] > options.stationarity_tolerance)
      throw std::runtime_error("frozen CI seed scaling changed the converged result");
    std::cout << "frozen_ci_result=" << energy << ',' << result.iterations << ','
              << result.block_actions << ',' << result.stationarity_residual_norms[0] << '\n';
  }
}

}  // namespace xmvb::cuda::test
