#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "pfaffian_vbscf/math/antisymm_codec.hpp"
#include "pfaffian_vbscf/scf/pf_orbital_grad_eval.hpp"
#include "pfaffian_vbscf/scf/pf_scf_eval.hpp"
#include "pfaffian_vbscf/types/pf_basis_data.hpp"
#include "runtime/cpp_vb_input_loader.hpp"
#include "vb/scf/cpp_orbital_gradient_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/vbscf_algorithm.hpp"

namespace {

struct Options {
  std::string input_path;
  int k = 3;
  int seed = 20260328;
  double init_step = 1.0;
  double armijo = 1.0e-4;
  int max_backtracks = 12;
};

struct StepResult {
  bool accepted = false;
  double step = 0.0;
  double trial_energy = 0.0;
  int backtracks = 0;
  xmvb::vb::CppVbInput trial_input;
};

/**
 * @brief Prints tool usage.
 */
void print_usage() {
  std::cerr << "usage: compare_pf_one_step <input.xmi> "
               "[--k K] [--seed S] [--init-step a] [--armijo c] [--max-backtracks n]\n";
}

/**
 * @brief Parses command-line options.
 */
Options parse_args(int argc, char** argv) {
  if (argc < 2 || ((argc - 2) % 2 != 0)) {
    print_usage();
    throw std::invalid_argument("invalid argument count");
  }

  Options opt;
  opt.input_path = argv[1];
  for (int i = 2; i < argc; i += 2) {
    const std::string name = argv[i];
    const std::string value = argv[i + 1];
    if (name == "--k") {
      opt.k = std::stoi(value);
      continue;
    }
    if (name == "--seed") {
      opt.seed = std::stoi(value);
      continue;
    }
    if (name == "--init-step") {
      opt.init_step = std::stod(value);
      continue;
    }
    if (name == "--armijo") {
      opt.armijo = std::stod(value);
      continue;
    }
    if (name == "--max-backtracks") {
      opt.max_backtracks = std::stoi(value);
      continue;
    }
    throw std::invalid_argument("unknown argument: " + name);
  }

  if (opt.k <= 0) {
    throw std::invalid_argument("--k must be positive");
  }
  if (opt.init_step <= 0.0) {
    throw std::invalid_argument("--init-step must be positive");
  }
  if (opt.armijo <= 0.0 || opt.armijo >= 1.0) {
    throw std::invalid_argument("--armijo must lie in (0, 1)");
  }
  if (opt.max_backtracks <= 0) {
    throw std::invalid_argument("--max-backtracks must be positive");
  }
  return opt;
}

int get_sparse_coefficient_count(
    const xmvb::vb::OrbitalPreparationInput& input,
    int orbital_index) {
  const int n_basis_functions = input.n_basis_functions;
  const int explicit_count =
      input.orbital_basis_counts[orbital_index];
  if (explicit_count > 1) {
    return explicit_count;
  }

  int count = 0;
  while (count < n_basis_functions) {
    const int basis_index =
        input.orbital_basis_index_table[
            orbital_index * n_basis_functions + count];
    if (basis_index == 0) {
      break;
    }
    ++count;
  }
  return count;
}

std::vector<int> collect_differentiable_parameter_indices(
    const xmvb::vb::OrbitalPreparationInput& input) {
  std::vector<int> out;
  for (int orbital = 0; orbital < input.n_orbitals; ++orbital) {
    const int count = get_sparse_coefficient_count(input, orbital);
    for (int coef = 0; coef < count; ++coef) {
      out.push_back(orbital * input.n_basis_functions + coef);
    }
  }
  return out;
}

xmvb::pfaffian_vbscf::PfBasisData make_basis(
    int k,
    int seed,
    int n_act,
    int n_alpha,
    int n_beta) {
  xmvb::pfaffian_vbscf::PfBasisData basis;
  basis.n_states = k;
  basis.n_active_orbitals = n_act;
  basis.n_alpha = n_alpha;
  basis.n_beta = n_beta;
  basis.states.resize(k);

  const int n_spin = 2 * n_act;
  const int n_param = xmvb::pfaffian_vbscf::packed_antisymm_size(n_spin);
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int state = 0; state < k; ++state) {
    auto& st = basis.states[state];
    st.n_active_orbitals = n_act;
    st.n_spin_orbitals = n_spin;
    st.packed_entries.resize(n_param, 0.0);
    double norm2 = 0.0;
    for (double& value : st.packed_entries) {
      value = dist(gen);
      norm2 += value * value;
    }
    const double norm = std::sqrt(std::max(1.0e-30, norm2));
    for (double& value : st.packed_entries) {
      value /= norm;
    }
  }
  return basis;
}

Eigen::VectorXd gather_gradient(
    const std::vector<double>& grad,
    const std::vector<int>& diff_idx) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(diff_idx.size()));
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    out[i] = grad[diff_idx[i]];
  }
  return out;
}

double inf_norm(const Eigen::VectorXd& vec) {
  double norm = 0.0;
  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    norm = std::max(norm, std::abs(vec[i]));
  }
  return norm;
}

double cosine_similarity(
    const Eigen::VectorXd& lhs,
    const Eigen::VectorXd& rhs) {
  const double lhs_norm = lhs.norm();
  const double rhs_norm = rhs.norm();
  if (lhs_norm == 0.0 || rhs_norm == 0.0) {
    return 0.0;
  }
  return lhs.dot(rhs) / (lhs_norm * rhs_norm);
}

xmvb::vb::CppVbInput stepped_input(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& diff_idx,
    const Eigen::VectorXd& direction,
    double step) {
  xmvb::vb::CppVbInput trial = input;
  for (Eigen::Index i = 0; i < direction.size(); ++i) {
    const int param_idx = diff_idx[i];
    trial.orbital_preparation_input.orbital_value_table[param_idx] +=
        step * direction[i];
  }
  return trial;
}

double eval_det_energy(
    const xmvb::vb::CppVbInput& input,
    double e_nuc) {
  xmvb::vb::CppVbScfEvaluator eval;
  return eval.evaluate(input, e_nuc).total_energy;
}

double eval_pf_energy(
    const xmvb::vb::CppVbInput& input,
    const xmvb::pfaffian_vbscf::PfBasisData& basis,
    double e_nuc) {
  xmvb::pfaffian_vbscf::PfScfEval eval;
  return eval.eval(input, basis, e_nuc).e_tot;
}

template <typename EvalFn>
StepResult backtracking_step(
    const xmvb::vb::CppVbInput& input,
    const std::vector<int>& diff_idx,
    const Eigen::VectorXd& grad,
    double energy0,
    double init_step,
    double armijo,
    int max_backtracks,
    EvalFn&& eval_energy) {
  StepResult best;
  best.trial_input = input;
  const Eigen::VectorXd direction = -grad;
  const double gtd = grad.dot(direction);

  double step = init_step;
  for (int bt = 0; bt < max_backtracks; ++bt) {
    xmvb::vb::CppVbInput trial = stepped_input(input, diff_idx, direction, step);
    const double trial_energy = eval_energy(trial);
    if (std::isfinite(trial_energy) &&
        trial_energy <= energy0 + armijo * step * gtd) {
      best.accepted = true;
      best.step = step;
      best.trial_energy = trial_energy;
      best.backtracks = bt;
      best.trial_input = std::move(trial);
      return best;
    }
    if (!best.accepted || trial_energy < best.trial_energy) {
      best.trial_energy = trial_energy;
      best.step = step;
      best.backtracks = bt;
      best.trial_input = std::move(trial);
    }
    step *= 0.5;
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opt = parse_args(argc, argv);
    const auto load = xmvb::vb::load_cpp_vb_input_with_timings(opt.input_path);
    const int n_act = load.input.orbital_preparation_input.n_active_orbitals;
    if (load.input.structure_data.alpha_det.empty() ||
        load.input.structure_data.beta_det.empty()) {
      throw std::runtime_error("determinant list is empty");
    }

    const int n_alpha =
        static_cast<int>(load.input.structure_data.alpha_det.front().size());
    const int n_beta =
        static_cast<int>(load.input.structure_data.beta_det.front().size());
    const auto basis = make_basis(opt.k, opt.seed, n_act, n_alpha, n_beta);
    const auto diff_idx =
        collect_differentiable_parameter_indices(load.input.orbital_preparation_input);

    xmvb::vb::CppOrbitalGradientEvaluator det_grad_eval;
    const auto det_grad_result =
        det_grad_eval.evaluate_without_reference_energy_gradient(
            load.input,
            load.nuclear_repulsion_energy);
    const Eigen::VectorXd det_grad =
        gather_gradient(det_grad_result.sparse_orbital_energy_gradient, diff_idx);
    const double det_e0 = det_grad_result.scf_result.total_energy;

    xmvb::pfaffian_vbscf::PfOrbitalGradEval pf_grad_eval;
    const auto pf_grad_result =
        pf_grad_eval.eval(load.input, basis, load.nuclear_repulsion_energy);
    const Eigen::VectorXd pf_grad =
        gather_gradient(pf_grad_result.sparse_orbital_energy_gradient, diff_idx);
    const double pf_e0 = pf_grad_result.scf_result.e_tot;

    const auto det_step = backtracking_step(
        load.input,
        diff_idx,
        det_grad,
        det_e0,
        opt.init_step,
        opt.armijo,
        opt.max_backtracks,
        [&](const xmvb::vb::CppVbInput& trial) {
          return eval_det_energy(trial, load.nuclear_repulsion_energy);
        });
    const auto pf_step = backtracking_step(
        load.input,
        diff_idx,
        pf_grad,
        pf_e0,
        opt.init_step,
        opt.armijo,
        opt.max_backtracks,
        [&](const xmvb::vb::CppVbInput& trial) {
          return eval_pf_energy(trial, basis, load.nuclear_repulsion_energy);
        });

    const double det_energy_after_pf =
        eval_det_energy(pf_step.trial_input, load.nuclear_repulsion_energy);
    const double pf_energy_after_det =
        eval_pf_energy(det_step.trial_input, basis, load.nuclear_repulsion_energy);

    std::cout << std::setprecision(12);
    std::cout << "k = " << opt.k << '\n';
    std::cout << "seed = " << opt.seed << '\n';
    std::cout << "n_diff_params = " << diff_idx.size() << '\n';
    std::cout << "det_initial_energy = " << det_e0 << '\n';
    std::cout << "pf_initial_energy = " << pf_e0 << '\n';
    std::cout << "det_grad_total_dt = "
              << det_grad_result.total_wall_time_seconds << '\n';
    std::cout << "det_grad_active_dt = "
              << det_grad_result.active_space_gradient_wall_time_seconds << '\n';
    std::cout << "det_grad_adj_dt = "
              << det_grad_result.active_space_adjoint_wall_time_seconds << '\n';
    std::cout << "pf_grad_total_dt = " << pf_grad_result.total_dt << '\n';
    std::cout << "pf_grad_active_dt = "
              << pf_grad_result.active_space_grad_dt << '\n';
    std::cout << "pf_grad_matrix_bp_dt = "
              << pf_grad_result.matrix_backprop_dt << '\n';
    std::cout << "pf_grad_ggo_bp_dt = "
              << pf_grad_result.two_electron_backprop_dt << '\n';
    std::cout << "pf_grad_ao_bp_dt = "
              << pf_grad_result.ao_h1e_backprop_dt << '\n';
    std::cout << "pf_grad_orb_bp_dt = "
              << pf_grad_result.orbital_backprop_dt << '\n';
    std::cout << "det_grad_inf = " << inf_norm(det_grad) << '\n';
    std::cout << "pf_grad_inf = " << inf_norm(pf_grad) << '\n';
    std::cout << "grad_cosine = " << cosine_similarity(det_grad, pf_grad) << '\n';

    std::cout << "det_step_accepted = " << (det_step.accepted ? 1 : 0) << '\n';
    std::cout << "det_step = " << det_step.step << '\n';
    std::cout << "det_energy_after_det = " << det_step.trial_energy << '\n';
    std::cout << "det_delta_det = " << det_step.trial_energy - det_e0 << '\n';
    std::cout << "pf_energy_after_det = " << pf_energy_after_det << '\n';
    std::cout << "pf_delta_det = " << pf_energy_after_det - pf_e0 << '\n';

    std::cout << "pf_step_accepted = " << (pf_step.accepted ? 1 : 0) << '\n';
    std::cout << "pf_step = " << pf_step.step << '\n';
    std::cout << "pf_energy_after_pf = " << pf_step.trial_energy << '\n';
    std::cout << "pf_delta_pf = " << pf_step.trial_energy - pf_e0 << '\n';
    std::cout << "det_energy_after_pf = " << det_energy_after_pf << '\n';
    std::cout << "det_delta_pf = " << det_energy_after_pf - det_e0 << '\n';

    return 0;
  } catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
    return 1;
  }
}
