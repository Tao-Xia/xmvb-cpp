#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "input/loading/loader.hpp"
#include "vbscf/derivatives/gradient/orbital/evaluator.hpp"
#include "vbscf/derivatives/hessian/exact/operator.hpp"
#include "vbscf/optimization/objective/reduced_hvp.hpp"
#include "vbscf/optimization/preconditioners/transported_lbfgs.hpp"
#include "vbscf/optimization/trust_region/truncated_newton.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/layout.hpp"

namespace {

using namespace xmvb::vb;
namespace fs = std::filesystem;

struct Options {
  std::string input_path;
  std::string orbitals_path;
  std::string orbitals_text_path;
  std::string lbfgs_history_steps_path;
  std::string dump_trial_orbitals_path;
  StructureEigensolver eigensolver = StructureEigensolver::Davidson;
  double trust_radius = 0.0;
  double target_kkt_relative_residual =
      std::numeric_limits<double>::quiet_NaN();
  double finite_difference_step = 0.0;
  bool target_kkt_explicit = false;
  bool audit_operator = false;
  bool audit_dense_reference = false;
  bool audit_response_heldout = false;
};

bool parse_bool(const std::string& value) {
  if (value == "true" || value == "1") return true;
  if (value == "false" || value == "0") return false;
  throw std::invalid_argument("invalid boolean value: " + value);
}

Options parse_options(int argc, char** argv) {
  if (argc < 2 || (argc - 2) % 2 != 0) {
    throw std::invalid_argument(
        "usage: audit_newton_step input.xmi "
        "--trust-radius value "
        "[--target-kkt-relative value] [--orbital-value-table-bin path] "
        "[--orbital-value-table-text path] "
        "[--lbfgs-history-steps-dir path] "
        "[--dump-trial-orbitals-bin path] [--finite-difference-step value] "
        "[--eigensolver davidson|dense] [--audit-operator true|false] "
        "[--audit-dense-reference true|false] "
        "[--audit-response-heldout true|false]");
  }
  Options options;
  options.input_path = argv[1];
  for (int index = 2; index < argc; index += 2) {
    const std::string name = argv[index];
    const std::string value = argv[index + 1];
    if (name == "--orbital-value-table-bin") {
      options.orbitals_path = value;
    } else if (name == "--orbital-value-table-text") {
      options.orbitals_text_path = value;
    } else if (name == "--lbfgs-history-steps-dir") {
      options.lbfgs_history_steps_path = value;
    } else if (name == "--dump-trial-orbitals-bin") {
      options.dump_trial_orbitals_path = value;
    } else if (name == "--trust-radius") {
      options.trust_radius = std::stod(value);
    } else if (name == "--target-kkt-relative") {
      options.target_kkt_relative_residual = std::stod(value);
      options.target_kkt_explicit = true;
    } else if (name == "--finite-difference-step") {
      options.finite_difference_step = std::stod(value);
    } else if (name == "--eigensolver") {
      if (value == "davidson") {
        options.eigensolver = StructureEigensolver::Davidson;
      } else if (value == "dense") {
        options.eigensolver = StructureEigensolver::Dense;
      } else {
        throw std::invalid_argument("unsupported structure eigensolver");
      }
    } else if (name == "--audit-operator") {
      options.audit_operator = parse_bool(value);
    } else if (name == "--audit-dense-reference") {
      options.audit_dense_reference = parse_bool(value);
    } else if (name == "--audit-response-heldout") {
      options.audit_response_heldout = parse_bool(value);
    } else {
      throw std::invalid_argument("unknown option: " + name);
    }
  }
  if (!(options.trust_radius > 0.0) ||
      !std::isfinite(options.trust_radius) ||
      (options.target_kkt_explicit &&
       (!std::isfinite(options.target_kkt_relative_residual) ||
        !(options.target_kkt_relative_residual >= 0.0 &&
          options.target_kkt_relative_residual < 1.0))) ||
      (!std::isfinite(options.finite_difference_step) ||
       options.finite_difference_step < 0.0) ||
      (!options.orbitals_path.empty() && !options.orbitals_text_path.empty()) ||
      (options.audit_dense_reference && !options.audit_operator)) {
    throw std::invalid_argument("invalid accepted-point audit options");
  }
  return options;
}

void load_orbitals(
    const std::string& path,
    OrbitalPreparationInput* input);

std::unique_ptr<OrbitalChart> build_chart(
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalGradientResult& gradient);

std::vector<fs::path> sorted_trace_step_directories(
    const std::string& steps_path) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(steps_path)) {
    const std::string name = entry.path().filename().string();
    if (entry.is_directory() &&
        name.compare(0, 5, "step_") == 0) {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    throw std::runtime_error("L-BFGS trace contains no accepted steps");
  }
  return paths;
}

/** @brief Reads a portable accepted-point fixture in Eigen storage order. */
void load_text_orbitals(
    const std::string& path,
    OrbitalPreparationInput* input) {
  if (input == nullptr) throw std::invalid_argument("null orbital input");
  std::ifstream file(path);
  Eigen::Index count = 0;
  if (!(file >> count) || count != input->orbital_value_table.size()) {
    throw std::runtime_error("text orbital table does not match the input chart");
  }
  for (Eigen::Index index = 0; index < count; ++index) {
    double value = 0.0;
    if (!(file >> value) || !std::isfinite(value)) {
      throw std::runtime_error("invalid accepted-point text orbital table");
    }
    input->orbital_value_table.data()[index] = value;
  }
  std::string extra;
  if (file >> extra) {
    throw std::runtime_error("unexpected data after the text orbital table");
  }
}

std::vector<PackedSecantPair> rebuild_lbfgs_history(
    const std::string& steps_path,
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    double nuclear_repulsion_energy,
    StructureEigensolver eigensolver,
    const StructureSolveAccuracy& accuracy,
    int history_size) {
  OrbitalGradientEvaluator evaluator;
  const Eigen::MatrixXd no_initial_eigenvectors;
  std::vector<PackedSecantPair> history;
  history.reserve(history_size);
  Eigen::VectorXd previous_parameters;
  Eigen::VectorXd previous_gradient;
  std::uint64_t previous_rank_signature = 0;
  int previous_reduced_size = -1;
  bool have_previous = false;
  for (const auto& step_path : sorted_trace_step_directories(steps_path)) {
    VbScfInput point = input;
    load_orbitals(
        (step_path / "orbital_value_table_f64.bin").string(),
        &point.orbital_preparation_input);
    const auto evaluated =
        evaluator.evaluate_without_reference_energy_gradient(
            point, {0}, {1.0}, nuclear_repulsion_energy,
            eigensolver, accuracy, no_initial_eigenvectors);
    auto chart = build_chart(point, layout, evaluated);
    const Eigen::VectorXd parameters =
        layout.pack(point.orbital_preparation_input);
    const Eigen::VectorXd gradient = layout.gather_from_full(
        evaluated.sparse_orbital_energy_gradient);
    if (have_previous) {
      const bool rank_changed =
          chart->rank_signature() != previous_rank_signature ||
          chart->reduced_size() != previous_reduced_size;
      if (rank_changed) {
        history.clear();
      } else {
        append_projected_secant_pair(
            *chart,
            parameters - previous_parameters,
            gradient - previous_gradient,
            history_size,
            &history);
      }
    }
    previous_parameters = parameters;
    previous_gradient = gradient;
    previous_rank_signature = chart->rank_signature();
    previous_reduced_size = chart->reduced_size();
    have_previous = true;
  }
  return history;
}

void load_orbitals(
    const std::string& path,
    OrbitalPreparationInput* input) {
  if (input == nullptr) {
    throw std::invalid_argument("null orbital input");
  }
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto bytes = static_cast<std::streamoff>(
      input->orbital_value_table.size() * sizeof(double));
  if (!file || file.tellg() != bytes) {
    throw std::runtime_error("orbital table does not match the input chart");
  }
  file.seekg(0);
  file.read(
      reinterpret_cast<char*>(input->orbital_value_table.data()), bytes);
  if (!file || !Eigen::Map<const Eigen::VectorXd>(
          input->orbital_value_table.data(),
          input->orbital_value_table.size()).allFinite()) {
    throw std::runtime_error("invalid accepted-point orbital table");
  }
}

std::unique_ptr<OrbitalChart> build_chart(
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalGradientResult& gradient) {
  const auto& orbital_input = input.orbital_preparation_input;
  const int n_inactive =
      (orbital_input.n_total_electrons - orbital_input.n_active_electrons) / 2;
  const int n_occupied = n_inactive + orbital_input.n_active_orbitals;
  return std::make_unique<OrbitalChart>(
      orbital_input,
      layout,
      gradient.orbital_preparation_result.auxiliary_orbital_matrix
          .leftCols(n_occupied),
      gradient.orbital_preparation_result.physical_orbital_frame
          .normalized_orbital_matrix,
      &gradient.ao_effective_one_electron_result.ao_effective_h1e);
}

class AcceptedPointHvp final : public ReducedHvp {
public:
  struct RecordedBlock {
    Eigen::MatrixXd directions;
    Eigen::MatrixXd images;
    std::uint64_t model_revision = 0;
  };

  AcceptedPointHvp(
      const OrbitalGradientResult& gradient,
      const VbScfInput& input,
      const SparseParameterLayout& layout,
      const OrbitalChart& chart,
      bool record_actions = false)
      : operator_(gradient.second_order_context, &input, layout, &chart),
        record_actions_(record_actions) {
    if (gradient.second_order_context == nullptr) {
      throw std::runtime_error("accepted point lacks a second-order context");
    }
  }

  Eigen::VectorXd apply(const Eigen::VectorXd& direction) override {
    Eigen::VectorXd image = operator_.apply_reduced(direction);
    if (record_actions_) {
      RecordedBlock block;
      block.directions = direction;
      block.images = image;
      block.model_revision = model_revision();
      recorded_blocks_.push_back(std::move(block));
    }
    return image;
  }

  Eigen::MatrixXd apply_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
    Eigen::MatrixXd images = operator_.apply_reduced_batch(directions);
    if (record_actions_) {
      recorded_blocks_.push_back(
          RecordedBlock{directions, images, model_revision()});
    }
    return images;
  }

  std::uint64_t model_revision() const noexcept override {
    return operator_.response_model_revision();
  }

  Eigen::MatrixXd apply_frozen_batch(
      const Eigen::Ref<const Eigen::MatrixXd>& directions) override {
    HvpComponents components;
    components.freeze_structure_response = true;
    return operator_.apply_reduced_batch(directions, components);
  }

  ExactHvpOperator::Diagnostics diagnostics() const {
    return operator_.diagnostics();
  }

  ResponseLowRankModel response_model() const {
    return operator_.response_low_rank_model();
  }

  void clear_recorded_blocks() { recorded_blocks_.clear(); }

  const std::vector<RecordedBlock>& recorded_blocks() const noexcept {
    return recorded_blocks_;
  }

private:
  ExactHvpOperator operator_;
  bool record_actions_ = false;
  std::vector<RecordedBlock> recorded_blocks_;
};

struct HvpAuditApplication {
  Eigen::MatrixXd images;
  ExactHvpOperator::Diagnostics diagnostics;
};

double relative_matrix_error(
    const Eigen::Ref<const Eigen::MatrixXd>& difference,
    const Eigen::Ref<const Eigen::MatrixXd>& reference) {
  return difference.stableNorm() /
      std::max(reference.stableNorm(), std::numeric_limits<double>::min());
}

void print_projected_skew(
    const std::string& label,
    const Eigen::Ref<const Eigen::MatrixXd>& directions,
    const Eigen::Ref<const Eigen::MatrixXd>& images) {
  if (directions.rows() != images.rows() ||
      directions.cols() != images.cols() || directions.cols() == 0) {
    throw std::invalid_argument("invalid HVP block in operator audit");
  }
  const Eigen::MatrixXd curvature = directions.transpose() * images;
  const Eigen::MatrixXd skew = curvature - curvature.transpose();
  Eigen::Index max_row = 0;
  Eigen::Index max_column = 0;
  const double max_abs_skew = skew.cwiseAbs().maxCoeff(
      &max_row, &max_column);
  std::cout << label << "_rank = " << directions.cols() << '\n'
            << label << "_curvature_frobenius = "
            << curvature.stableNorm() << '\n'
            << label << "_skew_frobenius = " << skew.stableNorm() << '\n'
            << label << "_relative_skew = "
            << relative_matrix_error(skew, curvature) << '\n'
            << label << "_max_abs_skew = " << max_abs_skew << '\n'
            << label << "_max_skew_row = " << max_row << '\n'
            << label << "_max_skew_column = " << max_column << '\n';
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> concatenate_recorded_blocks(
    const std::vector<AcceptedPointHvp::RecordedBlock>& blocks) {
  if (blocks.empty()) return {};
  const Eigen::Index dimension = blocks.front().directions.rows();
  Eigen::Index width = 0;
  for (const auto& block : blocks) {
    if (block.directions.rows() != dimension ||
        block.images.rows() != dimension ||
        block.directions.cols() != block.images.cols() ||
        !block.directions.allFinite() || !block.images.allFinite()) {
      throw std::runtime_error("recorded HVP block is inconsistent");
    }
    width += block.directions.cols();
  }
  Eigen::MatrixXd directions(dimension, width);
  Eigen::MatrixXd images(dimension, width);
  Eigen::Index first = 0;
  for (const auto& block : blocks) {
    directions.middleCols(first, block.directions.cols()) = block.directions;
    images.middleCols(first, block.images.cols()) = block.images;
    first += block.directions.cols();
  }
  return {std::move(directions), std::move(images)};
}

HvpAuditApplication apply_fresh_block(
    const OrbitalGradientResult& accepted,
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalChart& chart,
    const Eigen::Ref<const Eigen::MatrixXd>& directions,
    HvpComponents components = {}) {
  ExactHvpOperator operation(
      accepted.second_order_context, &input, layout, &chart);
  HvpAuditApplication result;
  result.images = operation.apply_reduced_batch(directions, components);
  result.diagnostics = operation.diagnostics();
  return result;
}

void print_response_diagnostics(
    const std::string& label,
    const ExactHvpOperator::Diagnostics& diagnostics) {
  std::cout << label << "_response_block_actions = "
            << diagnostics.structure_response_block_actions << '\n'
            << label << "_response_max_iterations = "
            << diagnostics.max_structure_response_iterations << '\n'
            << label << "_response_max_relative_residual = "
            << diagnostics.max_structure_response_relative_residual << '\n';
}

/** @brief Rejects a same-model identity defect above scaled roundoff. */
void require_model_identity(
    const std::string& label,
    const Eigen::MatrixXd& actual,
    const Eigen::MatrixXd& expected) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols() ||
      !actual.allFinite() || !expected.allFinite()) {
    throw std::runtime_error("invalid frozen response identity: " + label);
  }
  const double scale = std::max(
      {1.0, actual.stableNorm(), expected.stableNorm()});
  const double tolerance = 1024.0 * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max<Eigen::Index>(
          1, std::max(actual.rows(), actual.cols()))) * scale;
  const double error = (actual - expected).stableNorm();
  std::cout << label << "_error = " << error << '\n'
            << label << "_limit = " << tolerance << '\n';
  if (!(error <= tolerance)) {
    throw std::runtime_error("frozen response identity failed: " + label);
  }
}

/** @brief Builds three independent probes even when Newton stops at rank one. */
Eigen::MatrixXd response_audit_probes(Eigen::Index dimension) {
  if (dimension < 3) {
    throw std::runtime_error("response audit needs three reduced directions");
  }
  Eigen::MatrixXd probes(dimension, 3);
  for (Eigen::Index column = 0; column < probes.cols(); ++column) {
    for (Eigen::Index row = 0; row < dimension; ++row) {
      const double x = static_cast<double>(row + 1);
      const double k = static_cast<double>(column + 1);
      probes(row, column) = std::sin(std::sqrt(2.0) * x * k) +
          std::cos(std::sqrt(3.0) * x * (k + 1.0));
    }
    for (int pass = 0; pass < 2; ++pass) {
      for (Eigen::Index previous = 0; previous < column; ++previous) {
        probes.col(column) -= probes.col(previous).dot(probes.col(column)) *
            probes.col(previous);
      }
    }
    const double norm = probes.col(column).stableNorm();
    if (!(norm > std::sqrt(std::numeric_limits<double>::epsilon()))) {
      throw std::runtime_error("dependent frozen-response audit directions");
    }
    probes.col(column) /= norm;
  }
  return probes;
}

/** @brief Builds deterministic probes outside every sampled orbital direction. */
Eigen::MatrixXd heldout_response_probes(
    Eigen::Index dimension,
    const Eigen::Ref<const Eigen::MatrixXd>& sampled_directions,
    Eigen::Index count = 3) {
  if (dimension <= count || sampled_directions.rows() != dimension) {
    throw std::invalid_argument("invalid held-out response probe dimensions");
  }
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> sampled_qr(sampled_directions);
  sampled_qr.setThreshold(
      std::sqrt(std::numeric_limits<double>::epsilon()));
  const Eigen::Index sampled_rank = sampled_qr.rank();
  if (sampled_rank + count > dimension) {
    throw std::runtime_error("sampled HVP directions span the orbital space");
  }
  const Eigen::MatrixXd sampled_basis = sampled_qr.householderQ() *
      Eigen::MatrixXd::Identity(dimension, sampled_rank);
  Eigen::MatrixXd probes = response_audit_probes(dimension).leftCols(count);
  for (Eigen::Index column = 0; column < count; ++column) {
    for (int pass = 0; pass < 2; ++pass) {
      probes.col(column).noalias() -= sampled_basis *
          (sampled_basis.transpose() * probes.col(column));
      for (Eigen::Index previous = 0; previous < column; ++previous) {
        probes.col(column) -= probes.col(previous).dot(probes.col(column)) *
            probes.col(previous);
      }
    }
    const double norm = probes.col(column).stableNorm();
    if (!(norm > std::sqrt(std::numeric_limits<double>::epsilon()))) {
      throw std::runtime_error("dependent held-out response probe");
    }
    probes.col(column) /= norm;
  }
  std::cout << "response_heldout_sampled_rank = " << sampled_rank << '\n'
            << "response_heldout_probe_rank = " << count << '\n'
            << "response_heldout_sample_overlap = "
            << (sampled_basis.transpose() * probes).stableNorm() << '\n';
  return probes;
}

/** @brief Compares learned Schur modes with fresh response-only HVPs. */
void run_heldout_response_audit(
    const OrbitalGradientResult& accepted,
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalChart& chart,
    const AcceptedPointHvp& learned_hvp,
    const Eigen::VectorXd& baseline_direction) {
  const auto [recorded_directions, recorded_images] =
      concatenate_recorded_blocks(learned_hvp.recorded_blocks());
  (void)recorded_images;
  const Eigen::Index baseline_width =
      baseline_direction.size() == chart.reduced_size() ? 1 : 0;
  Eigen::MatrixXd sampled(
      chart.reduced_size(), recorded_directions.cols() + baseline_width);
  sampled.leftCols(recorded_directions.cols()) = recorded_directions;
  if (baseline_width != 0) sampled.rightCols(1) = baseline_direction;
  const Eigen::MatrixXd probes = heldout_response_probes(
      chart.reduced_size(), sampled);

  HvpComponents no_structure;
  no_structure.structure_response = false;
  const Eigen::MatrixXd full = apply_fresh_block(
      accepted, input, layout, chart, probes).images;
  const Eigen::MatrixXd fixed = apply_fresh_block(
      accepted, input, layout, chart, probes, no_structure).images;
  const Eigen::MatrixXd reference = full - fixed;
  const ResponseLowRankModel model = learned_hvp.response_model();
  const Eigen::MatrixXd learned = model.apply(probes);
  std::cout << "response_heldout_model_rank = "
            << model.orbital_couplings.cols() << '\n'
            << "response_heldout_reference_norm = "
            << reference.stableNorm() << '\n'
            << "response_heldout_full_model_relative_error = "
            << relative_matrix_error(learned - reference, reference) << '\n';

  const Eigen::MatrixXd schur = model.apply(
      Eigen::MatrixXd::Identity(chart.reduced_size(), chart.reduced_size()));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      0.5 * (schur + schur.transpose()));
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error("held-out Schur eigensolve failed");
  }
  std::vector<Eigen::Index> order(
      static_cast<std::size_t>(solver.eigenvalues().size()));
  for (Eigen::Index index = 0; index < solver.eigenvalues().size(); ++index) {
    order[static_cast<std::size_t>(index)] = index;
  }
  std::sort(order.begin(), order.end(), [&](Eigen::Index left, Eigen::Index right) {
    return std::abs(solver.eigenvalues()[left]) >
        std::abs(solver.eigenvalues()[right]);
  });
  const std::vector<int> requested_ranks = {1, 2, 5, 10, 20};
  Eigen::MatrixXd truncated = Eigen::MatrixXd::Zero(
      chart.reduced_size(), probes.cols());
  int next = 0;
  for (int mode = 0;
       mode < static_cast<int>(order.size()) &&
       next < static_cast<int>(requested_ranks.size());
       ++mode) {
    const Eigen::Index index = order[static_cast<std::size_t>(mode)];
    const Eigen::VectorXd vector = solver.eigenvectors().col(index);
    truncated.noalias() += solver.eigenvalues()[index] * vector *
        (vector.transpose() * probes);
    if (mode + 1 != requested_ranks[static_cast<std::size_t>(next)]) continue;
    std::cout << "response_heldout_rank_" << mode + 1
              << "_relative_error = "
              << relative_matrix_error(truncated - reference, reference)
              << '\n';
    ++next;
  }
  std::cout.flush();
}

/** @brief Certifies action identities within one frozen response model version. */
void run_frozen_response_audit(
    const OrbitalGradientResult& accepted,
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalChart& chart) {
  ExactHvpOperator operation(
      accepted.second_order_context, &input, layout, &chart);
  const Eigen::MatrixXd probes = response_audit_probes(chart.reduced_size());
  const Eigen::MatrixXd seed_images = operation.apply_reduced_batch(
      probes.leftCols(2));
  const std::uint64_t seed_revision = operation.response_model_revision();
  const ResponseLowRankModel seed_model =
      operation.response_low_rank_model();
  HvpComponents frozen;
  frozen.freeze_structure_response = true;
  require_model_identity("response_model_seed_replay", seed_images,
      operation.apply_reduced_batch(probes.leftCols(2), frozen));
  if (operation.response_model_revision() != seed_revision) {
    throw std::runtime_error("frozen seed action changed the response model");
  }

  // Enrichment changes the model legitimately. Refresh every old column before
  // assembling secants; never demand equality with the smaller seed model.
  const Eigen::MatrixXd added_image = operation.apply_reduced_batch(
      probes.rightCols(1));
  const std::uint64_t revision = operation.response_model_revision();
  const ResponseLowRankModel enriched_model =
      operation.response_low_rank_model();
  const Eigen::MatrixXd images = operation.apply_reduced_batch(probes, frozen);
  const Eigen::MatrixXd low_rank_refresh = seed_images +
      enriched_model.apply(probes.leftCols(2)) -
      seed_model.apply(probes.leftCols(2));
  std::cout << "response_model_seed_revision = " << seed_revision << '\n'
            << "response_model_enriched_revision = " << revision << '\n'
            << "response_model_probe_count = " << probes.cols() << '\n'
            << "response_model_enrichment_old_images_change = "
            << (images.leftCols(2) - seed_images).stableNorm() << '\n'
            << "response_model_low_rank_refresh_relative = "
            << relative_matrix_error(
                   low_rank_refresh - images.leftCols(2),
                   images.leftCols(2)) << '\n';
  std::cout << "response_model_seed_low_rank = "
            << seed_model.orbital_couplings.cols() << '\n'
            << "response_model_enriched_low_rank = "
            << enriched_model.orbital_couplings.cols() << '\n';
  require_model_identity(
      "response_model_low_rank_refresh",
      low_rank_refresh,
      images.leftCols(2));
  require_model_identity("response_model_new_column", images.rightCols(1),
      added_image);
  const Eigen::MatrixXd curvature = probes.transpose() * images;
  require_model_identity("response_model_symmetry", curvature,
      curvature.transpose());

  Eigen::MatrixXd scalar_images(images.rows(), images.cols());
  for (Eigen::Index column = 0; column < probes.cols(); ++column) {
    scalar_images.col(column) = operation.apply_reduced(probes.col(column), frozen);
  }
  require_model_identity("response_model_scalar_block", scalar_images, images);
  Eigen::MatrixXd segmented(images.rows(), images.cols());
  segmented.leftCols(1) = operation.apply_reduced_batch(probes.leftCols(1), frozen);
  segmented.rightCols(2) = operation.apply_reduced_batch(probes.rightCols(2), frozen);
  require_model_identity("response_model_segmentation", segmented, images);
  const Eigen::MatrixXd reversed_probes = probes.rowwise().reverse();
  const Eigen::MatrixXd reversed_images = operation.apply_reduced_batch(
      reversed_probes, frozen);
  require_model_identity("response_model_permutation",
      reversed_images.rowwise().reverse(), images);

  const Eigen::Vector3d coefficients(0.7, -1.3, 0.2);
  const Eigen::MatrixXd combination = probes * coefficients;
  require_model_identity("response_model_linearity",
      operation.apply_reduced_batch(combination, frozen), images * coefficients);
  require_model_identity("response_model_repeat", images,
      operation.apply_reduced_batch(probes, frozen));
  require_model_identity("response_model_zero",
      operation.apply_reduced_batch(
          Eigen::MatrixXd::Zero(probes.rows(), 1), frozen),
      Eigen::MatrixXd::Zero(probes.rows(), 1));
  if (operation.response_model_revision() != revision) {
    throw std::runtime_error("frozen response action changed the model revision");
  }
  std::cout << "response_model_invariants = PASS\n";
}

void run_operator_audit(
    const Options& options,
    const AcceptedPointHvp& recorded_hvp,
    const OrbitalGradientResult& accepted,
    const VbScfInput& input,
    const SparseParameterLayout& layout,
    const OrbitalChart& chart,
    double nuclear_repulsion_energy,
    const StructureSolveAccuracy& accuracy) {
  std::cout << std::setprecision(17);
  run_frozen_response_audit(accepted, input, layout, chart);
  const auto& blocks = recorded_hvp.recorded_blocks();
  std::cout << std::setprecision(17)
            << "operator_audit_history_comparison = adaptive_models_not_identity_checks\n"
            << "operator_audit_recorded_block_count = "
            << blocks.size() << '\n';
  if (blocks.empty()) {
    std::cout << "operator_audit_recorded_direction_count = 0\n";
    return;
  }
  const auto [directions, recorded_images] =
      concatenate_recorded_blocks(blocks);
  std::cout << "operator_audit_recorded_direction_count = "
            << directions.cols() << '\n';
  print_projected_skew(
      "operator_audit_recorded", directions, recorded_images);
  // These are historical samples, possibly from different response models.
  // Their differences are diagnostic, not frozen-model regression failures.
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    std::cout << "operator_audit_recorded_block_" << index
              << "_model_revision = " << blocks[index].model_revision << '\n';
    print_projected_skew(
        "operator_audit_recorded_block_" + std::to_string(index),
        blocks[index].directions,
        blocks[index].images);
  }

  ExactHvpOperator replay_operator(
      accepted.second_order_context, &input, layout, &chart);
  Eigen::MatrixXd replay_images(
      directions.rows(), directions.cols());
  Eigen::Index first = 0;
  for (const auto& block : blocks) {
    replay_images.middleCols(first, block.directions.cols()) =
        replay_operator.apply_reduced_batch(block.directions);
    first += block.directions.cols();
  }
  const auto replay_first_diagnostics = replay_operator.diagnostics();
  Eigen::MatrixXd warm_replay_images(
      directions.rows(), directions.cols());
  first = 0;
  for (const auto& block : blocks) {
    warm_replay_images.middleCols(first, block.directions.cols()) =
        replay_operator.apply_reduced_batch(block.directions);
    first += block.directions.cols();
  }
  std::cout << "operator_audit_recorded_vs_cold_replay_relative = "
            << relative_matrix_error(
                   recorded_images - replay_images, replay_images) << '\n'
            << "operator_audit_cold_vs_warm_replay_relative = "
            << relative_matrix_error(
                   warm_replay_images - replay_images, replay_images) << '\n';
  print_projected_skew(
      "operator_audit_cold_replay", directions, replay_images);
  print_projected_skew(
      "operator_audit_warm_replay", directions, warm_replay_images);
  print_response_diagnostics(
      "operator_audit_cold_replay", replay_first_diagnostics);
  print_response_diagnostics(
      "operator_audit_warm_replay", replay_operator.diagnostics());

  const HvpAuditApplication forward = apply_fresh_block(
      accepted, input, layout, chart, directions);
  Eigen::MatrixXd reversed_directions = directions.rowwise().reverse();
  HvpAuditApplication reversed = apply_fresh_block(
      accepted, input, layout, chart, reversed_directions);
  Eigen::MatrixXd reversed_images = reversed.images.rowwise().reverse();
  std::cout << "operator_audit_replay_vs_single_block_relative = "
            << relative_matrix_error(
                   forward.images - replay_images, replay_images) << '\n';
  std::cout << "operator_audit_forward_vs_reversed_relative = "
            << relative_matrix_error(
                   reversed_images - forward.images, forward.images) << '\n';
  print_projected_skew(
      "operator_audit_forward", directions, forward.images);
  print_response_diagnostics(
      "operator_audit_forward", forward.diagnostics);
  print_projected_skew(
      "operator_audit_reversed", directions, reversed_images);
  print_response_diagnostics(
      "operator_audit_reversed", reversed.diagnostics);

  HvpComponents no_structure_components;
  no_structure_components.structure_response = false;
  const HvpAuditApplication no_structure = apply_fresh_block(
      accepted,
      input,
      layout,
      chart,
      directions,
      no_structure_components);
  const Eigen::MatrixXd structure_images =
      forward.images - no_structure.images;
  print_projected_skew(
      "operator_audit_no_structure", directions, no_structure.images);
  print_projected_skew(
      "operator_audit_structure_difference", directions, structure_images);

  if (options.audit_dense_reference) {
    OrbitalGradientEvaluator evaluator;
    const Eigen::MatrixXd no_initial_eigenvectors;
    const OrbitalGradientResult dense_accepted =
        evaluator.evaluate_without_reference_energy_gradient(
            input,
            {0},
            {1.0},
            nuclear_repulsion_energy,
            StructureEigensolver::Dense,
            accuracy,
            no_initial_eigenvectors);
    const HvpAuditApplication dense_full = apply_fresh_block(
        dense_accepted, input, layout, chart, directions);
    const HvpAuditApplication dense_no_structure = apply_fresh_block(
        dense_accepted,
        input,
        layout,
        chart,
        directions,
        no_structure_components);
    const Eigen::MatrixXd dense_structure =
        dense_full.images - dense_no_structure.images;
    std::cout << "operator_audit_primary_vs_dense_full_relative = "
              << relative_matrix_error(
                     forward.images - dense_full.images,
                     dense_full.images)
              << '\n'
              << "operator_audit_primary_vs_dense_no_structure_relative = "
              << relative_matrix_error(
                     no_structure.images - dense_no_structure.images,
                     dense_no_structure.images)
              << '\n'
              << "operator_audit_primary_vs_dense_structure_relative = "
              << relative_matrix_error(
                     structure_images - dense_structure,
                     dense_structure)
              << '\n';
    print_projected_skew(
        "operator_audit_dense_full", directions, dense_full.images);
    print_projected_skew(
        "operator_audit_dense_no_structure",
        directions,
        dense_no_structure.images);
    print_projected_skew(
        "operator_audit_dense_structure_difference",
        directions,
        dense_structure);
    print_response_diagnostics(
        "operator_audit_dense_full", dense_full.diagnostics);
  }

  Eigen::VectorXd x = directions.col(0);
  Eigen::VectorXd y;
  if (directions.cols() >= 2) {
    y = directions.col(1);
  } else {
    y.resize(directions.rows());
    for (Eigen::Index row = 0; row < y.size(); ++row) {
      const double index = static_cast<double>(row + 1);
      y[row] = std::sin(std::sqrt(2.0) * index) +
          std::cos(std::sqrt(3.0) * index);
    }
    y.noalias() -= x.dot(y) / x.squaredNorm() * x;
    if (!(y.norm() > std::sqrt(std::numeric_limits<double>::epsilon()))) {
      throw std::runtime_error(
          "operator audit could not construct an additive probe");
    }
    y.normalize();
  }
  Eigen::MatrixXd x_block(x.size(), 1);
  Eigen::MatrixXd y_block(y.size(), 1);
  Eigen::MatrixXd sum_block(x.size(), 1);
  x_block.col(0) = x;
  y_block.col(0) = y;
  sum_block.col(0) = x + y;
  const Eigen::VectorXd hx = apply_fresh_block(
      accepted, input, layout, chart, x_block).images.col(0);
  const Eigen::VectorXd hy = apply_fresh_block(
      accepted, input, layout, chart, y_block).images.col(0);
  const Eigen::VectorXd hsum = apply_fresh_block(
      accepted, input, layout, chart, sum_block).images.col(0);
  std::cout << "operator_audit_cold_additivity_relative = "
            << (hsum - hx - hy).stableNorm() /
                   std::max(
                       (hx + hy).stableNorm(),
                       std::numeric_limits<double>::min())
            << '\n';

  if (options.finite_difference_step > 0.0) {
    Eigen::VectorXd probe = x + std::sqrt(2.0) * y;
    probe.normalize();
    Eigen::MatrixXd probe_block(probe.size(), 1);
    probe_block.col(0) = probe;
    const Eigen::VectorXd analytic = apply_fresh_block(
        accepted, input, layout, chart, probe_block).images.col(0);
    const auto evaluate_displaced_gradient = [&](double displacement) {
      VbScfInput point = input;
      point.orbital_preparation_input = chart.retract_step(
          input.orbital_preparation_input, probe, displacement);
      OrbitalGradientEvaluator evaluator;
      const Eigen::MatrixXd no_initial_eigenvectors;
      const auto evaluated =
          evaluator.evaluate_without_reference_energy_gradient(
              point,
              {0},
              {1.0},
              nuclear_repulsion_energy,
              options.eigensolver,
              accuracy,
              no_initial_eigenvectors);
      return chart.project_reduced_gradient(
          layout.gather_from_full(
              evaluated.sparse_orbital_energy_gradient));
    };
    const double h = options.finite_difference_step;
    const Eigen::VectorXd finite_difference =
        (evaluate_displaced_gradient(h) -
         evaluate_displaced_gradient(-h)) /
        (2.0 * h);
    std::cout << "operator_audit_probe_fd_relative = "
              << (analytic - finite_difference).stableNorm() /
                     std::max(
                         finite_difference.stableNorm(),
                         std::numeric_limits<double>::min())
              << '\n';
  }
  std::cout.flush();
}

const char* stop_reason_name(TruncatedNewtonStopReason reason) {
  switch (reason) {
    case TruncatedNewtonStopReason::None: return "none";
    case TruncatedNewtonStopReason::ModelKktConverged: return "model_kkt";
    case TruncatedNewtonStopReason::BelowOuterAccuracy: return "below_outer_accuracy";
    case TruncatedNewtonStopReason::CompleteSpace: return "complete_space";
    case TruncatedNewtonStopReason::DependentDirections: return "dependent_directions";
    case TruncatedNewtonStopReason::InvalidProjectedStep: return "invalid_projected_step";
    case TruncatedNewtonStopReason::RadiusAdjusted: return "radius_adjusted";
    case TruncatedNewtonStopReason::PreconditionedGradient: return "preconditioned_gradient";
  }
  return "unknown";
}

double relative_norm(
    const Eigen::VectorXd& numerator,
    const Eigen::VectorXd& denominator) {
  return numerator.stableNorm() /
      std::max(denominator.stableNorm(),
               std::numeric_limits<double>::min());
}

/** @brief Audits one fixed accepted-point orbital trust-region subproblem. */
void run_audit(const Options& options) {
  VbScfInputLoadOptions load_options;
  load_options.standard_two_electron_mode = StandardTwoElectronMode::Exact;
  auto loaded = load_vbscf_input_with_timings(
      options.input_path, load_options);
  VbScfInput input = std::move(loaded.input);
  if (!options.orbitals_path.empty()) {
    load_orbitals(options.orbitals_path, &input.orbital_preparation_input);
  } else if (!options.orbitals_text_path.empty()) {
    load_text_orbitals(
        options.orbitals_text_path, &input.orbital_preparation_input);
  }

  OrbitalGradientEvaluator evaluator;
  const Eigen::MatrixXd no_initial_eigenvectors;
  const StructureSolveAccuracy accuracy;
  const auto accepted = evaluator.evaluate_without_reference_energy_gradient(
      input, {0}, {1.0}, loaded.nuclear_repulsion_energy,
      options.eigensolver, accuracy, no_initial_eigenvectors);
  const SparseParameterLayout layout(input.orbital_preparation_input);
  auto chart = build_chart(input, layout, accepted);
  const Eigen::VectorXd packed_gradient = layout.gather_from_full(
      accepted.sparse_orbital_energy_gradient);
  const auto projected = chart->project_gradient(packed_gradient);
  const NonredundantRetractionMetric metric(
      *chart, layout, input.orbital_preparation_input);
  AcceptedPointHvp hvp(
      accepted,
      input,
      layout,
      *chart,
      options.audit_operator || options.audit_response_heldout);

  std::vector<PackedSecantPair> packed_secant_history;
  std::unique_ptr<TransportedReducedLbfgsPreconditioner>
      lbfgs_preconditioner;
  Eigen::VectorXd lbfgs_baseline_step;
  Eigen::VectorXd lbfgs_baseline_hessian_step;
  double lbfgs_baseline_kkt_relative =
      std::numeric_limits<double>::quiet_NaN();
  double correction_step_norm =
      std::numeric_limits<double>::quiet_NaN();
  std::size_t correction_model_restarts = 0;
  if (!options.lbfgs_history_steps_path.empty()) {
    constexpr int kHistorySize = 100;
    packed_secant_history = rebuild_lbfgs_history(
        options.lbfgs_history_steps_path,
        input,
        layout,
        loaded.nuclear_repulsion_energy,
        options.eigensolver,
        accuracy,
        kHistorySize);
    lbfgs_preconditioner =
        std::make_unique<TransportedReducedLbfgsPreconditioner>(
            build_transported_reduced_lbfgs_preconditioner(
                *chart,
                packed_secant_history,
                kHistorySize,
                LbfgsInitialInverse::OrbitalBlock));
  }

  const auto solve_start = std::chrono::steady_clock::now();
  if (lbfgs_preconditioner != nullptr) {
    lbfgs_baseline_step = -lbfgs_preconditioner->apply(
        projected.reduced_gradient);
    if (!lbfgs_baseline_step.allFinite() ||
        projected.reduced_gradient.dot(lbfgs_baseline_step) >= 0.0) {
      throw std::runtime_error(
          "reconstructed L-BFGS history produced no descent direction");
    }
    lbfgs_baseline_hessian_step = hvp.apply(lbfgs_baseline_step);
    lbfgs_baseline_kkt_relative = relative_norm(
        projected.reduced_gradient + lbfgs_baseline_hessian_step,
        projected.reduced_gradient);
  }

  const double target_kkt_relative_residual =
      options.target_kkt_explicit
      ? options.target_kkt_relative_residual
      : inexact_newton_forcing_term(
            projected.reduced_gradient.stableNorm(),
            projected.reduced_gradient.lpNorm<Eigen::Infinity>(),
            projected.reduced_gradient.stableNorm(),
            accuracy.gradient_tolerance);
  // An explicitly requested KKT target defines an algebraic oracle audit.
  // Do not let the production outer energy/gradient tolerances terminate the
  // subproblem before that requested residual has been reached.
  const double audit_gradient_tolerance =
      options.target_kkt_explicit ? 0.0 : accuracy.gradient_tolerance;
  TruncatedNewtonStepResult step;
  if (options.audit_operator || options.audit_response_heldout) {
    hvp.clear_recorded_blocks();
  }
  std::exception_ptr solve_error;
  try {
    if (lbfgs_preconditioner == nullptr) {
      step = solve_nonredundant_truncated_newton_step(
          metric, *chart, projected, options.trust_radius,
          accuracy.energy_tolerance, audit_gradient_tolerance,
          target_kkt_relative_residual, &hvp, nullptr);
      clamp_nonredundant_step_result_to_retract_tangent_radius(
          projected, options.trust_radius, metric, &step);
    } else {
      // Apply exact curvature as a residual correction to the complete L-BFGS
      // proposal.  Solving H delta = -(g + H p_B) makes p_B + delta an exact
      // Newton step without blending two independently scaled directions.
      OrbitalChart::ProjectionResult correction_projection;
      TruncatedNewtonStepResult correction;
      for (;;) {
        const std::uint64_t rhs_revision = hvp.model_revision();
        lbfgs_baseline_hessian_step =
            hvp.apply_frozen_batch(lbfgs_baseline_step).col(0);
        if (hvp.model_revision() != rhs_revision) {
          throw std::runtime_error(
              "baseline replay changed the response model");
        }
        correction_projection.reduced_gradient =
            projected.reduced_gradient + lbfgs_baseline_hessian_step;
        lbfgs_baseline_kkt_relative = relative_norm(
            correction_projection.reduced_gradient,
            projected.reduced_gradient);
        correction = solve_nonredundant_truncated_newton_step(
            metric, *chart, correction_projection, options.trust_radius,
            accuracy.energy_tolerance, audit_gradient_tolerance,
            target_kkt_relative_residual, &hvp, lbfgs_preconditioner.get());
        const std::uint64_t final_revision = hvp.model_revision();
        if (final_revision == rhs_revision) break;
        if (final_revision < rhs_revision) {
          throw std::runtime_error(
              "response model revision regressed during residual correction");
        }
        // A richer H changes both the correction matrix and g + H p_B.
        // Restart with the refreshed RHS rather than accepting a solve of the
        // old affine model. Revisions increase only on finite-space enrichment;
        // no empirical restart cap or frozen-old-model downgrade is needed.
        ++correction_model_restarts;
      }
      if (!truncated_newton_step_is_usable(
              correction, correction_projection.reduced_gradient) ||
          correction.trust_region_shift != 0.0 ||
          correction.reached_boundary) {
        throw std::runtime_error(
            "L-BFGS residual correction did not produce an interior Newton step");
      }
      correction_step_norm = metric.norm(correction.reduced_step);
      step = std::move(correction);
      step.reduced_step += lbfgs_baseline_step;
      step.reduced_hessian_times_step += lbfgs_baseline_hessian_step;
      const std::uint64_t combined_revision = hvp.model_revision();
      const Eigen::VectorXd combined_image =
          hvp.apply_frozen_batch(step.reduced_step).col(0);
      if (hvp.model_revision() != combined_revision) {
        throw std::runtime_error(
            "combined residual correction changed the response model");
      }
      require_model_identity("response_model_combined_correction",
          step.reduced_hessian_times_step, combined_image);
      step.reduced_hessian_times_step = combined_image;
      step.reduced_metric_times_step = metric.apply(step.reduced_step);
      step.retract_tangent_norm = metric.norm(step.reduced_step);
      step.predicted_decrease =
          -projected.reduced_gradient.dot(step.reduced_step) -
          0.5 * step.reduced_step.dot(step.reduced_hessian_times_step);
      refresh_truncated_newton_step_certificate(
          projected.reduced_gradient, &step);
      if (step.retract_tangent_norm > options.trust_radius) {
        throw std::runtime_error(
            "combined L-BFGS/Newton step lies outside the audit trust radius");
      }
    }
  } catch (...) {
    solve_error = std::current_exception();
  }
  const double solve_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - solve_start).count();
  if (options.audit_operator) {
    if (solve_error != nullptr) {
      try {
        std::rethrow_exception(solve_error);
      } catch (const std::exception& error) {
        std::cout << "operator_audit_subproblem_error = "
                  << error.what() << '\n';
      }
    }
    run_operator_audit(
        options,
        hvp,
        accepted,
        input,
        layout,
        *chart,
        loaded.nuclear_repulsion_energy,
        accuracy);
  }
  if (options.audit_response_heldout) {
    run_heldout_response_audit(
        accepted,
        input,
        layout,
        *chart,
        hvp,
        lbfgs_baseline_step);
  }
  if (solve_error != nullptr) std::rethrow_exception(solve_error);
  if (!truncated_newton_step_is_usable(
          step, projected.reduced_gradient)) {
    throw std::runtime_error("subproblem produced no usable descent step");
  }

  const Eigen::VectorXd cached_hs = step.reduced_hessian_times_step;
  const std::uint64_t step_revision = hvp.model_revision();
  const Eigen::VectorXd fresh_hs =
      hvp.apply_frozen_batch(step.reduced_step).col(0);
  if (options.audit_operator) {
    require_model_identity("response_model_cached_step", cached_hs, fresh_hs);
    if (hvp.model_revision() != step_revision) {
      throw std::runtime_error("step certificate changed the response model");
    }
  }
  const Eigen::VectorXd fresh_ms = metric.apply(step.reduced_step);
  const Eigen::VectorXd fresh_residual = projected.reduced_gradient +
      fresh_hs + step.trust_region_shift * fresh_ms;
  Eigen::VectorXd cached_residual;
  if (cached_hs.size() == fresh_hs.size()) {
    cached_residual = projected.reduced_gradient + cached_hs +
        step.trust_region_shift * fresh_ms;
  }
  const Eigen::MatrixXd& q = step.subspace.orthonormal_basis;
  double projected_residual_relative =
      std::numeric_limits<double>::quiet_NaN();
  if (q.rows() == fresh_residual.size() && q.cols() > 0) {
    projected_residual_relative =
        (q.transpose() * fresh_residual).stableNorm() /
        projected.reduced_gradient.stableNorm();
  }

  VbScfInput trial = input;
  trial.orbital_preparation_input = chart->retract_step(
      input.orbital_preparation_input, step.reduced_step);
  if (!options.dump_trial_orbitals_path.empty()) {
    const auto& values = trial.orbital_preparation_input.orbital_value_table;
    std::ofstream file(options.dump_trial_orbitals_path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("cannot open trial orbital table output");
    }
    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!file) {
      throw std::runtime_error("cannot write trial orbital table output");
    }
  }
  const auto trial_gradient = evaluator.evaluate_without_reference_energy_gradient(
      trial, {0}, {1.0}, loaded.nuclear_repulsion_energy,
      options.eigensolver, accuracy, no_initial_eigenvectors);
  const SparseParameterLayout trial_layout(trial.orbital_preparation_input);
  auto trial_chart = build_chart(trial, trial_layout, trial_gradient);
  const Eigen::VectorXd trial_projected_gradient =
      trial_chart->project_reduced_gradient(
          trial_layout.gather_from_full(
              trial_gradient.sparse_orbital_energy_gradient));
  const double actual_decrease =
      accepted.scf_result.total_energy -
      trial_gradient.scf_result.total_energy;
  const double fresh_predicted_decrease =
      -projected.reduced_gradient.dot(step.reduced_step) -
      0.5 * step.reduced_step.dot(fresh_hs);
  double directional_hvp_fd_relative =
      std::numeric_limits<double>::quiet_NaN();
  double directional_energy_fd_relative =
      std::numeric_limits<double>::quiet_NaN();
  if (options.finite_difference_step > 0.0) {
    const double h = options.finite_difference_step;
    const Eigen::VectorXd direction =
        step.reduced_step / metric.norm(step.reduced_step);
    const auto evaluate_displaced = [&](double displacement) {
      VbScfInput point = input;
      point.orbital_preparation_input = chart->retract_step(
          input.orbital_preparation_input, direction, displacement);
      const auto evaluated =
          evaluator.evaluate_without_reference_energy_gradient(
              point, {0}, {1.0}, loaded.nuclear_repulsion_energy,
              options.eigensolver, accuracy, no_initial_eigenvectors);
      return std::pair{
          chart->project_reduced_gradient(
              layout.gather_from_full(
                  evaluated.sparse_orbital_energy_gradient)),
          evaluated.scf_result.total_energy};
    };
    const auto [plus_gradient, plus_energy] = evaluate_displaced(h);
    const auto [minus_gradient, minus_energy] = evaluate_displaced(-h);
    const Eigen::VectorXd analytic = fresh_hs /
        metric.norm(step.reduced_step);
    const Eigen::VectorXd finite_difference =
        (plus_gradient - minus_gradient) / (2.0 * h);
    directional_hvp_fd_relative = relative_norm(
        analytic - finite_difference, finite_difference);
    const double analytic_curvature = direction.dot(analytic);
    const double energy_curvature =
        (plus_energy - 2.0 * accepted.scf_result.total_energy +
         minus_energy) / (h * h);
    directional_energy_fd_relative =
        std::abs(analytic_curvature - energy_curvature) /
        std::max(1.0, std::abs(energy_curvature));
  }
  const auto diagnostics = hvp.diagnostics();

  std::cout << std::setprecision(15)
            << "eigensolver = " << structure_eigensolver_name(options.eigensolver) << '\n'
            << "step_model = "
            << (lbfgs_preconditioner == nullptr
                    ? "direct_newton"
                    : "lbfgs_plus_exact_residual_correction") << '\n'
            << "lbfgs_history_pairs = "
            << packed_secant_history.size() << '\n'
            << "lbfgs_accepted_pairs = "
            << (lbfgs_preconditioner == nullptr
                    ? 0
                    : lbfgs_preconditioner->size()) << '\n'
            << "lbfgs_baseline_step_norm = "
            << (lbfgs_preconditioner == nullptr
                    ? std::numeric_limits<double>::quiet_NaN()
                    : metric.norm(lbfgs_baseline_step)) << '\n'
            << "lbfgs_baseline_kkt_relative = "
            << lbfgs_baseline_kkt_relative << '\n'
            << "correction_model_restarts = "
            << correction_model_restarts << '\n'
            << "newton_correction_step_norm = "
            << correction_step_norm << '\n'
            << "reduced_dimension = " << chart->reduced_size() << '\n'
            << "subspace_dimension = " << step.subspace_dimension << '\n'
            << "target_kkt_relative = " << target_kkt_relative_residual << '\n'
            << "stop_reason = " << stop_reason_name(step.stop_reason) << '\n'
            << "source_gradient_l2 = " << projected.reduced_gradient.stableNorm() << '\n'
            << "trial_gradient_l2 = " << trial_projected_gradient.stableNorm() << '\n'
            << "trial_gradient_inf = " << trial_projected_gradient.cwiseAbs().maxCoeff() << '\n'
            << "trust_radius = " << options.trust_radius << '\n'
            << "step_norm = " << metric.norm(step.reduced_step) << '\n'
            << "shift = " << step.trust_region_shift << '\n'
            << "boundary = " << step.reached_boundary << '\n'
            << "negative_curvature = " << step.encountered_negative_curvature << '\n'
            << "cached_kkt_relative = " << step.model_kkt_relative_residual << '\n'
            << "fresh_kkt_relative = "
            << relative_norm(fresh_residual, projected.reduced_gradient) << '\n'
            << "projected_fresh_kkt_relative = "
            << projected_residual_relative << '\n'
            << "cache_image_vs_fresh_relative = "
            << (cached_hs.size() == fresh_hs.size()
                    ? relative_norm(fresh_hs - cached_hs, fresh_hs)
                    : std::numeric_limits<double>::quiet_NaN()) << '\n'
            << "cached_recomputed_kkt_relative = "
            << (cached_residual.size() == fresh_residual.size()
                    ? relative_norm(cached_residual, projected.reduced_gradient)
                    : std::numeric_limits<double>::quiet_NaN()) << '\n'
            << "predicted_decrease = " << step.predicted_decrease << '\n'
            << "fresh_predicted_decrease = " << fresh_predicted_decrease << '\n'
            << "actual_decrease = " << actual_decrease << '\n'
            << "trust_ratio = " << actual_decrease / fresh_predicted_decrease << '\n'
            << "hvp_directions = " << diagnostics.apply_count << '\n'
            << "hvp_seconds = " << diagnostics.total_apply_wall_time_seconds << '\n'
            << "solve_seconds = " << solve_seconds << '\n'
            << "max_structure_response_relative_residual = "
            << diagnostics.max_structure_response_relative_residual << '\n'
            << "directional_hvp_fd_relative = "
            << directional_hvp_fd_relative << '\n'
            << "directional_energy_fd_relative = "
            << directional_energy_fd_relative << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run_audit(parse_options(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "audit_newton_step: " << error.what() << '\n';
    return 1;
  }
}
