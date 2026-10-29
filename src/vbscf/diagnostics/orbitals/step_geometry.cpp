#include "vbscf/diagnostics/orbitals/step_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "mcscf/optimization/trace.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/preparation/input.hpp"

namespace xmvb::vb::diagnostics {

OrbitalStepGeometry measure_orbital_step(const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                         const Eigen::Ref<const Eigen::VectorXd>& displacement,
                                         const Eigen::Ref<const Eigen::MatrixXd>& overlap) {
    if (coefficients.size() != displacement.size() || overlap.rows() != coefficients.size() ||
        overlap.cols() != coefficients.size())
        throw std::invalid_argument("orbital step geometry dimensions disagree");
    const double norm = std::sqrt(coefficients.dot(overlap * coefficients));
    if (!(norm > 0.0) || !std::isfinite(norm))
        throw std::invalid_argument("orbital step geometry has invalid origin norm");
    const Eigen::VectorXd x = coefficients / norm;
    const Eigen::VectorXd d = displacement / norm;
    OrbitalStepGeometry result;
    result.alpha = x.dot(overlap * d);
    const Eigen::VectorXd tangent = d - result.alpha * x;
    // Form the tangent norm directly: ||d||^2-alpha^2 loses accuracy for
    // nearly radial displacements. No AO matrix is copied or factored here.
    result.theta = std::sqrt(std::max(0.0, tangent.dot(overlap * tangent)));
    result.step_norm = std::hypot(result.alpha, result.theta);
    result.norm_ratio = std::hypot(1.0 + result.alpha, result.theta);
    result.actual_angle = std::atan2(result.theta, 1.0 + result.alpha);
    return result;
}

bool orbital_step_audit_enabled() {
    static const bool enabled = [] {
        const char* flag = std::getenv("XMVB_CIAH_ORBITAL_STEP_AUDIT");
        return flag && std::strcmp(flag, "1") == 0;
    }();
    return enabled && mcscf::diagnostics::ciah_event_trace_enabled();
}

void audit_orbital_step(const OrbitalPreparationInput& input,
                        const OrbitalChart& chart,
                        const Eigen::VectorXd& reduced_step,
                        const char* stage,
                        int micro,
                        const Eigen::VectorXd& base_reduced) {
    const Eigen::VectorXd raw = chart.expand_retract_input_tangent(input, reduced_step);
    Eigen::VectorXd offset;
    if (base_reduced.size())
        offset = chart.expand_retract_input_tangent(input, base_reduced);
    const int n_inactive = (input.n_total_electrons - input.n_active_electrons) / 2;
    const int n_occupied = n_inactive + input.n_active_orbitals;
    std::ofstream origin_file, trial_file;
    std::filesystem::path origin_path, trial_path;
    const char* folder = std::getenv("XMVB_CIAH_ORBITAL_PATH");
    if (folder && *folder &&
        (std::strncmp(stage, "micro_", 6) != 0 || std::strcmp(stage, "micro_increment") == 0)) {
        static int sequence = 0;
        ++sequence;
        std::filesystem::create_directories(folder);
        std::ostringstream name;
        name << "macro-" << std::setw(3) << std::setfill('0')
             << mcscf::diagnostics::ciah_trace_context.keyframe;
        origin_path = std::filesystem::path(folder) / (name.str() + "-origin.bin");
        name << '-' << stage << '-' << std::setw(4) << sequence;
        trial_path = std::filesystem::path(folder) / (name.str() + ".bin");
        if (!std::filesystem::exists(origin_path)) {
            origin_file.open(origin_path, std::ios::binary);
            if (!origin_file)
                throw std::runtime_error("cannot write orbital origin snapshot");
        }
        trial_file.open(trial_path, std::ios::binary);
        if (!trial_file)
            throw std::runtime_error("cannot write orbital path snapshot");
    }
    // Two AO vectors are reused serially. The shared AO overlap is borrowed;
    // neither a per-orbital metric copy nor a per-thread workspace is needed.
    Eigen::VectorXd b(input.n_basis_functions), delta(input.n_basis_functions);
    for (int orbital = 0; orbital < n_occupied; ++orbital) {
        b.setZero();
        delta.setZero();
        const int count = stored_sparse_orbital_coefficient_count(input, orbital);
        for (int slot = 0; slot < count; ++slot) {
            const auto flat = orbital * input.n_basis_functions + slot;
            const int basis = input.orbital_basis_index_table[flat] - 1;
            b[basis] = input.orbital_value_table[flat] + (offset.size() ? offset[flat] : 0.0);
            delta[basis] = raw[flat];
        }
        const auto geometry = measure_orbital_step(b, delta, input.ao_overlap_matrix);
        if (origin_file.is_open())
            origin_file.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(double));
        if (trial_file.is_open()) {
            const Eigen::VectorXd trial = b + delta;
            trial_file.write(reinterpret_cast<const char*>(trial.data()),
                             trial.size() * sizeof(double));
        }
        mcscf::diagnostics::ciah_trace(
            "orbital_step_geometry",
            "stage=%s,micro=%d,orbital=%d,role=%s,coefficients=%d,alpha=%.17g,theta=%.17g,"
            "step_norm=%.17g,actual_angle=%.17g,norm_ratio=%.17g",
            stage,
            micro,
            orbital + 1,
            orbital < n_inactive ? "inactive" : "active",
            count,
            geometry.alpha,
            geometry.theta,
            geometry.step_norm,
            geometry.actual_angle,
            geometry.norm_ratio);
    }
    if (trial_file.is_open()) {
        trial_file.close();
        if (origin_file.is_open())
            origin_file.close();
        if (!trial_file || !origin_file)
            throw std::runtime_error("orbital path snapshot write failed");
        mcscf::diagnostics::ciah_trace(
            "orbital_step_snapshot",
            "stage=%s,micro=%d,n_bf=%zu,n_core=%d,n_active=%zu,origin=%s,trial=%s",
            stage,
            micro,
            input.n_basis_functions,
            n_inactive,
            input.n_active_orbitals,
            origin_path.c_str(),
            trial_path.c_str());
    }
}

void audit_ciah_state(const OrbitalPreparationInput& input,
                      const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
                      const char* stage,
                      double energy) {
    const char* folder = std::getenv("XMVB_CIAH_ORBITAL_PATH");
    if (!orbital_step_audit_enabled() || !folder || !*folder)
        return;
    static std::size_t sequence = 0;
    std::filesystem::create_directories(folder);
    std::ostringstream name;
    name << "state-" << std::setw(3) << std::setfill('0')
         << mcscf::diagnostics::ciah_trace_context.keyframe << '-' << stage << '-' << ++sequence
         << ".bin";
    const auto path = std::filesystem::path(folder) / name.str();
    std::ofstream file(path, std::ios::binary);
    const std::uint64_t dimensions[] = {input.n_basis_functions,
                                        input.n_orbitals,
                                        static_cast<std::uint64_t>(coefficients.rows()),
                                        static_cast<std::uint64_t>(coefficients.cols())};
    file.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    Eigen::MatrixXd orbitals = Eigen::MatrixXd::Zero(input.n_basis_functions, input.n_orbitals);
    for (std::size_t i = 0; i < input.n_orbitals; ++i)
        for (int j = 0; j < stored_sparse_orbital_coefficient_count(input, i); ++j) {
            const auto slot = i * input.n_basis_functions + j;
            orbitals(input.orbital_basis_index_table[slot] - 1, i) =
                input.orbital_value_table[slot];
        }
    file.write(reinterpret_cast<const char*>(orbitals.data()), orbitals.size() * sizeof(double));
    file.write(reinterpret_cast<const char*>(coefficients.data()),
               coefficients.size() * sizeof(double));
    if (!file)
        throw std::runtime_error("cannot write CIAH state snapshot");
    mcscf::diagnostics::ciah_trace(
        "ciah_state_snapshot", "stage=%s,energy=%.17g,path=%s", stage, energy, path.c_str());
}

} // namespace xmvb::vb::diagnostics
