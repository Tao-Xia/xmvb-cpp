#include "vbscf/orbitals/charts/chart.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>

#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

namespace xmvb::vb {

bool OrbitalChart::supports_rotation_frame(int n_bf) const {
    if (!active_subspace_gauge_)
        return false;
    // Only the complete, unconstrained full-AO quotient has a bijective frame
    // map. HAO/selected-space charts keep their constrained local model.
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals) {
            if (orbital.local_parameter_size != n_bf)
                return false;
            for (int row = 0; row < orbital.local_parameter_size; ++row)
                if (block.basis_function_indices[orbital.block_rows[row]] != row)
                    return false;
        }
    return true;
}

void OrbitalChart::build_rotation_inverse(const OrbitalPreparationInput& input,
                                          const AnalyticOrbitalDiagonal& model) {
    const int n_core = (input.n_total_electrons - input.n_active_electrons) / 2;
    const int n_occ = n_core + input.n_active_orbitals;
    rotation_core_count_ = n_core;
    rotation_diagonals_[0] =
        model.rotation_diagonal.block(n_core, 0, input.n_basis_functions - n_core, n_core);
    rotation_diagonals_[1] = model.rotation_diagonal.block(
        n_occ, n_core, input.n_basis_functions - n_occ, input.n_active_orbitals);
    const std::array<Eigen::MatrixXd, 2> sources{model.inactive_inverse_transform,
                                                 model.active_inverse_transform};
    for (int group = 0; group < 2; ++group) {
        if (!sources[group].rows())
            continue;
        const Eigen::FullPivLU<Eigen::MatrixXd> factor(sources[group]);
        if (!factor.isInvertible())
            throw std::runtime_error("singular analytic diagonal source transform");
        rotation_source_inverses_[group] = factor.inverse();
    }
    for (auto& block : block_bases_)
        for (auto& orbital : block.orbitals) {
            const bool core = orbital.orbital_index < n_core;
            const int start = core ? n_core : n_occ;
            const int width = input.n_basis_functions - start;
            if (orbital.local_reduced_size != width)
                throw std::runtime_error("full-AO rotation map dimension mismatch");
            Eigen::VectorXd raw = Eigen::VectorXd::Zero(input.n_basis_functions);
            for (int row = 0; row < input.n_basis_functions; ++row)
                raw[row] =
                    input
                        .orbital_value_table[orbital.orbital_index * input.n_basis_functions + row];
            const double norm = std::sqrt(raw.dot(input.ao_overlap_matrix * raw));
            const Eigen::MatrixXd forward = orbital.ah_rotation_components.bottomRows(width);
            // L=||b|| Q_ext E for an S-horizontal lift. Recover E^{-1} from the
            // shared AO-metric Gram and existing frame map, without another AO
            // multiplication, factorization or explicit whitening.
            orbital.ah_frame_inverse.noalias() =
                norm * norm * orbital.inverse_raw_tangent_gram * forward.transpose();
            const Eigen::MatrixXd error =
                forward * orbital.ah_frame_inverse - Eigen::MatrixXd::Identity(width, width);
            const double bound = 512.0 * std::numeric_limits<double>::epsilon() *
                                 std::max(1, static_cast<int>(input.n_basis_functions)) *
                                 std::max(1.0, forward.norm() * orbital.ah_frame_inverse.norm());
            if (!orbital.ah_frame_inverse.allFinite() || error.norm() > bound)
                throw std::runtime_error(
                    "analytic rotation inverse failed its residual certificate");
            // The full-frame inverse replaces the signed local spectral inverse;
            // retain the independent positive block needed by P-LBFGS.
            orbital.ah_block_eigenvalues.resize(0);
            orbital.ah_block_eigenvectors.resize(0, 0);
        }
    has_rotation_inverse_ = true;
}

Eigen::VectorXd OrbitalChart::apply_inverse_rotation_diagonal(const Eigen::VectorXd& covector,
                                                              double eigenvalue) const {
    std::array<Eigen::MatrixXd, 2> sources;
    for (int group = 0; group < 2; ++group)
        sources[group] = Eigen::MatrixXd::Zero(rotation_diagonals_[group].rows(),
                                               rotation_diagonals_[group].cols());
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals) {
            const int group = orbital.orbital_index < rotation_core_count_ ? 0 : 1;
            const int column = orbital.orbital_index - (group ? rotation_core_count_ : 0);
            sources[group].col(column).noalias() =
                orbital.ah_frame_inverse.transpose() *
                covector.segment(orbital.local_reduced_offset, orbital.local_reduced_size);
        }
    for (int group = 0; group < 2; ++group) {
        if (!sources[group].size())
            continue;
        Eigen::MatrixXd frame = sources[group] * rotation_source_inverses_[group].transpose();
        for (Eigen::Index column = 0; column < frame.cols(); ++column)
            for (Eigen::Index row = 0; row < frame.rows(); ++row) {
                double denominator = rotation_diagonals_[group](row, column) - eigenvalue;
                // Identical signed denominator protection to the existing AH policy.
                if (std::abs(denominator) < 1.0e-8)
                    denominator = 1.0e-8;
                frame(row, column) /= denominator;
            }
        sources[group].noalias() = frame * rotation_source_inverses_[group];
    }
    Eigen::VectorXd result(covector.size());
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals) {
            const int group = orbital.orbital_index < rotation_core_count_ ? 0 : 1;
            const int column = orbital.orbital_index - (group ? rotation_core_count_ : 0);
            result.segment(orbital.local_reduced_offset, orbital.local_reduced_size).noalias() =
                orbital.ah_frame_inverse * sources[group].col(column);
        }
    if (!result.allFinite())
        throw std::runtime_error("non-finite full-frame AH inverse");
    return result;
}

double OrbitalChart::maximum_frame_component(const Eigen::VectorXd& step) const {
    std::array<Eigen::MatrixXd, 2> rotations;
    for (int group = 0; group < 2; ++group)
        rotations[group] = Eigen::MatrixXd::Zero(rotation_diagonals_[group].rows(),
                                                 rotation_diagonals_[group].cols());
    for (const auto& block : block_bases_)
        for (const auto& orbital : block.orbitals) {
            const int group = orbital.orbital_index < rotation_core_count_ ? 0 : 1;
            const Eigen::VectorXd external =
                orbital.ah_rotation_components.bottomRows(rotations[group].rows()) *
                step.segment(orbital.local_reduced_offset, orbital.local_reduced_size);
            rotations[group].noalias() += external * orbital.ah_source_weights.transpose();
        }
    double maximum = 0.0;
    for (const auto& rotation : rotations)
        if (rotation.size())
            maximum = std::max(maximum, rotation.cwiseAbs().maxCoeff());
    return maximum;
}

} // namespace xmvb::vb
