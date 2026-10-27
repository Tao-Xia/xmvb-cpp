#include "vbscf/orbitals/charts/chart.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>

#include "vbscf/optimization/preconditioners/hessian_diagonal.hpp"

namespace xmvb::vb {

bool OrbitalChart::supports_rotation_frame(int n_bf) const {
    // Full-AO source mixing is shared by complete and selected expansions.
    // Selected expansions retain the internal active tangent constraints.
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
    Eigen::MatrixXd internal_normals(input.n_active_orbitals, input.n_active_orbitals);
    rotation_diagonals_[0] =
        model.rotation_diagonal.block(n_core, 0, input.n_basis_functions - n_core, n_core);
    const int active_start = active_subspace_gauge_ ? n_occ : n_core;
    rotation_diagonals_[1] = model.rotation_diagonal.block(
        active_start, n_core, input.n_basis_functions - active_start, input.n_active_orbitals);
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
            const int start = core ? n_core : active_start;
            const int width = input.n_basis_functions - start;
            const int tangent_width = width - (!core && !active_subspace_gauge_ ? 1 : 0);
            if (orbital.local_reduced_size != tangent_width)
                throw std::runtime_error("full-AO rotation map dimension mismatch");
            Eigen::VectorXd raw = Eigen::VectorXd::Zero(input.n_basis_functions);
            for (int row = 0; row < input.n_basis_functions; ++row)
                raw[row] =
                    input
                        .orbital_value_table[orbital.orbital_index * input.n_basis_functions + row];
            const Eigen::VectorXd s_raw = input.ao_overlap_matrix * raw;
            const double norm = std::sqrt(raw.dot(s_raw));
            if (!core && !active_subspace_gauge_)
                internal_normals.col(orbital.orbital_index - n_core).noalias() =
                    model.mo_coefficients.middleCols(n_core, input.n_active_orbitals).transpose() *
                    s_raw / norm;
            const Eigen::MatrixXd forward = orbital.ah_rotation_components.bottomRows(width);
            // L=||b|| Q_ext E for an S-horizontal lift. Recover E^{-1} from the
            // shared AO-metric Gram and existing frame map, without another AO
            // multiplication, factorization or explicit whitening.
            if (!core && !active_subspace_gauge_) {
                // The selected active map is rectangular. Solve its right
                // inverse directly, without squaring its condition number.
                const Eigen::HouseholderQR<Eigen::MatrixXd> factor(forward);
                orbital.ah_frame_inverse = factor.solve(Eigen::MatrixXd::Identity(width, width));
            } else {
                orbital.ah_frame_inverse.noalias() =
                    norm * norm * orbital.inverse_raw_tangent_gram * forward.transpose();
            }
            const Eigen::MatrixXd error = orbital.ah_frame_inverse * forward -
                                          Eigen::MatrixXd::Identity(tangent_width, tangent_width);
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
    active_internal_bases_.clear();
    active_internal_eigenvalues_.resize(0);
    active_internal_eigenvectors_.resize(0, 0);
    const int m = input.n_active_orbitals;
    if (!active_subspace_gauge_ && m > 1) {
        // Y_i is tangent to its own normalized orbital, not to the entire
        // active span. Keep m(m-1) internal directions; no CAS closure is used.
        const int dim = m * (m - 1);
        Eigen::MatrixXd map = Eigen::MatrixXd::Zero(m * m, dim);
        const auto& source = model.active_inverse_transform;
        for (int i = 0; i < m; ++i) {
            const Eigen::MatrixXd normal = internal_normals.col(i);
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(normal);
            const Eigen::MatrixXd q = qr.householderQ() * Eigen::MatrixXd::Identity(m, m);
            active_internal_bases_.push_back(q.rightCols(m - 1));
            for (int k = 0; k < m - 1; ++k) {
                const Eigen::MatrixXd image = active_internal_bases_.back().col(k) * source.row(i);
                map.col(i * (m - 1) + k) = Eigen::Map<const Eigen::VectorXd>(image.data(), m * m);
            }
        }
        const Eigen::MatrixXd d = rotation_diagonals_[1].topRows(m);
        const Eigen::VectorXd weights = Eigen::Map<const Eigen::VectorXd>(d.data(), m * m);
        const Eigen::MatrixXd h = map.transpose() * weights.asDiagonal() * map;
        const Eigen::MatrixXd metric = map.transpose() * map;
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(h, metric);
        if (spectrum.info() != Eigen::Success)
            throw std::runtime_error("selected active diagonal projection failed");
        active_internal_eigenvalues_ = spectrum.eigenvalues();
        active_internal_eigenvectors_ = spectrum.eigenvectors();
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
        const int m = rotation_diagonals_[1].cols();
        const bool selected = group == 1 && !active_subspace_gauge_;
        const int offset = selected ? m : 0;
        Eigen::MatrixXd internal;
        if (selected) {
            internal = Eigen::MatrixXd::Zero(m, m);
            if (m > 1) {
                Eigen::VectorXd rhs(m * (m - 1));
                for (int i = 0; i < m; ++i)
                    rhs.segment(i * (m - 1), m - 1) =
                        active_internal_bases_[i].transpose() * sources[group].col(i).head(m);
                Eigen::VectorXd denominator = active_internal_eigenvalues_.array() - eigenvalue;
                for (Eigen::Index k = 0; k < denominator.size(); ++k)
                    if (std::abs(denominator[k]) < 1e-8)
                        denominator[k] = 1e-8;
                const Eigen::VectorXd response =
                    active_internal_eigenvectors_ *
                    (active_internal_eigenvectors_.transpose() * rhs).cwiseQuotient(denominator);
                for (int i = 0; i < m; ++i)
                    internal.col(i) =
                        active_internal_bases_[i] * response.segment(i * (m - 1), m - 1);
            }
        }
        const int rows = sources[group].rows() - offset;
        Eigen::MatrixXd frame =
            sources[group].bottomRows(rows) * rotation_source_inverses_[group].transpose();
        for (Eigen::Index column = 0; column < frame.cols(); ++column)
            for (Eigen::Index row = 0; row < frame.rows(); ++row) {
                double denominator = rotation_diagonals_[group](row + offset, column) - eigenvalue;
                if (std::abs(denominator) < 1e-8)
                    denominator = 1e-8;
                frame(row, column) /= denominator;
            }
        sources[group].bottomRows(rows).noalias() = frame * rotation_source_inverses_[group];
        if (selected)
            sources[group].topRows(m) = internal;
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
