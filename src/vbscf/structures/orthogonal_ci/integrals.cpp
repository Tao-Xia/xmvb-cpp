#include "vbscf/structures/orthogonal_ci/integrals.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <Eigen/Cholesky>

#include "core/pair_index.hpp"
#include "core/timing.hpp"
#include "integrals/active/two_electron.hpp"
#include "integrals/active/transform.hpp"
#include "integrals/active/ri_transform.hpp"

namespace xmvb::vb {
namespace {

Eigen::MatrixXd unpack_two_electron_kernel(const std::vector<double>& packed,
                                           int n_active_orbitals) {
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_active_orbitals);
    if (packed.size() !=
        xmvb::integrals::packed_active_two_electron_integral_count(n_active_orbitals)) {
        throw std::invalid_argument(
            "packed active two-electron direction has incompatible dimensions");
    }
    Eigen::MatrixXd result(n_pairs, n_pairs);
    for (int column = 0; column < n_pairs; ++column) {
        for (int row = 0; row < n_pairs; ++row) {
            result(row, column) =
                packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(row, column)];
        }
    }
    return result;
}

Eigen::MatrixXd
apply_ri_factor_generator(const Eigen::Ref<const Eigen::MatrixXd>& factors,
                          const Eigen::Ref<const Eigen::MatrixXd>& relative_transform) {
    const int n = static_cast<int>(relative_transform.rows());
    const int n_aux = static_cast<int>(factors.rows());
    if (n <= 0 || relative_transform.cols() != n || n_aux <= 0 ||
        factors.cols() != xmvb::integrals::packed_active_pair_count(n)) {
        throw std::invalid_argument("RI factor generator has incompatible dimensions");
    }
    Eigen::MatrixXd direction(factors.rows(), factors.cols());
    const std::size_t elements_per_aux = 2ULL * n * n;
    const int tile_size = static_cast<int>(std::min<std::size_t>(
        n_aux,
        std::max<std::size_t>(1,
                              xmvb::workspace::available() / sizeof(double) / elements_per_aux)));
    for (int first = 0; first < n_aux; first += tile_size) {
        const int count = std::min(tile_size, n_aux - first);
        Eigen::MatrixXd stacked(count * n, n);
        for (int aux = 0; aux < count; ++aux) {
            for (int row = 0; row < n; ++row) {
                for (int column = 0; column < n; ++column) {
                    stacked(aux * n + row, column) =
                        factors(first + aux, xmvb::core::PairIndex::packed_pair_index(row, column));
                }
            }
        }
        const Eigen::MatrixXd right = stacked * relative_transform;
        for (int aux = 0; aux < count; ++aux) {
            for (int row = 0; row < n; ++row) {
                for (int column = 0; column <= row; ++column) {
                    direction(first + aux, xmvb::core::PairIndex::packed_pair_index(row, column)) =
                        -right(aux * n + row, column) - right(aux * n + column, row);
                }
            }
        }
    }
    return direction;
}

} // namespace

OrthogonalActiveIntegrals
orthogonalize_active_integrals(const std::vector<double>& active_overlap,
                               const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
                               const xmvb::integrals::ActiveSpaceTwoElectronResult& two_electron,
                               int n_active_orbitals) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ActiveFrame);
    if (n_active_orbitals <= 0 ||
        active_overlap.size() != static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals ||
        one_electron.rows() != n_active_orbitals || one_electron.cols() != n_active_orbitals ||
        !one_electron.allFinite()) {
        throw std::invalid_argument("invalid active-space orthogonalization input");
    }
    const Eigen::Map<const Eigen::MatrixXd> overlap(
        active_overlap.data(), n_active_orbitals, n_active_orbitals);
    OrthogonalActiveIntegrals result;
    result.input_pretransformed = two_electron.orbital_transform.size() != 0;
    if (result.input_pretransformed) {
        if (two_electron.orbital_transform.rows() != n_active_orbitals ||
            two_electron.orbital_transform.cols() != n_active_orbitals ||
            !two_electron.orbital_transform.allFinite()) {
            throw std::invalid_argument("pretransformed two-electron frame is invalid");
        }
        result.orbital_transform = two_electron.orbital_transform;
    } else {
        Eigen::LLT<Eigen::MatrixXd> cholesky(overlap);
        if (cholesky.info() != Eigen::Success) {
            throw std::runtime_error("active orbital overlap is not positive definite");
        }
        result.orbital_transform = cholesky.matrixU();
    }
    const Eigen::MatrixXd inverse_transform =
        result.orbital_transform.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_active_orbitals, n_active_orbitals));
    result.one_electron =
        result.input_pretransformed
            ? Eigen::MatrixXd(one_electron)
            : Eigen::MatrixXd(inverse_transform.transpose() * one_electron * inverse_transform);

    if (two_electron.representation ==
        xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
        result.ri_pair_factors = result.input_pretransformed
                                     ? two_electron.ri_active_pair_factors
                                     : xmvb::integrals::transform_ri_pair_factors(
                                           two_electron.ri_active_pair_factors, inverse_transform);
    }
    Eigen::MatrixXd original_kernel = unpack_two_electron_kernel(
        two_electron.packed_active_two_electron_integrals, n_active_orbitals);
    result.pair_kernel =
        result.input_pretransformed
            ? std::move(original_kernel)
            : xmvb::integrals::transform_pair_kernel(original_kernel, inverse_transform);
    return result;
}

OrthogonalActiveIntegrals orthogonalize_active_integral_direction(
    const OrthogonalActiveIntegrals& accepted,
    const std::vector<double>& overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron_direction,
    const std::vector<double>& packed_two_electron_direction,
    int n_active_orbitals,
    const Eigen::MatrixXd* ri_factor_direction) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ActiveFrameDirection);
    if (n_active_orbitals <= 0 || accepted.orbital_transform.rows() != n_active_orbitals ||
        accepted.orbital_transform.cols() != n_active_orbitals ||
        accepted.one_electron.rows() != n_active_orbitals ||
        accepted.one_electron.cols() != n_active_orbitals ||
        overlap_direction.size() !=
            static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals ||
        one_electron_direction.rows() != n_active_orbitals ||
        one_electron_direction.cols() != n_active_orbitals || !one_electron_direction.allFinite()) {
        throw std::invalid_argument("invalid orthogonal active-integral direction");
    }
    const Eigen::Map<const Eigen::MatrixXd> delta_overlap_raw(
        overlap_direction.data(), n_active_orbitals, n_active_orbitals);
    const Eigen::MatrixXd delta_overlap = 0.5 * (delta_overlap_raw + delta_overlap_raw.transpose());
    const Eigen::MatrixXd inverse_transform =
        accepted.orbital_transform.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_active_orbitals, n_active_orbitals));
    const Eigen::MatrixXd normalized_overlap_direction =
        inverse_transform.transpose() * delta_overlap * inverse_transform;
    Eigen::MatrixXd relative_transform =
        Eigen::MatrixXd::Zero(n_active_orbitals, n_active_orbitals);
    for (int column = 0; column < n_active_orbitals; ++column) {
        relative_transform(column, column) = 0.5 * normalized_overlap_direction(column, column);
        for (int row = 0; row < column; ++row) {
            relative_transform(row, column) = normalized_overlap_direction(row, column);
        }
    }

    OrthogonalActiveIntegrals result;
    result.orbital_transform = relative_transform * accepted.orbital_transform;
    result.input_pretransformed = accepted.input_pretransformed;
    result.one_electron = accepted.input_pretransformed
                              ? Eigen::MatrixXd(one_electron_direction)
                              : Eigen::MatrixXd(inverse_transform.transpose() *
                                                one_electron_direction * inverse_transform);
    result.one_electron.noalias() -= relative_transform.transpose() * accepted.one_electron +
                                     accepted.one_electron * relative_transform;

    if (ri_factor_direction) {
        Eigen::MatrixXd delta_factors;
        if (accepted.input_pretransformed) {
            if (ri_factor_direction->rows() != accepted.ri_pair_factors.rows() ||
                ri_factor_direction->cols() != accepted.ri_pair_factors.cols()) {
                throw std::invalid_argument(
                    "pretransformed RI factor direction has incompatible dimensions");
            }
            delta_factors = *ri_factor_direction +
                            apply_ri_factor_generator(accepted.ri_pair_factors, relative_transform);
        } else {
            delta_factors = xmvb::integrals::transform_ri_pair_factor_direction_from_orthogonal(
                accepted.ri_pair_factors,
                *ri_factor_direction,
                inverse_transform,
                relative_transform);
        }
        const auto& factors = accepted.ri_pair_factors;
        const Eigen::MatrixXd mixed = factors.transpose() * delta_factors;
        result.pair_kernel = mixed + mixed.transpose();
    } else {
        const Eigen::MatrixXd delta_original_kernel =
            unpack_two_electron_kernel(packed_two_electron_direction, n_active_orbitals);
        result.pair_kernel =
            accepted.input_pretransformed
                ? delta_original_kernel
                : xmvb::integrals::transform_pair_kernel(delta_original_kernel, inverse_transform);
        // The Cholesky generator acts on one orbital index at a time.
        result.pair_kernel +=
            xmvb::integrals::apply_pair_generator(accepted.pair_kernel, -relative_transform);
    }
    result.pair_kernel = 0.5 * (result.pair_kernel + result.pair_kernel.transpose()).eval();
    return result;
}

NonorthogonalActiveIntegralAdjoint backpropagate_orthogonal_active_integral_adjoint(
    const OrthogonalActiveIntegrals& accepted,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_one_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_pair_kernel_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& generator_gradient) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ActiveFrameAdjoint);
    const int n_orbitals = static_cast<int>(accepted.one_electron.rows());
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    if (n_orbitals <= 0 || accepted.one_electron.cols() != n_orbitals ||
        accepted.orbital_transform.rows() != n_orbitals ||
        accepted.orbital_transform.cols() != n_orbitals || accepted.pair_kernel.rows() != n_pairs ||
        accepted.pair_kernel.cols() != n_pairs ||
        orthogonal_one_electron_gradient.rows() != n_orbitals ||
        orthogonal_one_electron_gradient.cols() != n_orbitals ||
        orthogonal_pair_kernel_gradient.rows() != n_pairs ||
        orthogonal_pair_kernel_gradient.cols() != n_pairs ||
        generator_gradient.rows() != n_orbitals || generator_gradient.cols() != n_orbitals) {
        throw std::invalid_argument(
            "orthogonal active-integral adjoint has incompatible dimensions");
    }

    const Eigen::MatrixXd inverse_transform =
        accepted.orbital_transform.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_orbitals, n_orbitals));

    NonorthogonalActiveIntegralAdjoint result;
    result.one_electron =
        inverse_transform * orthogonal_one_electron_gradient * inverse_transform.transpose();
    result.one_electron = (0.5 * (result.one_electron + result.one_electron.transpose())).eval();
    result.pair_kernel = accepted.input_pretransformed
                             ? Eigen::MatrixXd(orthogonal_pair_kernel_gradient)
                             : xmvb::integrals::pullback_pair_kernel(
                                   orthogonal_pair_kernel_gradient, inverse_transform);
    result.pair_kernel = (0.5 * (result.pair_kernel + result.pair_kernel.transpose())).eval();

    Eigen::MatrixXd relative_gradient = generator_gradient;
    relative_gradient.noalias() -=
        accepted.one_electron * orthogonal_one_electron_gradient.transpose() +
        accepted.one_electron.transpose() * orthogonal_one_electron_gradient;
    relative_gradient -= xmvb::integrals::orbital_generator_adjoint(
        accepted.pair_kernel, orthogonal_pair_kernel_gradient);

    Eigen::MatrixXd normalized_overlap_gradient = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
    for (int column = 0; column < n_orbitals; ++column) {
        normalized_overlap_gradient(column, column) = 0.5 * relative_gradient(column, column);
        for (int row = 0; row < column; ++row) {
            const double value = 0.5 * relative_gradient(row, column);
            normalized_overlap_gradient(row, column) = value;
            normalized_overlap_gradient(column, row) = value;
        }
    }
    result.overlap =
        inverse_transform * normalized_overlap_gradient * inverse_transform.transpose();
    result.overlap = (0.5 * (result.overlap + result.overlap.transpose())).eval();
    return result;
}

NonorthogonalActiveIntegralAdjoint backpropagate_orthogonal_active_integral_adjoint_direction(
    const OrthogonalActiveIntegrals& accepted,
    const OrthogonalActiveIntegrals& direction,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_one_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_pair_kernel_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& generator_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_one_electron_gradient_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_pair_kernel_gradient_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& generator_gradient_direction) {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::ActiveFrameAdjointDirection);
    const int n_orbitals = static_cast<int>(accepted.one_electron.rows());
    const int n_pairs = xmvb::integrals::packed_active_pair_count(n_orbitals);
    if (n_orbitals <= 0 || direction.one_electron.rows() != n_orbitals ||
        direction.one_electron.cols() != n_orbitals || direction.pair_kernel.rows() != n_pairs ||
        direction.pair_kernel.cols() != n_pairs ||
        direction.orbital_transform.rows() != n_orbitals ||
        direction.orbital_transform.cols() != n_orbitals ||
        orthogonal_one_electron_gradient_direction.rows() != n_orbitals ||
        orthogonal_one_electron_gradient_direction.cols() != n_orbitals ||
        orthogonal_pair_kernel_gradient_direction.rows() != n_pairs ||
        orthogonal_pair_kernel_gradient_direction.cols() != n_pairs ||
        generator_gradient_direction.rows() != n_orbitals ||
        generator_gradient_direction.cols() != n_orbitals) {
        throw std::invalid_argument(
            "orthogonal active-integral adjoint direction has incompatible dimensions");
    }

    const Eigen::MatrixXd inverse_transform =
        accepted.orbital_transform.template triangularView<Eigen::Upper>().solve(
            Eigen::MatrixXd::Identity(n_orbitals, n_orbitals));
    const Eigen::MatrixXd relative_direction = direction.orbital_transform * inverse_transform;
    const Eigen::MatrixXd inverse_transform_direction = -inverse_transform * relative_direction;

    NonorthogonalActiveIntegralAdjoint result;
    result.one_electron = inverse_transform_direction * orthogonal_one_electron_gradient *
                              inverse_transform.transpose() +
                          inverse_transform * orthogonal_one_electron_gradient_direction *
                              inverse_transform.transpose() +
                          inverse_transform * orthogonal_one_electron_gradient *
                              inverse_transform_direction.transpose();
    result.one_electron = (0.5 * (result.one_electron + result.one_electron.transpose())).eval();
    if (accepted.input_pretransformed) {
        result.pair_kernel = orthogonal_pair_kernel_gradient_direction +
                             xmvb::integrals::pullback_pair_generator(
                                 orthogonal_pair_kernel_gradient, -relative_direction);
    } else {
        result.pair_kernel = xmvb::integrals::pullback_pair_kernel_direction(
            orthogonal_pair_kernel_gradient,
            orthogonal_pair_kernel_gradient_direction,
            inverse_transform,
            inverse_transform_direction);
    }
    result.pair_kernel = (0.5 * (result.pair_kernel + result.pair_kernel.transpose())).eval();

    Eigen::MatrixXd relative_gradient = generator_gradient;
    relative_gradient.noalias() -=
        accepted.one_electron * orthogonal_one_electron_gradient.transpose() +
        accepted.one_electron.transpose() * orthogonal_one_electron_gradient;
    relative_gradient -= xmvb::integrals::orbital_generator_adjoint(
        accepted.pair_kernel, orthogonal_pair_kernel_gradient);
    Eigen::MatrixXd relative_gradient_direction = generator_gradient_direction;
    relative_gradient_direction.noalias() -=
        direction.one_electron * orthogonal_one_electron_gradient.transpose() +
        accepted.one_electron * orthogonal_one_electron_gradient_direction.transpose() +
        direction.one_electron.transpose() * orthogonal_one_electron_gradient +
        accepted.one_electron.transpose() * orthogonal_one_electron_gradient_direction;
    relative_gradient_direction -= xmvb::integrals::orbital_generator_adjoint(
        direction.pair_kernel, orthogonal_pair_kernel_gradient);
    relative_gradient_direction -= xmvb::integrals::orbital_generator_adjoint(
        accepted.pair_kernel, orthogonal_pair_kernel_gradient_direction);

    Eigen::MatrixXd normalized_overlap_gradient = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
    Eigen::MatrixXd normalized_overlap_gradient_direction =
        Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
    for (int column = 0; column < n_orbitals; ++column) {
        normalized_overlap_gradient(column, column) = 0.5 * relative_gradient(column, column);
        normalized_overlap_gradient_direction(column, column) =
            0.5 * relative_gradient_direction(column, column);
        for (int row = 0; row < column; ++row) {
            const double value = 0.5 * relative_gradient(row, column);
            normalized_overlap_gradient(row, column) = value;
            normalized_overlap_gradient(column, row) = value;
            const double direction_value = 0.5 * relative_gradient_direction(row, column);
            normalized_overlap_gradient_direction(row, column) = direction_value;
            normalized_overlap_gradient_direction(column, row) = direction_value;
        }
    }
    result.overlap =
        inverse_transform_direction * normalized_overlap_gradient * inverse_transform.transpose() +
        inverse_transform * normalized_overlap_gradient_direction * inverse_transform.transpose() +
        inverse_transform * normalized_overlap_gradient * inverse_transform_direction.transpose();
    result.overlap = (0.5 * (result.overlap + result.overlap.transpose())).eval();
    return result;
}

} // namespace xmvb::vb
