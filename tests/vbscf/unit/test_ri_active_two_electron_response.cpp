#include <algorithm>
#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "core/pair_index.hpp"
#include "integrals/response/ri.hpp"
#include "integrals/response/ri_internal.hpp"
#include "integrals/active/pair_map.hpp"
#include "integrals/active/ri.hpp"

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool is_close(const Eigen::Ref<const Eigen::MatrixXd>& actual,
              const Eigen::Ref<const Eigen::MatrixXd>& expected,
              double tolerance) {
    return (actual - expected).norm() <= tolerance * std::max(1.0, expected.norm());
}

Eigen::MatrixXd build_pair_map(const Eigen::Ref<const Eigen::MatrixXd>& coefficients) {
    const int n_bf = static_cast<int>(coefficients.rows());
    const int n_active = static_cast<int>(coefficients.cols());
    const int n_ao_pairs = n_bf * (n_bf + 1) / 2;
    const int n_active_pairs = n_active * (n_active + 1) / 2;
    Eigen::MatrixXd pair_map(n_ao_pairs, n_active_pairs);
    int ao_pair = 0;
    for (int first_basis = 0; first_basis < n_bf; ++first_basis) {
        for (int second_basis = 0; second_basis <= first_basis; ++second_basis, ++ao_pair) {
            int active_pair = 0;
            for (int first_active = 0; first_active < n_active; ++first_active) {
                for (int second_active = 0; second_active <= first_active;
                     ++second_active, ++active_pair) {
                    double value = coefficients(first_basis, first_active) *
                                   coefficients(second_basis, second_active);
                    if (first_basis != second_basis) {
                        value += coefficients(second_basis, first_active) *
                                 coefficients(first_basis, second_active);
                    }
                    pair_map(ao_pair, active_pair) = value;
                }
            }
        }
    }
    return pair_map;
}

xmvb::integrals::ActiveSpaceTwoElectronResult
make_result(const Eigen::Ref<const Eigen::MatrixXd>& coefficients,
            const xmvb::integrals::RiAoFactorization& factorization) {
    xmvb::integrals::ActiveSpaceTwoElectronResult result;
    result.representation =
        xmvb::integrals::ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity;
    result.n_auxiliary_functions = factorization.n_auxiliary_functions;
    result.dense_active_coefficients = coefficients;
    result.ri_active_pair_factors =
        factorization.metric_whitened_ao_pair_factors * build_pair_map(coefficients);
    return result;
}

std::vector<double> pack_gram(const Eigen::Ref<const Eigen::MatrixXd>& factors) {
    const Eigen::MatrixXd gram = factors.transpose() * factors;
    const int n_pairs = static_cast<int>(gram.rows());
    std::vector<double> packed(static_cast<std::size_t>(n_pairs) * (n_pairs + 1) / 2, 0.0);
    for (int first = 0; first < n_pairs; ++first) {
        for (int second = 0; second <= first; ++second) {
            packed[xmvb::core::PairIndex::packed_pair_of_pairs_index(first, second)] =
                gram(first, second);
        }
    }
    return packed;
}

double packed_dot(const std::vector<double>& left, const std::vector<double>& right) {
    require(left.size() == right.size(), "packed dot size mismatch");
    double result = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
}

void check_direct_factor_direction() {
    constexpr int n_bf = 50;
    constexpr int n_active = 24;
    constexpr int n_auxiliary = 7;
    constexpr int n_ao_pairs = n_bf * (n_bf + 1) / 2;

    xmvb::integrals::RiAoFactorization factorization;
    factorization.n_basis_functions = n_bf;
    factorization.n_auxiliary_functions = n_auxiliary;
    factorization.n_packed_ao_pairs = n_ao_pairs;
    factorization.metric_whitened_ao_pair_factors.resize(n_auxiliary, n_ao_pairs);
    for (Eigen::Index index = 0; index < factorization.metric_whitened_ao_pair_factors.size();
         ++index) {
        factorization.metric_whitened_ao_pair_factors.data()[index] =
            std::sin(0.013 * static_cast<double>(index + 1));
    }

    Eigen::MatrixXd coefficients(n_bf, n_active);
    Eigen::MatrixXd direction(n_bf, n_active);
    for (Eigen::Index index = 0; index < coefficients.size(); ++index) {
        coefficients.data()[index] = std::cos(0.021 * static_cast<double>(index + 2));
        direction.data()[index] = std::sin(0.017 * static_cast<double>(index + 3));
    }

    const auto accepted_result = make_result(coefficients, factorization);
    const auto transformed =
        xmvb::integrals::transform_ri_active_integrals(factorization, coefficients);
    require(
        is_close(transformed.ri_active_pair_factors, accepted_result.ri_active_pair_factors, 2e-13),
        "two-stage RI forward transform differs from the independent pair-map oracle");
    const auto shared_cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, transformed, n_active);
    require(transformed.ri_one_index_factors &&
                shared_cache.accepted_one_index_factors == transformed.ri_one_index_factors,
            "RI response rebuilt or copied the forward one-index transform");
    const auto cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, accepted_result, n_active);
    const Eigen::MatrixXd actual =
        xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(cache, direction);
    xmvb::integrals::PackedOrbitalPairMapMatrix pair_direction;
    xmvb::integrals::build_packed_orbital_pair_map_directional_derivative(
        coefficients, direction, &pair_direction);
    const Eigen::MatrixXd expected = factorization.metric_whitened_ao_pair_factors * pair_direction;
    require(is_close(actual, expected, 2.0e-13),
            "direct RI active-pair factor direction changed the packed result");
    const Eigen::MatrixXd second_direction = -0.37 * direction + 0.11 * coefficients;
    const auto batch = xmvb::integrals::compute_ri_active_pair_factor_directional_derivative_batch(
        cache, {direction, second_direction});
    require(batch.size() == 2 && is_close(batch[0], actual, 2.0e-13),
            "direct RI active-pair batch changed its first direction");
    require(is_close(batch[1],
                     xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(
                         cache, second_direction),
                     2.0e-13),
            "direct RI active-pair batch changed its second direction");
}

/** @brief Independent scalar oracle for shared and strided factor pullbacks. */
void check_rectangular_pullback() {
    constexpr int n_bf = 17, n_active = 5, count = 5, pairs = 15;
    xmvb::integrals::RiActiveTwoElectronResponseCache cache;
    cache.n_basis_functions = n_bf;
    cache.n_active_orbitals = n_active;
    for (int a = 0; a < n_active; ++a)
        for (int b = 0; b <= a; ++b) {
            cache.active_pair_first_indices.push_back(a);
            cache.active_pair_second_indices.push_back(b);
        }
    const auto values = [](int rows, int cols) {
        Eigen::MatrixXd matrix(rows, cols);
        for (Eigen::Index i = 0; i < matrix.size(); ++i)
            matrix.data()[i] = std::sin(0.19 * (i + 1));
        return matrix;
    };
    const Eigen::MatrixXd adjoint = values(12, pairs);
    const Eigen::MatrixXd delta_adjoint = -0.31 * adjoint;
    const Eigen::MatrixXd initial = values(n_bf, n_active);
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    const int saved_dynamic = omp_get_dynamic();
#endif
    for (int threads : {1, 2, 4, 16}) {
#ifdef _OPENMP
        omp_set_num_threads(threads);
        omp_set_dynamic(threads == 16);
#else
        (void)threads;
#endif
        for (bool strided : {false, true}) {
            const int stride = strided ? 9 : count, offset = strided ? 3 : 0;
            const int delta_stride = strided ? 8 : count;
            const int delta_offset = strided ? 1 : 0;
            const auto image = values(stride * n_active, n_bf);
            const auto delta_image = values(delta_stride * n_active, n_bf);
            Eigen::MatrixXd expected = initial, fixed = initial;
            for (int p = 0; p < pairs; ++p) {
                const int a = cache.active_pair_first_indices[p];
                const int b = cache.active_pair_second_indices[p];
                for (int k = 0; k < count; ++k)
                    for (int mu = 0; mu < n_bf; ++mu) {
                        expected(mu, b) += image(a * stride + offset + k, mu) * adjoint(4 + k, p);
                        expected(mu, a) += image(b * stride + offset + k, mu) * adjoint(4 + k, p);
                        fixed(mu, b) +=
                            delta_image(a * delta_stride + delta_offset + k, mu) *
                                adjoint(4 + k, p) +
                            image(a * stride + offset + k, mu) * delta_adjoint(4 + k, p);
                        fixed(mu, a) +=
                            delta_image(b * delta_stride + delta_offset + k, mu) *
                                adjoint(4 + k, p) +
                            image(b * stride + offset + k, mu) * delta_adjoint(4 + k, p);
                    }
            }
            for (std::size_t budget : {sizeof(double) * n_active * n_active, std::size_t{65536}}) {
                auto actual = initial, actual_fixed = initial;
                xmvb::integrals::detail::accumulate_ri_factor_tile(
                    cache, image, stride, offset, adjoint, 4, count, budget, actual);
                xmvb::integrals::detail::accumulate_ri_fixed_factor_tile(cache,
                                                                         image,
                                                                         stride,
                                                                         offset,
                                                                         delta_image,
                                                                         delta_stride,
                                                                         delta_offset,
                                                                         adjoint,
                                                                         delta_adjoint,
                                                                         4,
                                                                         count,
                                                                         budget,
                                                                         actual_fixed);
                require(is_close(actual, expected, 3e-13),
                        "rectangular RI pullback changed a packed-pair multiplicity or stride");
                require(is_close(actual_fixed, fixed, 3e-13),
                        "rectangular RI jet omitted a factor derivative or changed accumulation");
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved);
    omp_set_dynamic(saved_dynamic);
#endif
}

/** Independent physical oracle: differentiate L*T(C) and its Gram matrix
 * directly from the dense AO-pair map, one orbital coordinate at a time.
 */
Eigen::MatrixXd dense_pair_map_pullback(const xmvb::integrals::RiAoFactorization& factorization,
                                        const Eigen::MatrixXd& coefficients,
                                        const std::vector<double>& gradient) {
    constexpr double step = 1e-6;
    Eigen::MatrixXd answer(coefficients.rows(), coefficients.cols());
    for (int a = 0; a < coefficients.cols(); ++a)
        for (int mu = 0; mu < coefficients.rows(); ++mu) {
            Eigen::MatrixXd plus = coefficients, minus = coefficients;
            plus(mu, a) += step;
            minus(mu, a) -= step;
            const Eigen::MatrixXd high_factors =
                factorization.metric_whitened_ao_pair_factors * build_pair_map(plus);
            const Eigen::MatrixXd low_factors =
                factorization.metric_whitened_ao_pair_factors * build_pair_map(minus);
            const auto high = pack_gram(high_factors);
            const auto low = pack_gram(low_factors);
            answer(mu, a) = (packed_dot(gradient, high) - packed_dot(gradient, low)) / (2 * step);
        }
    return answer;
}

void check_changing_adjoint_block() {
    constexpr int n_bf = 7, n_active = 4, n_auxiliary = 5;
    xmvb::integrals::RiAoFactorization factorization;
    factorization.n_basis_functions = n_bf;
    factorization.n_auxiliary_functions = n_auxiliary;
    factorization.n_packed_ao_pairs = n_bf * (n_bf + 1) / 2;
    factorization.metric_whitened_ao_pair_factors.resize(n_auxiliary,
                                                         factorization.n_packed_ao_pairs);
    for (Eigen::Index i = 0; i < factorization.metric_whitened_ao_pair_factors.size(); ++i)
        factorization.metric_whitened_ao_pair_factors.data()[i] =
            0.13 * std::sin(0.29 * (i + 1)) + 0.07 * std::cos(0.11 * (i + 3));
    Eigen::MatrixXd coefficients(n_bf, n_active);
    for (Eigen::Index i = 0; i < coefficients.size(); ++i)
        coefficients.data()[i] = 0.23 * std::sin(0.17 * (i + 1)) + 0.19 * std::cos(0.31 * (i + 2));
    const auto accepted = make_result(coefficients, factorization);
    const auto cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, accepted, n_active);
    require(cache.accepted_one_index_factors != nullptr,
            "RI changing-adjoint fixture lacks resident one-index factors");
    auto three_active = cache;
    xmvb::integrals::prepare_ri_orbital_pair_products(three_active);
    require(static_cast<bool>(three_active.orbital_pair_products),
            "RI changing-adjoint fixture lacks three-active image");
    auto streamed = cache;
    streamed.accepted_one_index_factors.reset();
    streamed.auxiliary_tile_size = 3;
    auto three_active_only = three_active;
    three_active_only.accepted_one_index_factors.reset();

    const int pairs = n_active * (n_active + 1) / 2;
    const int packed = pairs * (pairs + 1) / 2;
    std::array<std::vector<double>, 3> gradients;
    for (auto& gradient : gradients)
        gradient.resize(packed);
    for (int i = 0; i < packed; ++i) {
        gradients[0][i] = 0.07 * std::sin(0.23 * (i + 1));
        gradients[1][i] = 0.31 * gradients[0][i] + 0.05 * std::cos(0.37 * (i + 2));
        gradients[2][i] = -0.43 * gradients[1][i] + 0.18 * gradients[0][i] + 0.01 * (i % 4);
    }
    std::array<Eigen::MatrixXd, 3> expected;
    for (int r = 0; r < 3; ++r)
        expected[r] = dense_pair_map_pullback(factorization, coefficients, gradients[r]);
    std::vector<double> coordinate(packed);
    for (int i = 0; i < packed; ++i)
        coordinate[i] = 0.01 * (i % 4);
    const Eigen::MatrixXd coordinate_expected =
        dense_pair_map_pullback(factorization, coefficients, coordinate);
    const std::array<const std::vector<double>*, 3> requests{
        &gradients[0], &gradients[1], &gradients[2]};
    for (const auto* variant :
         std::array<const xmvb::integrals::RiActiveTwoElectronResponseCache*, 4>{
             &cache, &three_active, &streamed, &three_active_only})
        for (const std::size_t budget : {std::size_t{4096}, std::size_t{65536}}) {
            const auto two =
                xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient_batch(
                    std::span<const std::vector<double>* const>(requests.data(), 2),
                    *variant,
                    budget);
            const auto three =
                xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient_batch(
                    std::span<const std::vector<double>* const>(requests.data(), requests.size()),
                    *variant,
                    budget);
            require(two.size() == 2 && three.size() == 3,
                    "RI changing-adjoint block returned the wrong number of gradients");
            for (int r = 0; r < 3; ++r) {
                require(is_close(three[r], expected[r], 2e-8),
                        "RI changing-adjoint block differs from physical dense pair-map gradient");
                if (r < 2)
                    require(is_close(two[r], expected[r], 2e-8),
                            "RI changing-adjoint width-two panel changed a physical gradient");
                require(
                    is_close(three[r],
                             xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                                 gradients[r], *variant),
                             2e-12),
                    "RI changing-adjoint scalar and block consumers disagree");
            }
            require(
                is_close(three[2], -0.43 * three[1] + 0.18 * three[0] + coordinate_expected, 2e-8),
                "RI changing-adjoint mixed cotangent lost linearity");
        }
    require(xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient_batch({}, cache)
                .empty(),
            "empty RI changing-adjoint block did not stay empty");
    bool limited = false;
    try {
        xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient_batch(
            std::span<const std::vector<double>* const>(requests.data(), 2), streamed, 64);
    } catch (const std::length_error&) {
        limited = true;
    }
    require(limited, "RI changing-adjoint block did not honor a subminimal workspace");
}

} // namespace

int main() {
    check_rectangular_pullback();
    check_direct_factor_direction();
    check_changing_adjoint_block();
    constexpr int n_bf = 3;
    constexpr int n_active = 2;
    constexpr int n_auxiliary = 4;
    constexpr int n_ao_pairs = n_bf * (n_bf + 1) / 2;

    xmvb::integrals::RiAoFactorization factorization;
    factorization.n_basis_functions = n_bf;
    factorization.n_auxiliary_functions = n_auxiliary;
    factorization.n_packed_ao_pairs = n_ao_pairs;
    factorization.metric_whitened_ao_pair_factors.resize(n_auxiliary, n_ao_pairs);
    factorization.metric_whitened_ao_pair_factors << 0.7, -0.2, 0.4, 0.1, -0.3, 0.8, -0.1, 0.9, 0.2,
        -0.5, 0.6, 0.3, 0.4, 0.1, -0.7, 0.8, 0.2, -0.6, -0.3, 0.5, 0.6, -0.2, 0.9, 0.1;

    Eigen::MatrixXd coefficients(n_bf, n_active);
    coefficients << 0.8, -0.3, 0.2, 0.9, -0.5, 0.4;
    Eigen::MatrixXd direction(n_bf, n_active);
    direction << -0.2, 0.5, 0.7, -0.1, 0.3, 0.6;

    xmvb::integrals::PackedOrbitalPairMapMatrix common_pair_map;
    xmvb::integrals::build_packed_orbital_pair_map(coefficients, &common_pair_map);
    require((common_pair_map - build_pair_map(coefficients)).norm() <= 1.0e-14,
            "common packed orbital pair map changed the canonical convention");

    const auto accepted_result = make_result(coefficients, factorization);
    const auto cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, accepted_result, n_active);
    const Eigen::MatrixXd factor_direction =
        xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(cache, direction);
    const Eigen::MatrixXd second_direction = 0.43 * direction - 0.08 * coefficients;
    const auto factor_direction_batch =
        xmvb::integrals::compute_ri_active_pair_factor_directional_derivative_batch(
            cache, {direction, second_direction});
    require(factor_direction_batch.size() == 2 &&
                is_close(factor_direction_batch[0], factor_direction, 1.0e-13) &&
                is_close(factor_direction_batch[1],
                         xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(
                             cache, second_direction),
                         1.0e-13),
            "packed RI active-pair factor block disagrees with scalar actions");

    constexpr double step = 1.0e-6;
    const auto plus_result = make_result(coefficients + step * direction, factorization);
    const auto minus_result = make_result(coefficients - step * direction, factorization);
    const Eigen::MatrixXd finite_difference_factor_direction =
        (plus_result.ri_active_pair_factors - minus_result.ri_active_pair_factors) / (2.0 * step);
    require(is_close(factor_direction, finite_difference_factor_direction, 1.0e-9),
            "RI active-pair factor direction failed finite difference");

    const std::vector<double> packed_integral_direction =
        xmvb::integrals::compute_ri_packed_active_two_electron_integral_directional_derivative(
            cache, factor_direction);
    const Eigen::MatrixXd packed_integral_direction_batch = xmvb::integrals::
        compute_ri_packed_active_two_electron_integral_directional_derivative_batch(
            cache, factor_direction_batch);
    const std::vector<double> second_packed_integral_direction =
        xmvb::integrals::compute_ri_packed_active_two_electron_integral_directional_derivative(
            cache, factor_direction_batch[1]);
    require(packed_integral_direction_batch.rows() ==
                    static_cast<Eigen::Index>(packed_integral_direction.size()) &&
                packed_integral_direction_batch.cols() == 2,
            "RI packed active-2e direction block has the wrong shape");
    for (Eigen::Index index = 0; index < packed_integral_direction_batch.rows(); ++index) {
        require(std::abs(packed_integral_direction_batch(index, 0) -
                         packed_integral_direction[static_cast<std::size_t>(index)]) <= 1.0e-13 &&
                    std::abs(packed_integral_direction_batch(index, 1) -
                             second_packed_integral_direction[static_cast<std::size_t>(index)]) <=
                        1.0e-13,
                "RI packed active-2e direction block disagrees with scalar actions");
    }
    const std::vector<double> plus_integrals = pack_gram(plus_result.ri_active_pair_factors);
    const std::vector<double> minus_integrals = pack_gram(minus_result.ri_active_pair_factors);
    std::vector<double> finite_difference_integral_direction(plus_integrals.size(), 0.0);
    for (std::size_t index = 0; index < plus_integrals.size(); ++index) {
        finite_difference_integral_direction[index] =
            (plus_integrals[index] - minus_integrals[index]) / (2.0 * step);
        require(std::abs(packed_integral_direction[index] -
                         finite_difference_integral_direction[index]) <= 1.0e-8,
                "RI packed active-2e direction failed finite difference");
    }

    std::vector<double> packed_adjoint(packed_integral_direction.size(), 0.0);
    for (std::size_t index = 0; index < packed_adjoint.size(); ++index) {
        packed_adjoint[index] =
            0.15 + 0.11 * static_cast<double>(index) - 0.03 * static_cast<double>(index * index);
    }
    const Eigen::MatrixXd pullback =
        xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(packed_adjoint,
                                                                              cache);
    auto prepared = cache;
    xmvb::integrals::prepare_ri_orbital_pair_products(prepared);
    require(static_cast<bool>(prepared.orbital_pair_products),
            "small RI orbital-pair image was not prepared");
    const double* products = prepared.orbital_pair_products->data();
    xmvb::integrals::prepare_ri_orbital_pair_products(prepared);
    require(products == prepared.orbital_pair_products->data(),
            "accepted RI orbital-pair image was needlessly rebuilt");
    {
        const auto bytes = xmvb::workspace::retained();
        const auto shared = prepared;
        require(shared.orbital_pair_products == prepared.orbital_pair_products &&
                    shared.accepted_one_index_factors == prepared.accepted_one_index_factors &&
                    bytes == xmvb::workspace::retained(),
                "copying a response cache duplicated or double-charged accepted data");
    }
    require(is_close(xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                         packed_adjoint, prepared),
                     pullback,
                     1e-13),
            "shared RI orbital-pair pullback differs from explicit factor contraction");
    const double directional_duality = packed_dot(packed_adjoint, packed_integral_direction);
    require(std::abs(directional_duality - (pullback.array() * direction.array()).sum()) <= 1.0e-10,
            "RI packed-adjoint pullback violates directional duality");

    // Isolate every packed entry so diagonal entries exercise the doubled
    // factor adjoint while off-diagonal entries exercise both symmetric sides.
    for (std::size_t packed_index = 0; packed_index < packed_adjoint.size(); ++packed_index) {
        std::vector<double> coordinate_adjoint(packed_adjoint.size(), 0.0);
        coordinate_adjoint[packed_index] = 1.0;
        const Eigen::MatrixXd coordinate_pullback =
            xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                coordinate_adjoint, cache);
        require(std::abs(packed_integral_direction[packed_index] -
                         (coordinate_pullback.array() * direction.array()).sum()) <= 1.0e-10,
                "RI packed coordinate pullback has a symmetry-factor error");
        require(is_close(xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                             coordinate_adjoint, prepared),
                         coordinate_pullback,
                         1e-13),
                "shared orbital-pair pullback changed a packed coordinate multiplicity");
    }

    const auto plus_cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, plus_result, n_active);
    const auto minus_cache = xmvb::integrals::build_ri_active_two_electron_response_cache(
        factorization, minus_result, n_active);
    const Eigen::MatrixXd finite_difference_pullback_direction =
        (xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(packed_adjoint,
                                                                               plus_cache) -
         xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(packed_adjoint,
                                                                               minus_cache)) /
        (2.0 * step);
    const Eigen::MatrixXd fixed_adjoint_direction =
        xmvb::integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
            packed_adjoint, cache, direction, &factor_direction);
    require(is_close(fixed_adjoint_direction, finite_difference_pullback_direction, 1.0e-8),
            "RI fixed-adjoint orbital HVP failed finite difference");

    const Eigen::MatrixXd internally_built_factor_direction =
        xmvb::integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
            packed_adjoint, cache, direction);
    require(is_close(internally_built_factor_direction, fixed_adjoint_direction, 1.0e-13),
            "RI fixed-adjoint HVP changed when reusing factor direction");
    // Force streaming and a non-divisible tile boundary. This is the same
    // factor-native operator, not a separate numerical/reference path.
    auto tiled_cache = cache;
    tiled_cache.accepted_one_index_factors.reset();
    tiled_cache.auxiliary_tile_size = 3;
    auto prepared_tiled = tiled_cache;
    xmvb::integrals::prepare_ri_orbital_pair_products(prepared_tiled);
    require(is_close(xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                         packed_adjoint, prepared_tiled),
                     pullback,
                     1e-13),
            "RI orbital-pair image changed when built from auxiliary tiles");
    require(is_close(xmvb::integrals::compute_ri_active_pair_factor_directional_derivative(
                         tiled_cache, direction),
                     factor_direction,
                     1.0e-13),
            "streamed RI factor direction changed at an auxiliary tile boundary");
    const auto tiled_batch =
        xmvb::integrals::compute_ri_active_pair_factor_directional_derivative_batch(
            tiled_cache, {direction, second_direction});
    require(is_close(tiled_batch[0], factor_direction_batch[0], 1.0e-13) &&
                is_close(tiled_batch[1], factor_direction_batch[1], 1.0e-13),
            "streamed RI direction block changed the operator");
    require(is_close(xmvb::integrals::backpropagate_ri_packed_active_two_electron_gradient(
                         packed_adjoint, tiled_cache),
                     pullback,
                     1.0e-13),
            "streamed RI gradient changed the operator");
    require(is_close(xmvb::integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
                         packed_adjoint, tiled_cache, direction, &factor_direction),
                     fixed_adjoint_direction,
                     1.0e-13),
            "streamed RI HVP changed the operator");
    // Also tile directional factors when accepted factors are retained.
    auto cached_tile = cache;
    cached_tile.auxiliary_tile_size = 1;
    require(is_close(xmvb::integrals::apply_ri_packed_active_two_electron_adjoint_hessian_vector(
                         packed_adjoint, cached_tile, direction, &factor_direction),
                     fixed_adjoint_direction,
                     1.0e-13),
            "retained accepted RI factors changed the tiled HVP");
    return 0;
}
