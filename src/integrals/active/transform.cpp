#include "integrals/active/transform.hpp"

#include <stdexcept>

namespace xmvb::integrals {
namespace {

Eigen::Index pair_index(Eigen::Index first, Eigen::Index second) {
    if (first < second) {
        const Eigen::Index tmp = first;
        first = second;
        second = tmp;
    }
    return first * (first + 1) / 2 + second;
}

Eigen::Index orbital_count(const Eigen::Ref<const Eigen::MatrixXd>& kernel) {
    if (kernel.rows() != kernel.cols()) {
        throw std::invalid_argument("packed pair kernel must be square");
    }
    const Eigen::Index pairs = kernel.rows();
    Eigen::Index n = 0;
    while (n * (n + 1) / 2 < pairs) {
        ++n;
    }
    if (n <= 0 || n * (n + 1) / 2 != pairs) {
        throw std::invalid_argument("packed pair kernel requires a triangular pair count");
    }
    return n;
}

void validate(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
              const Eigen::Ref<const Eigen::MatrixXd>& transform) {
    const Eigen::Index n = transform.rows();
    const Eigen::Index pairs = n * (n + 1) / 2;
    if (n <= 0 || transform.cols() != n || matrix.rows() != pairs || matrix.cols() != pairs) {
        throw std::invalid_argument("packed pair transform dimensions are inconsistent");
    }
}

Eigen::MatrixXd
expand(const Eigen::Ref<const Eigen::MatrixXd>& packed, Eigen::Index n, bool pullback) {
    Eigen::MatrixXd full(n, n * n * n);
    for (Eigen::Index d = 0; d < n; ++d) {
        for (Eigen::Index c = 0; c < n; ++c) {
            const Eigen::Index cd = pair_index(c, d);
            const double cd_weight = pullback && c != d ? 0.5 : 1.0;
            for (Eigen::Index b = 0; b < n; ++b) {
                for (Eigen::Index a = 0; a < n; ++a) {
                    const double ab_weight = pullback && a != b ? 0.5 : 1.0;
                    full(a, b + n * (c + n * d)) =
                        ab_weight * cd_weight * packed(pair_index(a, b), cd);
                }
            }
        }
    }
    return full;
}

// The first tensor index is contiguous. After each product, rotate that index
// to the last position so the next untransformed index becomes contiguous.
void transform_four_indices(Eigen::MatrixXd* full,
                            Eigen::MatrixXd* full_direction,
                            const Eigen::Ref<const Eigen::MatrixXd>& q,
                            const Eigen::Ref<const Eigen::MatrixXd>* dq,
                            bool pullback) {
    const Eigen::Index n = q.rows();
    Eigen::MatrixXd product(n, n * n * n);
    Eigen::MatrixXd product_direction;
    if (full_direction != nullptr) {
        product_direction.resize(n, n * n * n);
    }
    for (int stage = 0; stage < 4; ++stage) {
        if (pullback) {
            product.noalias() = q * (*full);
            if (full_direction != nullptr) {
                product_direction.noalias() = (*dq) * (*full);
                product_direction.noalias() += q * (*full_direction);
            }
        } else {
            product.noalias() = q.transpose() * (*full);
            if (full_direction != nullptr) {
                product_direction.noalias() = dq->transpose() * (*full);
                product_direction.noalias() += q.transpose() * (*full_direction);
            }
        }
        for (Eigen::Index x3 = 0; x3 < n; ++x3) {
            for (Eigen::Index x2 = 0; x2 < n; ++x2) {
                for (Eigen::Index x1 = 0; x1 < n; ++x1) {
                    for (Eigen::Index x0 = 0; x0 < n; ++x0) {
                        const Eigen::Index before = x0 + n * (x1 + n * (x2 + n * x3));
                        const Eigen::Index after = x1 + n * (x2 + n * (x3 + n * x0));
                        full->data()[after] = product.data()[before];
                        if (full_direction != nullptr) {
                            full_direction->data()[after] = product_direction.data()[before];
                        }
                    }
                }
            }
        }
    }
}

Eigen::MatrixXd repack(const Eigen::MatrixXd& full, Eigen::Index n, bool pullback) {
    const Eigen::Index pairs = n * (n + 1) / 2;
    Eigen::MatrixXd packed(pairs, pairs);
    for (Eigen::Index d = 0; d < n; ++d) {
        for (Eigen::Index c = d; c < n; ++c) {
            const double cd_weight = pullback && c != d ? 2.0 : 1.0;
            for (Eigen::Index b = 0; b < n; ++b) {
                for (Eigen::Index a = b; a < n; ++a) {
                    const double ab_weight = pullback && a != b ? 2.0 : 1.0;
                    packed(pair_index(a, b), pair_index(c, d)) =
                        ab_weight * cd_weight * full(a, b + n * (c + n * d));
                }
            }
        }
    }
    return packed;
}

} // namespace

Eigen::MatrixXd transform_pair_kernel(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                      const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform) {
    validate(kernel, orbital_transform);
    Eigen::MatrixXd full = expand(kernel, orbital_transform.rows(), false);
    transform_four_indices(&full, nullptr, orbital_transform, nullptr, false);
    return repack(full, orbital_transform.rows(), false);
}

Eigen::MatrixXd
transform_pair_kernel_direction(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                const Eigen::Ref<const Eigen::MatrixXd>& kernel_direction,
                                const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform,
                                const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction) {
    validate(kernel, orbital_transform);
    validate(kernel_direction, orbital_transform);
    if (orbital_direction.rows() != orbital_transform.rows() ||
        orbital_direction.cols() != orbital_transform.cols()) {
        throw std::invalid_argument("packed pair transform direction has inconsistent dimensions");
    }
    Eigen::MatrixXd full = expand(kernel, orbital_transform.rows(), false);
    Eigen::MatrixXd tangent = expand(kernel_direction, orbital_transform.rows(), false);
    transform_four_indices(&full, &tangent, orbital_transform, &orbital_direction, false);
    return repack(tangent, orbital_transform.rows(), false);
}

Eigen::MatrixXd pullback_pair_kernel(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                                     const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform) {
    validate(gradient, orbital_transform);
    Eigen::MatrixXd full = expand(gradient, orbital_transform.rows(), true);
    transform_four_indices(&full, nullptr, orbital_transform, nullptr, true);
    return repack(full, orbital_transform.rows(), true);
}

Eigen::MatrixXd
pullback_pair_kernel_direction(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                               const Eigen::Ref<const Eigen::MatrixXd>& gradient_direction,
                               const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform,
                               const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction) {
    validate(gradient, orbital_transform);
    validate(gradient_direction, orbital_transform);
    if (orbital_direction.rows() != orbital_transform.rows() ||
        orbital_direction.cols() != orbital_transform.cols()) {
        throw std::invalid_argument("packed pair pullback direction has inconsistent dimensions");
    }
    Eigen::MatrixXd full = expand(gradient, orbital_transform.rows(), true);
    Eigen::MatrixXd tangent = expand(gradient_direction, orbital_transform.rows(), true);
    transform_four_indices(&full, &tangent, orbital_transform, &orbital_direction, true);
    return repack(tangent, orbital_transform.rows(), true);
}

Eigen::MatrixXd apply_pair_generator(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                     const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction) {
    const Eigen::Index n = orbital_count(kernel);
    if (orbital_direction.rows() != n || orbital_direction.cols() != n) {
        throw std::invalid_argument("pair generator direction has inconsistent dimensions");
    }
    const Eigen::Index pairs = kernel.rows();
    Eigen::MatrixXd result(pairs, pairs);
    for (Eigen::Index j = 0; j < n; ++j) {
        for (Eigen::Index i = j; i < n; ++i) {
            const Eigen::Index ij = pair_index(i, j);
            for (Eigen::Index l = 0; l < n; ++l) {
                for (Eigen::Index k = l; k < n; ++k) {
                    const Eigen::Index kl = pair_index(k, l);
                    double value = 0.0;
                    for (Eigen::Index a = 0; a < n; ++a) {
                        value += orbital_direction(a, i) * kernel(pair_index(a, j), kl) +
                                 orbital_direction(a, j) * kernel(pair_index(i, a), kl) +
                                 orbital_direction(a, k) * kernel(ij, pair_index(a, l)) +
                                 orbital_direction(a, l) * kernel(ij, pair_index(k, a));
                    }
                    result(ij, kl) = value;
                }
            }
        }
    }
    return result;
}

Eigen::MatrixXd
pullback_pair_generator(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                        const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction) {
    const Eigen::Index n = orbital_count(gradient);
    if (orbital_direction.rows() != n || orbital_direction.cols() != n) {
        throw std::invalid_argument("pair generator direction has inconsistent dimensions");
    }
    const Eigen::Index pairs = gradient.rows();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(pairs, pairs);
    // For a source pair (i,j), L(D) replaces either orbital with a. Scatter
    // each nonzero L(output,source) into both sides of L G + G L^T. The two
    // replacements intentionally coincide for diagonal source pairs.
    const auto scatter =
        [&result, &gradient, pairs](Eigen::Index output, Eigen::Index source, double value) {
            if (value == 0.0)
                return;
            for (Eigen::Index other = 0; other < pairs; ++other) {
                result(output, other) += value * gradient(source, other);
                result(other, output) += value * gradient(other, source);
            }
        };
    for (Eigen::Index j = 0; j < n; ++j) {
        for (Eigen::Index i = j; i < n; ++i) {
            const Eigen::Index source = pair_index(i, j);
            for (Eigen::Index a = 0; a < n; ++a) {
                scatter(pair_index(a, j), source, orbital_direction(a, i));
                scatter(pair_index(i, a), source, orbital_direction(a, j));
            }
        }
    }
    return result;
}

Eigen::MatrixXd orbital_generator_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                          const Eigen::Ref<const Eigen::MatrixXd>& gradient) {
    const Eigen::Index n = orbital_count(kernel);
    if (gradient.rows() != kernel.rows() || gradient.cols() != kernel.cols()) {
        throw std::invalid_argument("orbital generator adjoint dimensions are inconsistent");
    }
    // For a packed pair (i,j), summing both ordered orientations gives weight
    // 1/(2-delta_ij) to each. The two product terms then contribute a factor 2.
    // Columns enumerate (j,k,l), with k,l also expanded in both orientations.
    Eigen::MatrixXd expanded_kernel(n, n * n * n);
    Eigen::MatrixXd expanded_gradient(n, n * n * n);
    for (Eigen::Index l = 0; l < n; ++l) {
        for (Eigen::Index k = 0; k < n; ++k) {
            const Eigen::Index kl = pair_index(k, l);
            const double kl_weight = k == l ? 1.0 : 0.5;
            for (Eigen::Index j = 0; j < n; ++j) {
                const Eigen::Index column = j + n * (k + n * l);
                for (Eigen::Index a = 0; a < n; ++a) {
                    const Eigen::Index aj = pair_index(a, j);
                    const double pair_weight = (a == j ? 1.0 : 0.5) * kl_weight;
                    expanded_kernel(a, column) = kernel(aj, kl);
                    expanded_gradient(a, column) = pair_weight * gradient(aj, kl);
                }
            }
        }
    }
    Eigen::MatrixXd result(n, n);
    result.noalias() = expanded_kernel * expanded_gradient.transpose();

    // Exchange the packed-pair roles for the other two orbital indices. Neither
    // kernel nor gradient is assumed symmetric under that exchange.
    for (Eigen::Index l = 0; l < n; ++l) {
        for (Eigen::Index k = 0; k < n; ++k) {
            const Eigen::Index kl = pair_index(k, l);
            const double kl_weight = k == l ? 1.0 : 0.5;
            for (Eigen::Index j = 0; j < n; ++j) {
                const Eigen::Index column = j + n * (k + n * l);
                for (Eigen::Index a = 0; a < n; ++a) {
                    const Eigen::Index aj = pair_index(a, j);
                    const double pair_weight = (a == j ? 1.0 : 0.5) * kl_weight;
                    expanded_kernel(a, column) = kernel(kl, aj);
                    expanded_gradient(a, column) = pair_weight * gradient(kl, aj);
                }
            }
        }
    }
    result.noalias() += expanded_kernel * expanded_gradient.transpose();
    result *= 2.0;
    return result;
}

} // namespace xmvb::integrals
