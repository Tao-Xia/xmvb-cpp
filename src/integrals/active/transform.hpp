#pragma once

#include <Eigen/Core>

namespace xmvb::integrals {

/**
 * @brief Transform a packed two-pair kernel through a square orbital matrix.
 *
 * Pair indices are p(i,j)=i(i+1)/2+j for i>=j. The input need only be
 * symmetric within each orbital pair; it need not be symmetric under pair
 * exchange. If T(Q) maps old pairs to new pairs, the result is T(Q)^T K T(Q),
 * where T(Q)[p(a,b),p(i,j)] = Q(a,i)Q(b,j) +
 * (a!=b ? Q(b,i)Q(a,j) : 0).
 *
 * Four successive orbital-index products cost O(n^5) time and O(n^4) space.
 * @throws std::invalid_argument for non-square Q or an incompatible kernel.
 */
Eigen::MatrixXd transform_pair_kernel(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                      const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform);

/** @brief Directional derivative of transform_pair_kernel in both inputs. */
Eigen::MatrixXd
transform_pair_kernel_direction(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                const Eigen::Ref<const Eigen::MatrixXd>& kernel_direction,
                                const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform,
                                const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction);

/**
 * @brief Frobenius adjoint of transform_pair_kernel with respect to its kernel.
 *
 * Returns T(Q) G T(Q)^T. Off-diagonal packed-pair multiplicities are handled
 * explicitly; replacing Q by Q^T in transform_pair_kernel is not equivalent.
 */
Eigen::MatrixXd pullback_pair_kernel(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                                     const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform);

/** @brief Directional derivative of pullback_pair_kernel in both inputs. */
Eigen::MatrixXd
pullback_pair_kernel_direction(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                               const Eigen::Ref<const Eigen::MatrixXd>& gradient_direction,
                               const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform,
                               const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction);

/**
 * @brief Apply the orbital generator to a packed kernel at Q=I.
 *
 * Returns d/dt transform_pair_kernel(K,I+tD) at t=0. The input may lack
 * pair-exchange symmetry. Costs O(n^5) time and O(n^4) output storage,
 * with no four-index tensor workspace.
 */
Eigen::MatrixXd apply_pair_generator(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                     const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction);

/**
 * @brief Frobenius adjoint of the pair generator with respect to its kernel.
 *
 * If L(D) is the directional derivative of the symmetric-pair map T(Q) at
 * Q=I, returns L(D) G + G L(D)^T. G need not have pair-exchange symmetry.
 * Costs O(n^5) time and O(n^4) output storage, without a four-index tensor
 * workspace or an identity-matrix product.
 */
Eigen::MatrixXd pullback_pair_generator(const Eigen::Ref<const Eigen::MatrixXd>& gradient,
                                        const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction);

/**
 * @brief Gradient in Q of <G, transform_pair_kernel(K,Q)> at Q=I.
 *
 * Returns a dense old-orbital-by-new-orbital matrix. K and G may each lack
 * pair-exchange symmetry. Two shared n-by-n^3 workspaces feed two matrix
 * products, for O(n^5) time and O(n^4) auxiliary space.
 */
Eigen::MatrixXd orbital_generator_adjoint(const Eigen::Ref<const Eigen::MatrixXd>& kernel,
                                          const Eigen::Ref<const Eigen::MatrixXd>& gradient);

} // namespace xmvb::integrals
