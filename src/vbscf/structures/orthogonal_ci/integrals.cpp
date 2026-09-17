#include "vbscf/structures/orthogonal_ci/integrals.hpp"

#include <stdexcept>

#include <Eigen/Cholesky>

#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/construction/kernel.hpp"

namespace xmvb::vb {
namespace {

Eigen::MatrixXd pair_transform(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform) {
  const int n_orbitals = static_cast<int>(orbital_transform.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  for (int new_first = 0; new_first < n_orbitals; ++new_first) {
    for (int new_second = 0; new_second <= new_first; ++new_second) {
      const int new_pair = TwoElectronIndexer::packed_pair_index(
          new_first, new_second);
      for (int old_first = 0; old_first < n_orbitals; ++old_first) {
        for (int old_second = 0; old_second <= old_first; ++old_second) {
          const int old_pair = TwoElectronIndexer::packed_pair_index(
              old_first, old_second);
          double coefficient =
              orbital_transform(old_first, new_first) *
              orbital_transform(old_second, new_second);
          if (old_first != old_second) {
            coefficient +=
                orbital_transform(old_second, new_first) *
                orbital_transform(old_first, new_second);
          }
          result(old_pair, new_pair) = coefficient;
        }
      }
    }
  }
  return result;
}

Eigen::MatrixXd pair_transform_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform,
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_direction) {
  const int n_orbitals = static_cast<int>(orbital_transform.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  for (int new_first = 0; new_first < n_orbitals; ++new_first) {
    for (int new_second = 0; new_second <= new_first; ++new_second) {
      const int new_pair = TwoElectronIndexer::packed_pair_index(
          new_first, new_second);
      for (int old_first = 0; old_first < n_orbitals; ++old_first) {
        for (int old_second = 0; old_second <= old_first; ++old_second) {
          const int old_pair = TwoElectronIndexer::packed_pair_index(
              old_first, old_second);
          double coefficient =
              orbital_direction(old_first, new_first) *
                  orbital_transform(old_second, new_second) +
              orbital_transform(old_first, new_first) *
                  orbital_direction(old_second, new_second);
          if (old_first != old_second) {
            coefficient +=
                orbital_direction(old_second, new_first) *
                    orbital_transform(old_first, new_second) +
                orbital_transform(old_second, new_first) *
                    orbital_direction(old_first, new_second);
          }
          result(old_pair, new_pair) = coefficient;
        }
      }
    }
  }
  return result;
}

Eigen::MatrixXd pair_transform_adjoint(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform,
    const Eigen::Ref<const Eigen::MatrixXd>& pair_gradient) {
  const int n_orbitals = static_cast<int>(orbital_transform.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  if (orbital_transform.cols() != n_orbitals ||
      pair_gradient.rows() != n_pairs || pair_gradient.cols() != n_pairs) {
    throw std::invalid_argument(
        "pair-transform adjoint has incompatible dimensions");
  }
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  for (int new_first = 0; new_first < n_orbitals; ++new_first) {
    for (int new_second = 0; new_second <= new_first; ++new_second) {
      const int new_pair = TwoElectronIndexer::packed_pair_index(
          new_first, new_second);
      for (int old_first = 0; old_first < n_orbitals; ++old_first) {
        for (int old_second = 0; old_second <= old_first; ++old_second) {
          const int old_pair = TwoElectronIndexer::packed_pair_index(
              old_first, old_second);
          const double weight = pair_gradient(old_pair, new_pair);
          result(old_first, new_first) +=
              weight * orbital_transform(old_second, new_second);
          result(old_second, new_second) +=
              weight * orbital_transform(old_first, new_first);
          if (old_first != old_second) {
            result(old_second, new_first) +=
                weight * orbital_transform(old_first, new_second);
            result(old_first, new_second) +=
                weight * orbital_transform(old_second, new_first);
          }
        }
      }
    }
  }
  return result;
}

Eigen::MatrixXd unpack_two_electron_kernel(
    const std::vector<double>& packed,
    int n_active_orbitals) {
  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  if (packed.size() !=
      packed_active_two_electron_integral_count(n_active_orbitals)) {
    throw std::invalid_argument(
        "packed active two-electron direction has incompatible dimensions");
  }
  Eigen::MatrixXd result(n_pairs, n_pairs);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < n_pairs; ++row) {
      result(row, column) = packed[
          TwoElectronIndexer::packed_pair_of_pairs_index(row, column)];
    }
  }
  return result;
}

void pack_two_electron_kernel(
    const Eigen::Ref<const Eigen::MatrixXd>& kernel,
    int n_active_orbitals,
    ActiveSpaceTwoElectronResult* result) {
  result->representation = ActiveSpaceTwoElectronRepresentation::PackedExact;
  result->packed_active_two_electron_integrals.resize(
      packed_active_two_electron_integral_count(n_active_orbitals));
  for (int column = 0; column < kernel.cols(); ++column) {
    for (int row = 0; row <= column; ++row) {
      result->packed_active_two_electron_integrals[
          TwoElectronIndexer::packed_pair_of_pairs_index(row, column)] =
          kernel(row, column);
    }
  }
}

}  // namespace

OrthogonalActiveIntegrals orthogonalize_active_integrals(
    const std::vector<double>& active_overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron,
    const ActiveSpaceTwoElectronResult& two_electron,
    int n_active_orbitals) {
  if (n_active_orbitals <= 0 ||
      active_overlap.size() !=
          static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals ||
      one_electron.rows() != n_active_orbitals ||
      one_electron.cols() != n_active_orbitals ||
      !one_electron.allFinite()) {
    throw std::invalid_argument("invalid active-space orthogonalization input");
  }
  const Eigen::Map<const Eigen::MatrixXd> overlap(
      active_overlap.data(),
      n_active_orbitals,
      n_active_orbitals);
  Eigen::LLT<Eigen::MatrixXd> cholesky(overlap);
  if (cholesky.info() != Eigen::Success) {
    throw std::runtime_error(
        "active orbital overlap is not positive definite");
  }

  OrthogonalActiveIntegrals result;
  result.orbital_transform = cholesky.matrixU();
  const Eigen::MatrixXd inverse_transform =
      result.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(
              n_active_orbitals,
              n_active_orbitals));
  result.one_electron =
      inverse_transform.transpose() * one_electron * inverse_transform;

  const int n_pairs = packed_active_pair_count(n_active_orbitals);
  const Eigen::MatrixXd packed_pair_transform =
      pair_transform(inverse_transform);

  Eigen::MatrixXd original_kernel(n_pairs, n_pairs);
  const ActiveSpaceTwoElectronView view =
      make_active_space_two_electron_view(two_electron);
  for (int column = 0; column < n_pairs; ++column) {
    for (int row = 0; row < n_pairs; ++row) {
      original_kernel(row, column) =
          lookup_active_space_two_electron_kernel_value(
              view, row, column, n_active_orbitals);
    }
  }
  result.pair_kernel =
      packed_pair_transform.transpose() * original_kernel *
      packed_pair_transform;
  pack_two_electron_kernel(
      result.pair_kernel,
      n_active_orbitals,
      &result.two_electron);
  return result;
}

OrthogonalActiveIntegrals orthogonalize_active_integral_direction(
    const OrthogonalActiveIntegrals& accepted,
    const std::vector<double>& overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& one_electron_direction,
    const std::vector<double>& packed_two_electron_direction,
    int n_active_orbitals) {
  if (n_active_orbitals <= 0 ||
      accepted.orbital_transform.rows() != n_active_orbitals ||
      accepted.orbital_transform.cols() != n_active_orbitals ||
      accepted.one_electron.rows() != n_active_orbitals ||
      accepted.one_electron.cols() != n_active_orbitals ||
      overlap_direction.size() !=
          static_cast<std::size_t>(n_active_orbitals) * n_active_orbitals ||
      one_electron_direction.rows() != n_active_orbitals ||
      one_electron_direction.cols() != n_active_orbitals ||
      !one_electron_direction.allFinite()) {
    throw std::invalid_argument(
        "invalid orthogonal active-integral direction");
  }
  const Eigen::Map<const Eigen::MatrixXd> delta_overlap_raw(
      overlap_direction.data(),
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd delta_overlap =
      0.5 * (delta_overlap_raw + delta_overlap_raw.transpose());
  const Eigen::MatrixXd inverse_transform =
      accepted.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(
              n_active_orbitals,
              n_active_orbitals));
  const Eigen::MatrixXd normalized_overlap_direction =
      inverse_transform.transpose() * delta_overlap * inverse_transform;
  Eigen::MatrixXd relative_transform = Eigen::MatrixXd::Zero(
      n_active_orbitals,
      n_active_orbitals);
  for (int column = 0; column < n_active_orbitals; ++column) {
    relative_transform(column, column) =
        0.5 * normalized_overlap_direction(column, column);
    for (int row = 0; row < column; ++row) {
      relative_transform(row, column) =
          normalized_overlap_direction(row, column);
    }
  }

  OrthogonalActiveIntegrals result;
  result.orbital_transform =
      relative_transform * accepted.orbital_transform;
  result.one_electron =
      inverse_transform.transpose() * one_electron_direction *
          inverse_transform -
      relative_transform.transpose() * accepted.one_electron -
      accepted.one_electron * relative_transform;

  const Eigen::MatrixXd packed_pair_transform =
      pair_transform(inverse_transform);
  const Eigen::MatrixXd delta_original_kernel =
      unpack_two_electron_kernel(
          packed_two_electron_direction,
          n_active_orbitals);
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
      n_active_orbitals,
      n_active_orbitals);
  const Eigen::MatrixXd relative_pair_direction =
      pair_transform_direction(identity, -relative_transform);
  result.pair_kernel =
      packed_pair_transform.transpose() * delta_original_kernel *
          packed_pair_transform +
      relative_pair_direction.transpose() * accepted.pair_kernel +
      accepted.pair_kernel * relative_pair_direction;
  result.pair_kernel = 0.5 *
      (result.pair_kernel + result.pair_kernel.transpose());
  pack_two_electron_kernel(
      result.pair_kernel,
      n_active_orbitals,
      &result.two_electron);
  return result;
}

NonorthogonalActiveIntegralAdjoint
backpropagate_orthogonal_active_integral_adjoint(
    const OrthogonalActiveIntegrals& accepted,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_one_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_pair_kernel_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& generator_gradient) {
  const int n_orbitals = static_cast<int>(accepted.one_electron.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  if (n_orbitals <= 0 || accepted.one_electron.cols() != n_orbitals ||
      accepted.orbital_transform.rows() != n_orbitals ||
      accepted.orbital_transform.cols() != n_orbitals ||
      accepted.pair_kernel.rows() != n_pairs ||
      accepted.pair_kernel.cols() != n_pairs ||
      orthogonal_one_electron_gradient.rows() != n_orbitals ||
      orthogonal_one_electron_gradient.cols() != n_orbitals ||
      orthogonal_pair_kernel_gradient.rows() != n_pairs ||
      orthogonal_pair_kernel_gradient.cols() != n_pairs ||
      generator_gradient.rows() != n_orbitals ||
      generator_gradient.cols() != n_orbitals) {
    throw std::invalid_argument(
        "orthogonal active-integral adjoint has incompatible dimensions");
  }

  const Eigen::MatrixXd inverse_transform =
      accepted.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(n_orbitals, n_orbitals));
  const Eigen::MatrixXd packed_pair_transform =
      pair_transform(inverse_transform);

  NonorthogonalActiveIntegralAdjoint result;
  result.one_electron =
      inverse_transform * orthogonal_one_electron_gradient *
      inverse_transform.transpose();
  result.one_electron =
      0.5 * (result.one_electron + result.one_electron.transpose());
  result.pair_kernel =
      packed_pair_transform * orthogonal_pair_kernel_gradient *
      packed_pair_transform.transpose();
  result.pair_kernel =
      0.5 * (result.pair_kernel + result.pair_kernel.transpose());

  Eigen::MatrixXd relative_gradient = generator_gradient;
  relative_gradient.noalias() -=
      accepted.one_electron * orthogonal_one_electron_gradient.transpose() +
      accepted.one_electron.transpose() * orthogonal_one_electron_gradient;
  const Eigen::MatrixXd pair_transform_gradient =
      accepted.pair_kernel * orthogonal_pair_kernel_gradient.transpose() +
      accepted.pair_kernel.transpose() * orthogonal_pair_kernel_gradient;
  relative_gradient.noalias() -= pair_transform_adjoint(
      Eigen::MatrixXd::Identity(n_orbitals, n_orbitals),
      pair_transform_gradient);

  Eigen::MatrixXd normalized_overlap_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    normalized_overlap_gradient(column, column) =
        0.5 * relative_gradient(column, column);
    for (int row = 0; row < column; ++row) {
      const double value = 0.5 * relative_gradient(row, column);
      normalized_overlap_gradient(row, column) = value;
      normalized_overlap_gradient(column, row) = value;
    }
  }
  result.overlap =
      inverse_transform * normalized_overlap_gradient *
      inverse_transform.transpose();
  result.overlap = 0.5 * (result.overlap + result.overlap.transpose());
  return result;
}

NonorthogonalActiveIntegralAdjoint
backpropagate_orthogonal_active_integral_adjoint_direction(
    const OrthogonalActiveIntegrals& accepted,
    const OrthogonalActiveIntegrals& direction,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_one_electron_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& orthogonal_pair_kernel_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>& generator_gradient,
    const Eigen::Ref<const Eigen::MatrixXd>&
        orthogonal_one_electron_gradient_direction,
    const Eigen::Ref<const Eigen::MatrixXd>&
        orthogonal_pair_kernel_gradient_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& generator_gradient_direction) {
  const int n_orbitals = static_cast<int>(accepted.one_electron.rows());
  const int n_pairs = packed_active_pair_count(n_orbitals);
  if (n_orbitals <= 0 || direction.one_electron.rows() != n_orbitals ||
      direction.one_electron.cols() != n_orbitals ||
      direction.pair_kernel.rows() != n_pairs ||
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
      accepted.orbital_transform
          .template triangularView<Eigen::Upper>()
          .solve(Eigen::MatrixXd::Identity(n_orbitals, n_orbitals));
  const Eigen::MatrixXd relative_direction =
      direction.orbital_transform * inverse_transform;
  const Eigen::MatrixXd inverse_transform_direction =
      -inverse_transform * relative_direction;
  const Eigen::MatrixXd packed_pair_transform =
      pair_transform(inverse_transform);
  const Eigen::MatrixXd packed_pair_transform_direction =
      pair_transform_direction(
          inverse_transform,
          inverse_transform_direction);

  NonorthogonalActiveIntegralAdjoint result;
  result.one_electron =
      inverse_transform_direction * orthogonal_one_electron_gradient *
          inverse_transform.transpose() +
      inverse_transform * orthogonal_one_electron_gradient_direction *
          inverse_transform.transpose() +
      inverse_transform * orthogonal_one_electron_gradient *
          inverse_transform_direction.transpose();
  result.one_electron =
      0.5 * (result.one_electron + result.one_electron.transpose());
  result.pair_kernel =
      packed_pair_transform_direction * orthogonal_pair_kernel_gradient *
          packed_pair_transform.transpose() +
      packed_pair_transform * orthogonal_pair_kernel_gradient_direction *
          packed_pair_transform.transpose() +
      packed_pair_transform * orthogonal_pair_kernel_gradient *
          packed_pair_transform_direction.transpose();
  result.pair_kernel =
      0.5 * (result.pair_kernel + result.pair_kernel.transpose());

  const Eigen::MatrixXd pair_transform_gradient =
      accepted.pair_kernel * orthogonal_pair_kernel_gradient.transpose() +
      accepted.pair_kernel.transpose() * orthogonal_pair_kernel_gradient;
  const Eigen::MatrixXd pair_transform_gradient_direction =
      direction.pair_kernel * orthogonal_pair_kernel_gradient.transpose() +
      accepted.pair_kernel *
          orthogonal_pair_kernel_gradient_direction.transpose() +
      direction.pair_kernel.transpose() * orthogonal_pair_kernel_gradient +
      accepted.pair_kernel.transpose() *
          orthogonal_pair_kernel_gradient_direction;
  Eigen::MatrixXd relative_gradient = generator_gradient;
  relative_gradient.noalias() -=
      accepted.one_electron * orthogonal_one_electron_gradient.transpose() +
      accepted.one_electron.transpose() * orthogonal_one_electron_gradient;
  relative_gradient.noalias() -= pair_transform_adjoint(
      Eigen::MatrixXd::Identity(n_orbitals, n_orbitals),
      pair_transform_gradient);
  Eigen::MatrixXd relative_gradient_direction = generator_gradient_direction;
  relative_gradient_direction.noalias() -=
      direction.one_electron * orthogonal_one_electron_gradient.transpose() +
      accepted.one_electron *
          orthogonal_one_electron_gradient_direction.transpose() +
      direction.one_electron.transpose() * orthogonal_one_electron_gradient +
      accepted.one_electron.transpose() *
          orthogonal_one_electron_gradient_direction;
  relative_gradient_direction.noalias() -= pair_transform_adjoint(
      Eigen::MatrixXd::Identity(n_orbitals, n_orbitals),
      pair_transform_gradient_direction);

  Eigen::MatrixXd normalized_overlap_gradient = Eigen::MatrixXd::Zero(
      n_orbitals, n_orbitals);
  Eigen::MatrixXd normalized_overlap_gradient_direction =
      Eigen::MatrixXd::Zero(n_orbitals, n_orbitals);
  for (int column = 0; column < n_orbitals; ++column) {
    normalized_overlap_gradient(column, column) =
        0.5 * relative_gradient(column, column);
    normalized_overlap_gradient_direction(column, column) =
        0.5 * relative_gradient_direction(column, column);
    for (int row = 0; row < column; ++row) {
      const double value = 0.5 * relative_gradient(row, column);
      normalized_overlap_gradient(row, column) = value;
      normalized_overlap_gradient(column, row) = value;
      const double direction_value =
          0.5 * relative_gradient_direction(row, column);
      normalized_overlap_gradient_direction(row, column) = direction_value;
      normalized_overlap_gradient_direction(column, row) = direction_value;
    }
  }
  result.overlap =
      inverse_transform_direction * normalized_overlap_gradient *
          inverse_transform.transpose() +
      inverse_transform * normalized_overlap_gradient_direction *
          inverse_transform.transpose() +
      inverse_transform * normalized_overlap_gradient *
          inverse_transform_direction.transpose();
  result.overlap = 0.5 * (result.overlap + result.overlap.transpose());
  return result;
}

}  // namespace xmvb::vb
