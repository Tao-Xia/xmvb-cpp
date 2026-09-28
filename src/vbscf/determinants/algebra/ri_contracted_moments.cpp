#include "vbscf/determinants/algebra/ri_contracted_moments.hpp"

#include <stdexcept>

namespace xmvb::vb {
namespace {

Eigen::Map<const Eigen::MatrixXd> table_matrix(
    const RiContractedMoments::Table& table,
    Eigen::Index row,
    int dimension) {
  return Eigen::Map<const Eigen::MatrixXd>(
      table.data() + row * dimension * dimension,
      dimension,
      dimension);
}

Eigen::Map<Eigen::MatrixXd> mutable_table_matrix(
    RiContractedMoments::Table* table,
    Eigen::Index row,
    int dimension) {
  return Eigen::Map<Eigen::MatrixXd>(
      table->data() + row * dimension * dimension,
      dimension,
      dimension);
}

}  // namespace

void RiContractedMoments::initialize(
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_overlap,
    const Eigen::Ref<const Table>& channels) {
  if (inverse_overlap.rows() != inverse_overlap.cols() ||
      channels.cols() != inverse_overlap.size()) {
    throw std::invalid_argument(
        "RI contracted-moment anchor dimensions differ");
  }
  inverse_overlap_ = inverse_overlap;
  channels_ = channels;
  first_moments_.resizeLike(channels_);
  second_moments_.resizeLike(channels_);
  const int n = dimension();
  for (Eigen::Index auxiliary = 0;
       auxiliary < channel_count();
       ++auxiliary) {
    const Eigen::Map<const Eigen::MatrixXd> accepted = channel(auxiliary);
    Eigen::Map<Eigen::MatrixXd> first =
        mutable_table_matrix(&first_moments_, auxiliary, n);
    Eigen::Map<Eigen::MatrixXd> second =
        mutable_table_matrix(&second_moments_, auxiliary, n);
    first.noalias() = accepted * inverse_overlap_;
    second.noalias() = accepted * first;
  }
  rebuild_aggregates();
}

void RiContractedMoments::update(
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_left,
    const Eigen::Ref<const Eigen::MatrixXd>& inverse_right,
    const Eigen::Ref<const Table>& channel_left,
    const Eigen::Ref<const Table>& channel_right) {
  const int n = dimension();
  if (inverse_left.rows() != n || inverse_right.rows() != n ||
      inverse_left.cols() != inverse_right.cols() ||
      channel_left.rows() != channel_count() ||
      channel_right.rows() != channel_count() ||
      channel_left.cols() != channel_right.cols() ||
      (n == 0 ? channel_left.cols() != 0
              : channel_left.cols() % n != 0)) {
    throw std::invalid_argument(
        "RI contracted-moment update dimensions differ");
  }
  const int channel_rank =
      n == 0 ? 0 : static_cast<int>(channel_left.cols() / n);
  const Eigen::MatrixXd old_inverse = inverse_overlap_;
  const Eigen::MatrixXd new_inverse =
      old_inverse + inverse_left * inverse_right.transpose();

  for (Eigen::Index auxiliary = 0;
       auxiliary < channel_count();
       ++auxiliary) {
    Eigen::Map<Eigen::MatrixXd> accepted =
        mutable_table_matrix(&channels_, auxiliary, n);
    Eigen::Map<Eigen::MatrixXd> first =
        mutable_table_matrix(&first_moments_, auxiliary, n);
    Eigen::Map<Eigen::MatrixXd> second =
        mutable_table_matrix(&second_moments_, auxiliary, n);
    const Eigen::MatrixXd old_accepted = accepted;
    const Eigen::MatrixXd old_first = first;
    const Eigen::Map<const Eigen::MatrixXd> left =
        Eigen::Map<const Eigen::MatrixXd>(
            channel_left.data() + auxiliary * n * channel_rank,
            n,
            channel_rank);
    const Eigen::Map<const Eigen::MatrixXd> right =
        Eigen::Map<const Eigen::MatrixXd>(
            channel_right.data() + auxiliary * n * channel_rank,
            n,
            channel_rank);

    const Eigen::MatrixXd new_accepted =
        old_accepted + left * right.transpose();
    Eigen::MatrixXd new_first = old_first;
    if (channel_rank != 0) {
      new_first.noalias() +=
          left * (right.transpose() * old_inverse);
    }
    if (inverse_left.cols() != 0) {
      new_first.noalias() +=
          (new_accepted * inverse_left) * inverse_right.transpose();
    }

    Eigen::MatrixXd new_second = second;
    if (channel_rank != 0) {
      new_second.noalias() +=
          (new_accepted * left) * (right.transpose() * old_inverse);
      new_second.noalias() += left * (right.transpose() * old_first);
    }
    if (inverse_left.cols() != 0) {
      new_second.noalias() +=
          (new_accepted * (new_accepted * inverse_left)) *
          inverse_right.transpose();
    }

    accepted = new_accepted;
    first = new_first;
    second = new_second;
  }
  inverse_overlap_ = new_inverse;
  rebuild_aggregates();
}

Eigen::Map<const Eigen::MatrixXd> RiContractedMoments::channel(
    Eigen::Index auxiliary) const {
  return table_matrix(channels_, auxiliary, dimension());
}

Eigen::Map<const Eigen::MatrixXd> RiContractedMoments::first_moment(
    Eigen::Index auxiliary) const {
  return table_matrix(first_moments_, auxiliary, dimension());
}

Eigen::Map<const Eigen::MatrixXd> RiContractedMoments::second_moment(
    Eigen::Index auxiliary) const {
  return table_matrix(second_moments_, auxiliary, dimension());
}

void RiContractedMoments::rebuild_aggregates() {
  second_coefficient_sum_ = 0.0;
  second_response_moment_ =
      Eigen::MatrixXd::Zero(dimension(), dimension());
  for (Eigen::Index auxiliary = 0;
       auxiliary < channel_count();
       ++auxiliary) {
    const Eigen::Map<const Eigen::MatrixXd> accepted = channel(auxiliary);
    const double trace = accepted.trace();
    second_coefficient_sum_ += 0.5 *
        (trace * trace -
         accepted.cwiseProduct(accepted.transpose()).sum());
    second_response_moment_.noalias() +=
        trace * first_moment(auxiliary) - second_moment(auxiliary);
  }
}

}  // namespace xmvb::vb
