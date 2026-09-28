#include "vbscf/determinants/pairs/woodbury_ri.hpp"

#include <algorithm>
#include <stdexcept>

#include "vbscf/determinants/pairs/woodbury_core.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"

namespace xmvb::vb {
namespace {

std::vector<int> changed_positions(
    const std::vector<int>& old_string,
    const std::vector<int>& new_string) {
  if (old_string.size() != new_string.size()) {
    return {};
  }
  std::vector<int> changed;
  for (int index = 0; index < static_cast<int>(old_string.size()); ++index) {
    if (old_string[index] != new_string[index]) {
      changed.push_back(index);
    }
  }
  return changed;
}

bool factors_cover_strings(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    Eigen::Index factor_columns) {
  for (const int right : occupied_right) {
    for (const int left : occupied_left) {
      if (right < 0 || left < 0 ||
          TwoElectronIndexer::packed_pair_index(right, left) >=
              factor_columns) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

WoodburyRiState::WoodburyRiState() = default;
WoodburyRiState::~WoodburyRiState() = default;

Eigen::Map<const Eigen::MatrixXd> WoodburyRiState::channel(
    Eigen::Index auxiliary) const {
  return moments_.channel(auxiliary);
}

bool WoodburyRiState::initialize(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) {
  core_.reset();
  n_electrons_ = static_cast<int>(occupied_left.size());
  if (static_cast<int>(occupied_right.size()) != n_electrons_ ||
      overlap.rows() != n_electrons_ || overlap.cols() != n_electrons_ ||
      !factors_cover_strings(
          occupied_left, occupied_right, ri_factors.cols())) {
    return false;
  }

  auto core = std::make_unique<WoodburyCore>(overlap);
  RiContractedMoments::Table channels(
      ri_factors.rows(),
      static_cast<Eigen::Index>(n_electrons_) * n_electrons_);
  Eigen::MatrixXd transition(n_electrons_, n_electrons_);
  for (Eigen::Index auxiliary = 0;
       auxiliary < ri_factors.rows();
       ++auxiliary) {
    for (int left = 0; left < n_electrons_; ++left) {
      for (int right = 0; right < n_electrons_; ++right) {
        transition(right, left) = ri_factors(
            auxiliary,
            TwoElectronIndexer::packed_pair_index(
                occupied_right[right], occupied_left[left]));
      }
    }
    Eigen::Map<Eigen::MatrixXd> channel(
        channels.data() + auxiliary * n_electrons_ * n_electrons_,
        n_electrons_,
        n_electrons_);
    channel.noalias() = core->inverse_base() * transition;
  }
  if (!channels.allFinite()) {
    return false;
  }
  moments_.initialize(core->inverse_base(), channels);
  occupied_left_ = occupied_left;
  occupied_right_ = occupied_right;
  core_ = std::move(core);
  return true;
}

void WoodburyRiState::update_base_channels(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  if (left.cols() == 0) {
    return;
  }
  const Eigen::Index n_auxiliary = moments_.channel_count();
  const int rank = left.cols();
  RiContractedMoments::Table channel_left(
      n_auxiliary, n_electrons_ * rank);
  RiContractedMoments::Table channel_right(
      n_auxiliary, n_electrons_ * rank);
  // WoodburyCore returns the physical overlap-update right factor `V`.
  // Its inverse update is `delta K = L (V^T K)`, whereas every channel
  // changes as `delta A_Q = L (V^T A_Q)`.
  const Eigen::MatrixXd inverse_right =
      moments_.inverse_overlap().transpose() * right;
  for (Eigen::Index auxiliary = 0;
       auxiliary < n_auxiliary;
       ++auxiliary) {
    Eigen::Map<Eigen::MatrixXd>(
        channel_left.data() + auxiliary * n_electrons_ * rank,
        n_electrons_,
        rank) = left;
    Eigen::Map<Eigen::MatrixXd>(
        channel_right.data() + auxiliary * n_electrons_ * rank,
        n_electrons_,
        rank).noalias() = channel(auxiliary).transpose() * right;
  }
  moments_.update(left, inverse_right, channel_left, channel_right);
}

bool WoodburyRiState::update_right(
    const std::vector<int>& occupied_right_new,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_new,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) {
  if (!valid() ||
      overlap_new.rows() != n_electrons_ ||
      overlap_new.cols() != n_electrons_ ||
      !factors_cover_strings(
          occupied_left_, occupied_right_new, ri_factors.cols())) {
    return false;
  }
  const std::vector<int> rows =
      changed_positions(occupied_right_, occupied_right_new);
  if (rows.empty()) {
    return occupied_right_new == occupied_right_ &&
        overlap_new.isApprox(core_->overlap());
  }

  Eigen::MatrixXd update_left =
      Eigen::MatrixXd::Zero(n_electrons_, rows.size());
  Eigen::MatrixXd update_right(n_electrons_, rows.size());
  for (int local = 0; local < static_cast<int>(rows.size()); ++local) {
    update_left(rows[local], local) = 1.0;
    update_right.col(local) =
        (overlap_new.row(rows[local]) -
         core_->overlap().row(rows[local])).transpose();
  }
  WoodburyBaseUpdate base_update;
  try {
    base_update = core_->append(update_left, update_right);
  } catch (const std::runtime_error&) {
    // The caller owns the graph traversal and will establish a fresh exact
    // anchor for this pair.  append has not committed an uncertified base.
    return false;
  }
  update_base_channels(base_update.left, base_update.right);

  const int transition_rank = static_cast<int>(rows.size());
  RiContractedMoments::Table transition_left(
      moments_.channel_count(), n_electrons_ * transition_rank);
  RiContractedMoments::Table transition_right(
      moments_.channel_count(), n_electrons_ * transition_rank);
  for (Eigen::Index auxiliary = 0;
       auxiliary < moments_.channel_count();
       ++auxiliary) {
    Eigen::Map<Eigen::MatrixXd> left_factors(
        transition_left.data() +
            auxiliary * n_electrons_ * transition_rank,
        n_electrons_,
        transition_rank);
    Eigen::Map<Eigen::MatrixXd> right_factors(
        transition_right.data() +
            auxiliary * n_electrons_ * transition_rank,
        n_electrons_,
        transition_rank);
    for (int local = 0; local < transition_rank; ++local) {
      const int row = rows[local];
      left_factors.col(local) = core_->inverse_base().col(row);
      for (int left = 0; left < n_electrons_; ++left) {
        right_factors(left, local) =
            ri_factors(
                auxiliary,
                TwoElectronIndexer::packed_pair_index(
                    occupied_right_new[row], occupied_left_[left])) -
            ri_factors(
                auxiliary,
                TwoElectronIndexer::packed_pair_index(
                    occupied_right_[row], occupied_left_[left]));
      }
    }
  }
  const Eigen::MatrixXd no_inverse_update(n_electrons_, 0);
  moments_.update(
      no_inverse_update,
      no_inverse_update,
      transition_left,
      transition_right);
  occupied_right_ = occupied_right_new;
  return moments_.channels().allFinite();
}

bool WoodburyRiState::update_left(
    const std::vector<int>& occupied_left_new,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_new,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) {
  if (!valid() ||
      overlap_new.rows() != n_electrons_ ||
      overlap_new.cols() != n_electrons_ ||
      !factors_cover_strings(
          occupied_left_new, occupied_right_, ri_factors.cols())) {
    return false;
  }
  const std::vector<int> columns =
      changed_positions(occupied_left_, occupied_left_new);
  if (columns.empty()) {
    return occupied_left_new == occupied_left_ &&
        overlap_new.isApprox(core_->overlap());
  }

  Eigen::MatrixXd update_left(n_electrons_, columns.size());
  Eigen::MatrixXd update_right =
      Eigen::MatrixXd::Zero(n_electrons_, columns.size());
  for (int local = 0; local < static_cast<int>(columns.size()); ++local) {
    update_left.col(local) =
        overlap_new.col(columns[local]) -
        core_->overlap().col(columns[local]);
    update_right(columns[local], local) = 1.0;
  }
  WoodburyBaseUpdate base_update;
  try {
    base_update = core_->append(update_left, update_right);
  } catch (const std::runtime_error&) {
    return false;
  }
  update_base_channels(base_update.left, base_update.right);

  const int transition_rank = static_cast<int>(columns.size());
  RiContractedMoments::Table transition_left(
      moments_.channel_count(), n_electrons_ * transition_rank);
  RiContractedMoments::Table transition_right(
      moments_.channel_count(), n_electrons_ * transition_rank);
  for (Eigen::Index auxiliary = 0;
       auxiliary < moments_.channel_count();
       ++auxiliary) {
    Eigen::Map<Eigen::MatrixXd> left_factors(
        transition_left.data() +
            auxiliary * n_electrons_ * transition_rank,
        n_electrons_,
        transition_rank);
    Eigen::Map<Eigen::MatrixXd> right_factors(
        transition_right.data() +
            auxiliary * n_electrons_ * transition_rank,
        n_electrons_,
        transition_rank);
    right_factors.setZero();
    for (int local = 0; local < transition_rank; ++local) {
      const int column = columns[local];
      Eigen::VectorXd transition_delta(n_electrons_);
      for (int right = 0; right < n_electrons_; ++right) {
        transition_delta(right) =
            ri_factors(
                auxiliary,
                TwoElectronIndexer::packed_pair_index(
                    occupied_right_[right], occupied_left_new[column])) -
            ri_factors(
                auxiliary,
                TwoElectronIndexer::packed_pair_index(
                    occupied_right_[right], occupied_left_[column]));
      }
      left_factors.col(local).noalias() =
          core_->inverse_base() * transition_delta;
      right_factors(column, local) = 1.0;
    }
  }
  const Eigen::MatrixXd no_inverse_update(n_electrons_, 0);
  moments_.update(
      no_inverse_update,
      no_inverse_update,
      transition_left,
      transition_right);
  occupied_left_ = occupied_left_new;
  return moments_.channels().allFinite();
}

int WoodburyRiState::core_rank() const noexcept {
  return valid() ? core_->rank() : 0;
}

int WoodburyRiState::overlap_nullity() const noexcept {
  return valid() ? core_->nullity() : 0;
}

const Eigen::MatrixXd& WoodburyRiState::regular_inverse() const {
  if (!valid() || core_->rank() != 0) {
    throw std::logic_error(
        "physical overlap inverse is available only for a regular Woodbury state");
  }
  return core_->inverse_base();
}

double WoodburyRiState::overlap_determinant() const {
  if (!valid()) {
    throw std::logic_error("Woodbury RI state is not initialized");
  }
  return core_->determinant();
}

Eigen::MatrixXd WoodburyRiState::first_cofactor() const {
  if (!valid()) {
    throw std::logic_error("Woodbury RI state is not initialized");
  }
  return core_->first_cofactor();
}

Eigen::VectorXd WoodburyRiState::first_cofactor_auxiliary() const {
  if (!valid()) {
    throw std::logic_error("Woodbury RI state is not initialized");
  }
  Eigen::VectorXd result(moments_.channel_count());
  for (Eigen::Index auxiliary = 0;
       auxiliary < moments_.channel_count();
       ++auxiliary) {
    result(auxiliary) =
        core_->first_channel_contraction(channel(auxiliary));
  }
  return result;
}

double WoodburyRiState::one_electron_contraction(
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron) const {
  if (!valid()) {
    throw std::logic_error("Woodbury RI state is not initialized");
  }
  return core_->first_contraction(occupied_one_electron);
}

double WoodburyRiState::two_electron_contraction() const {
  if (!valid()) {
    throw std::logic_error("Woodbury RI state is not initialized");
  }
  if (core_->rank() == 0) {
    return core_->determinant() * moments_.second_coefficient_sum();
  }
  double result = 0.0;
  for (Eigen::Index auxiliary = 0;
       auxiliary < moments_.channel_count();
       ++auxiliary) {
    result += core_->second_channel_contraction(channel(auxiliary));
  }
  return result;
}

Eigen::MatrixXd WoodburyRiState::hamiltonian_overlap_gradient(
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) const {
  if (!valid() ||
      occupied_one_electron.rows() != n_electrons_ ||
      occupied_one_electron.cols() != n_electrons_ ||
      ri_factors.rows() != moments_.channel_count() ||
      !factors_cover_strings(
          occupied_left_, occupied_right_, ri_factors.cols())) {
    throw std::invalid_argument(
        "Woodbury RI Hamiltonian-gradient dimensions differ");
  }
  Eigen::MatrixXd result =
      core_->first_contraction_gradient(occupied_one_electron)
          .overlap_gradient;
  if (core_->rank() == 0) {
    result.noalias() += core_->determinant() *
        (moments_.second_coefficient_sum() *
             core_->inverse_base().transpose() -
         moments_.second_response_moment().transpose());
    return result;
  }
  Eigen::MatrixXd transition(n_electrons_, n_electrons_);
  for (Eigen::Index auxiliary = 0;
       auxiliary < moments_.channel_count();
       ++auxiliary) {
    for (int left = 0; left < n_electrons_; ++left) {
      for (int right = 0; right < n_electrons_; ++right) {
        transition(right, left) = ri_factors(
            auxiliary,
            TwoElectronIndexer::packed_pair_index(
                occupied_right_[right], occupied_left_[left]));
      }
    }
    result.noalias() += core_->second_channel_contraction_gradient(
        channel(auxiliary), transition).overlap_gradient;
  }
  return result;
}

Eigen::MatrixXd WoodburyRiState::regular_two_electron_inverse_gradient()
    const {
  if (!valid() || core_->rank() != 0) {
    throw std::logic_error(
        "regular RI inverse gradient requires a regular Woodbury state");
  }
  Eigen::MatrixXd result;
  result.noalias() =
      core_->overlap().transpose() *
      moments_.second_response_moment().transpose() *
      core_->overlap().transpose();
  return result;
}

WoodburyRiDirection WoodburyRiState::hamiltonian_direction(
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron,
    const Eigen::Ref<const Eigen::MatrixXd>& occupied_one_electron_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap_direction,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factor_direction,
    bool auxiliary_projection) const {
  if (!valid() ||
      occupied_one_electron.rows() != n_electrons_ ||
      occupied_one_electron.cols() != n_electrons_ ||
      occupied_one_electron_direction.rows() != n_electrons_ ||
      occupied_one_electron_direction.cols() != n_electrons_ ||
      overlap_direction.rows() != n_electrons_ ||
      overlap_direction.cols() != n_electrons_ ||
      ri_factors.rows() != moments_.channel_count() ||
      ri_factor_direction.rows() != moments_.channel_count() ||
      ri_factor_direction.cols() != ri_factors.cols() ||
      !factors_cover_strings(
          occupied_left_, occupied_right_, ri_factors.cols())) {
    throw std::invalid_argument(
        "Woodbury RI Hamiltonian-direction dimensions differ");
  }

  WoodburyRiDirection result;
  const WoodburyContractionDirection one_electron =
      core_->first_contraction_gradient_direction(
          occupied_one_electron,
          overlap_direction,
          occupied_one_electron_direction);
  result.hamiltonian = one_electron.value;
  result.hamiltonian_overlap_gradient = one_electron.overlap_gradient;
  result.first_cofactor =
      core_->first_cofactor_direction(overlap_direction);
  result.overlap_determinant =
      (core_->first_cofactor().cwiseProduct(overlap_direction)).sum();
  if (auxiliary_projection) {
    result.accepted_auxiliary.resize(moments_.channel_count());
    result.directional_auxiliary.resize(moments_.channel_count());
  }

  Eigen::MatrixXd transition(n_electrons_, n_electrons_);
  Eigen::MatrixXd transition_direction(n_electrons_, n_electrons_);
  const Eigen::MatrixXd first_cofactor = core_->first_cofactor();
  for (Eigen::Index auxiliary = 0;
       auxiliary < moments_.channel_count();
       ++auxiliary) {
    for (int left = 0; left < n_electrons_; ++left) {
      for (int right = 0; right < n_electrons_; ++right) {
        const int pair = TwoElectronIndexer::packed_pair_index(
            occupied_right_[right], occupied_left_[left]);
        transition(right, left) = ri_factors(auxiliary, pair);
        transition_direction(right, left) =
            ri_factor_direction(auxiliary, pair);
      }
    }
    const WoodburyContractionDirection two_electron =
        core_->second_factor_contraction_gradient_direction(
            transition, overlap_direction, transition_direction);
    result.hamiltonian += two_electron.value;
    result.hamiltonian_overlap_gradient.noalias() +=
        two_electron.overlap_gradient;
    if (auxiliary_projection) {
      result.accepted_auxiliary(auxiliary) =
          (first_cofactor.cwiseProduct(transition)).sum();
      result.directional_auxiliary(auxiliary) =
          (result.first_cofactor.cwiseProduct(transition)).sum() +
          (first_cofactor.cwiseProduct(transition_direction)).sum();
    }
  }
  return result;
}

}  // namespace xmvb::vb
