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

Eigen::Map<Eigen::MatrixXd> WoodburyRiState::channel(
    Eigen::Index auxiliary) {
  return Eigen::Map<Eigen::MatrixXd>(
      channels_.data() + auxiliary * n_electrons_ * n_electrons_,
      n_electrons_,
      n_electrons_);
}

Eigen::Map<const Eigen::MatrixXd> WoodburyRiState::channel(
    Eigen::Index auxiliary) const {
  return Eigen::Map<const Eigen::MatrixXd>(
      channels_.data() + auxiliary * n_electrons_ * n_electrons_,
      n_electrons_,
      n_electrons_);
}

bool WoodburyRiState::initialize(
    const std::vector<int>& occupied_left,
    const std::vector<int>& occupied_right,
    const Eigen::Ref<const Eigen::MatrixXd>& overlap,
    const Eigen::Ref<const Eigen::MatrixXd>& ri_factors) {
  core_.reset();
  channels_.resize(0, 0);
  n_electrons_ = static_cast<int>(occupied_left.size());
  if (static_cast<int>(occupied_right.size()) != n_electrons_ ||
      overlap.rows() != n_electrons_ || overlap.cols() != n_electrons_ ||
      !factors_cover_strings(
          occupied_left, occupied_right, ri_factors.cols())) {
    return false;
  }

  auto core = std::make_unique<WoodburyCore>(overlap);
  channels_.resize(
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
    channel(auxiliary).noalias() = core->inverse_base() * transition;
  }
  if (!channels_.allFinite()) {
    channels_.resize(0, 0);
    return false;
  }
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
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
       ++auxiliary) {
    Eigen::Map<Eigen::MatrixXd> value = channel(auxiliary);
    value.noalias() += left * (right.transpose() * value);
  }
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
  const WoodburyBaseUpdate base_update =
      core_->append(update_left, update_right);
  update_base_channels(base_update.left, base_update.right);

  Eigen::MatrixXd transition_delta =
      Eigen::MatrixXd::Zero(n_electrons_, n_electrons_);
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
       ++auxiliary) {
    for (const int row : rows) {
      for (int left = 0; left < n_electrons_; ++left) {
        transition_delta(row, left) =
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
    Eigen::Map<Eigen::MatrixXd> value = channel(auxiliary);
    for (const int row : rows) {
      value.noalias() +=
          core_->inverse_base().col(row) * transition_delta.row(row);
    }
  }
  occupied_right_ = occupied_right_new;
  return channels_.allFinite();
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
  const WoodburyBaseUpdate base_update =
      core_->append(update_left, update_right);
  update_base_channels(base_update.left, base_update.right);

  Eigen::MatrixXd transition_delta =
      Eigen::MatrixXd::Zero(n_electrons_, n_electrons_);
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
       ++auxiliary) {
    for (const int column : columns) {
      for (int right = 0; right < n_electrons_; ++right) {
        transition_delta(right, column) =
            ri_factors(
                auxiliary,
                TwoElectronIndexer::packed_pair_index(
                    occupied_right_[right], occupied_left_new[column])) -
            ri_factors(
                auxiliary,
                TwoElectronIndexer::packed_pair_index(
                    occupied_right_[right], occupied_left_[column]));
      }
    }
    Eigen::Map<Eigen::MatrixXd> value = channel(auxiliary);
    for (const int column : columns) {
      value.col(column).noalias() +=
          core_->inverse_base() * transition_delta.col(column);
    }
  }
  occupied_left_ = occupied_left_new;
  return channels_.allFinite();
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
  Eigen::VectorXd result(channels_.rows());
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
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
  double result = 0.0;
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
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
      ri_factors.rows() != channels_.rows() ||
      !factors_cover_strings(
          occupied_left_, occupied_right_, ri_factors.cols())) {
    throw std::invalid_argument(
        "Woodbury RI Hamiltonian-gradient dimensions differ");
  }
  Eigen::MatrixXd result =
      core_->first_contraction_gradient(occupied_one_electron)
          .overlap_gradient;
  Eigen::MatrixXd transition(n_electrons_, n_electrons_);
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
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
      ri_factors.rows() != channels_.rows() ||
      ri_factor_direction.rows() != channels_.rows() ||
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
    result.accepted_auxiliary.resize(channels_.rows());
    result.directional_auxiliary.resize(channels_.rows());
  }

  Eigen::MatrixXd transition(n_electrons_, n_electrons_);
  Eigen::MatrixXd transition_direction(n_electrons_, n_electrons_);
  const Eigen::MatrixXd first_cofactor = core_->first_cofactor();
  for (Eigen::Index auxiliary = 0;
       auxiliary < channels_.rows();
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
