#include "vb/matrices/union_graph_rank_predictor.hpp"

#include <algorithm>
#include <stdexcept>

namespace xmvb::vb {

namespace {

constexpr int kConservativeRankOneMinGeneralComponents = 2;
constexpr double kConservativeRankOneOffblockThreshold =
    7.170436807639788e-02;
constexpr double kConservativeRankOneSecondSingularThreshold =
    7.021155999642759e-07;

bool qualifies_for_conservative_multi_general_rank_one(
    const UnionGraphScreeningSummary& screening_summary,
    int capped_available_rank) {
  return capped_available_rank > 1 &&
      screening_summary.n_general_components >=
          kConservativeRankOneMinGeneralComponents &&
      screening_summary.offblock_overlap_fraction <=
          kConservativeRankOneOffblockThreshold &&
      screening_summary.max_cross_block_second_singular <=
          kConservativeRankOneSecondSingularThreshold;
}

}  // namespace

const char* union_graph_rank_prediction_reason_name(
    UnionGraphRankPredictionReason reason) {
  switch (reason) {
    case UnionGraphRankPredictionReason::SingleComponentExact:
      return "single_component_exact";
    case UnionGraphRankPredictionReason::ZeroOffblockNoGeneral:
      return "zero_offblock_no_general";
    case UnionGraphRankPredictionReason::ConservativeMultiGeneralRankOne:
      return "conservative_multi_general_rank_one";
    case UnionGraphRankPredictionReason::LowSecondSingular:
      return "low_second_singular";
    case UnionGraphRankPredictionReason::LowThirdSingular:
      return "low_third_singular";
    case UnionGraphRankPredictionReason::FallbackToMaxRank:
      return "fallback_to_max_rank";
  }
  return "unknown";
}

UnionGraphRankPrediction predict_union_graph_rank_cap(
    const UnionGraphScreeningSummary& screening_summary,
    UnionGraphRankPredictorOptions options) {
  if (options.max_predicted_rank < -1) {
    throw std::invalid_argument("max_predicted_rank must be >= -1");
  }
  if (options.zero_rank_offblock_threshold < 0.0) {
    throw std::invalid_argument("zero_rank_offblock_threshold must be non-negative");
  }
  if (options.rank_one_second_singular_threshold < 0.0) {
    throw std::invalid_argument("rank_one_second_singular_threshold must be non-negative");
  }
  if (options.rank_two_third_singular_threshold < 0.0) {
    throw std::invalid_argument("rank_two_third_singular_threshold must be non-negative");
  }

  const int available_rank = screening_summary.max_numerical_rank;
  const int capped_available_rank =
      options.max_predicted_rank >= 0
          ? std::min(options.max_predicted_rank, available_rank)
          : available_rank;

  UnionGraphRankPrediction prediction;
  if (screening_summary.component_count <= 1) {
    prediction.predicted_rank_cap = 0;
    prediction.reason = UnionGraphRankPredictionReason::SingleComponentExact;
    return prediction;
  }

  if (screening_summary.offblock_overlap_fraction <=
          options.zero_rank_offblock_threshold &&
      screening_summary.n_general_components == 0) {
    prediction.predicted_rank_cap = 0;
    prediction.reason = UnionGraphRankPredictionReason::ZeroOffblockNoGeneral;
    return prediction;
  }

  if (qualifies_for_conservative_multi_general_rank_one(
          screening_summary,
          capped_available_rank)) {
    prediction.predicted_rank_cap = 1;
    prediction.reason =
        UnionGraphRankPredictionReason::ConservativeMultiGeneralRankOne;
    return prediction;
  }

  if (screening_summary.max_cross_block_second_singular <=
      options.rank_one_second_singular_threshold) {
    prediction.predicted_rank_cap = std::min(1, capped_available_rank);
    prediction.reason = UnionGraphRankPredictionReason::LowSecondSingular;
    return prediction;
  }

  if (screening_summary.max_cross_block_third_singular <=
      options.rank_two_third_singular_threshold) {
    prediction.predicted_rank_cap = std::min(2, capped_available_rank);
    prediction.reason = UnionGraphRankPredictionReason::LowThirdSingular;
    return prediction;
  }

  prediction.predicted_rank_cap = capped_available_rank;
  prediction.reason = UnionGraphRankPredictionReason::FallbackToMaxRank;
  return prediction;
}

}  // namespace xmvb::vb
