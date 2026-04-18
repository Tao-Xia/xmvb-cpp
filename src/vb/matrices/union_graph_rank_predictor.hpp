#pragma once

#include "vb/matrices/union_graph_screening.hpp"

namespace xmvb::vb {

enum class UnionGraphRankPredictionReason {
  SingleComponentExact,
  ZeroOffblockNoGeneral,
  ConservativeMultiGeneralRankOne,
  LowSecondSingular,
  LowThirdSingular,
  FallbackToMaxRank,
};

const char* union_graph_rank_prediction_reason_name(
    UnionGraphRankPredictionReason reason);

struct UnionGraphRankPredictorOptions {
  double zero_rank_offblock_threshold = 0.0;
  double rank_one_second_singular_threshold = 0.0;
  double rank_two_third_singular_threshold = 0.0;
  int max_predicted_rank = -1;
};

struct UnionGraphRankPrediction {
  int predicted_rank_cap = 0;
  UnionGraphRankPredictionReason reason =
      UnionGraphRankPredictionReason::SingleComponentExact;
};

UnionGraphRankPrediction predict_union_graph_rank_cap(
    const UnionGraphScreeningSummary& screening_summary,
    UnionGraphRankPredictorOptions options = {});

}  // namespace xmvb::vb
