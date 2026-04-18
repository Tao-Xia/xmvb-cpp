#include "vb/scf/cpp_active_space_gradient_result_utils.hpp"

#include <stdexcept>

#include "vb/orbital/active_space_two_electron_utils.hpp"

namespace xmvb::vb {

void initialize_active_space_gradient_probe_result(
    const CppVbInput& input,
    const TimedPreparedActiveSpaceContext& timed_active_space_context,
    CppActiveSpaceGradientResult* result) {
  if (result == nullptr) {
    throw std::invalid_argument("result must not be null");
  }

  const auto& timings = timed_active_space_context.timings;
  const auto& prepared_active_space =
      timed_active_space_context.prepared_active_space;

  result->orbital_preparation_wall_time_seconds =
      timings.orbital_preparation_wall_time_seconds;
  result->ao_effective_one_electron_wall_time_seconds =
      timings.ao_effective_one_electron_wall_time_seconds;
  result->active_one_electron_wall_time_seconds =
      timings.active_one_electron_wall_time_seconds;
  result->active_two_electron_wall_time_seconds =
      timings.active_two_electron_wall_time_seconds;
  result->structure_matrix_wall_time_seconds = 0.0;
  result->eigensolver_wall_time_seconds = 0.0;

  result->orbital_preparation_result = prepared_active_space.orbital_result;
  result->ao_effective_one_electron_result =
      prepared_active_space.ao_effective_one_electron_result;
  result->active_orbital_overlap_matrix =
      prepared_active_space.orbital_result.active_orbital_overlap_matrix;
  result->active_space_one_electron_result =
      prepared_active_space.active_space_one_electron_result;
  result->active_space_two_electron_result =
      prepared_active_space.active_space_two_electron_result;
  result->active_orbital_overlap_gradient.assign(
      prepared_active_space.orbital_result.active_orbital_overlap_matrix.size(),
      0.0);
  result->active_one_electron_gradient.assign(
      prepared_active_space.active_space_one_electron_result.h1e_act.size(),
      0.0);
  result->packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(
          input.orbital_preparation_input.n_active_orbitals),
      0.0);
}

void initialize_active_space_gradient_probe_result(
    const CppVbInput& input,
    TimedPreparedActiveSpaceContext&& timed_active_space_context,
    CppActiveSpaceGradientResult* result) {
  if (result == nullptr) {
    throw std::invalid_argument("result must not be null");
  }

  const auto& timings = timed_active_space_context.timings;
  auto& prepared_active_space =
      timed_active_space_context.prepared_active_space;

  result->orbital_preparation_wall_time_seconds =
      timings.orbital_preparation_wall_time_seconds;
  result->ao_effective_one_electron_wall_time_seconds =
      timings.ao_effective_one_electron_wall_time_seconds;
  result->active_one_electron_wall_time_seconds =
      timings.active_one_electron_wall_time_seconds;
  result->active_two_electron_wall_time_seconds =
      timings.active_two_electron_wall_time_seconds;
  result->structure_matrix_wall_time_seconds = 0.0;
  result->eigensolver_wall_time_seconds = 0.0;

  result->orbital_preparation_result =
      std::move(prepared_active_space.orbital_result);
  result->ao_effective_one_electron_result =
      std::move(prepared_active_space.ao_effective_one_electron_result);
  result->active_orbital_overlap_matrix =
      result->orbital_preparation_result.active_orbital_overlap_matrix;
  result->active_space_one_electron_result =
      std::move(prepared_active_space.active_space_one_electron_result);
  result->active_space_two_electron_result =
      std::move(prepared_active_space.active_space_two_electron_result);
  result->active_orbital_overlap_gradient.assign(
      result->orbital_preparation_result.active_orbital_overlap_matrix.size(),
      0.0);
  result->active_one_electron_gradient.assign(
      result->active_space_one_electron_result.h1e_act.size(),
      0.0);
  result->packed_active_two_electron_gradient.assign(
      packed_active_two_electron_integral_count(
          input.orbital_preparation_input.n_active_orbitals),
      0.0);
}

}  // namespace xmvb::vb
