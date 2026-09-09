#pragma once

// Compatibility header. New code should include
// "vbscf/diagnostics/orbital_chart_audit.hpp".
#include "vb/orbital/sparse_orbital_parameter_view.hpp"
#include "vbscf/diagnostics/orbital_chart_audit.hpp"

namespace xmvb::vb {

using SparseOrbitalGaugeAudit = OrbitalChartAudit;

inline SparseOrbitalGaugeAudit audit_sparse_orbital_gauge(
    const OrbitalPreparationInput& input,
    const SparseParameterLayout& parameter_layout,
    const Eigen::MatrixXd* current_packed_reduced_basis = nullptr) {
  return audit_orbital_chart(
      input, parameter_layout, current_packed_reduced_basis);
}

}  // namespace xmvb::vb
