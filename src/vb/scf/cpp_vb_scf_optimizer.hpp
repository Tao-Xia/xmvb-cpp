#pragma once

// Compatibility header. New code should include
// "vbscf/optimization/vbscf_optimizer.hpp".
#include "vb/scf/cpp_vb_scf_evaluator.hpp"
#include "vb/scf/cpp_vb_scf_optimizer_result.hpp"
#include "vbscf/optimization/vbscf_optimizer.hpp"

namespace xmvb::vb {

using CppVbScfOptimizerBackend = VbScfOptimizerBackend;
using CppVbScfOptimizerOptions = VbScfOptimizerOptions;
using CppVbScfOptimizer = VbScfOptimizer;

inline bool cpp_vb_scf_optimizer_backend_supported(
    CppVbScfOptimizerBackend backend) {
  return vbscf_optimizer_backend_supported(backend);
}

inline const char* cpp_vb_scf_optimizer_backend_name(
    CppVbScfOptimizerBackend backend) {
  return vbscf_optimizer_backend_name(backend);
}

}  // namespace xmvb::vb
