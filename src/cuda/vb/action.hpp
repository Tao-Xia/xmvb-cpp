#pragma once

#include <memory>
#include <Eigen/Core>

#include "cuda/memory/context.hpp"
#include "linalg/views.hpp"
#include "vbscf/structures/assembly/action.hpp"

namespace xmvb::cuda {

/** Exact GPU H/M action in the original selected VB structure coordinates.
 * The sparse incidence and elementary orbital transformations are borrowed
 * from the CPU mathematical representation and uploaded once. Numerical
 * action work uses one RHS at a time and bounded channel slabs; no structure
 * square or dense compound orbital transformation is formed.
 */
class StructureAction {
 public:
  StructureAction(Context& context, const vb::StructureAction& source);
  ~StructureAction();
  StructureAction(const StructureAction&) = delete;
  StructureAction& operator=(const StructureAction&) = delete;

  void apply(linalg::ConstMatrixView vectors,
             linalg::MatrixView hamiltonian, linalg::MatrixView overlap) const;
  vb::StructureActionResult apply(
      const Eigen::Ref<const Eigen::MatrixXd>& vectors) const;
  int size() const noexcept;
  int n_structures() const noexcept { return size(); }
  const vb::StructureDiagonal& preconditioner_diagonal() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace xmvb::cuda
