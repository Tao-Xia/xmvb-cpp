#pragma once

#include <memory>

#include "linalg/views.hpp"

namespace xmvb::cuda {

class Context;
class StructureAction;
enum class ProductMap;

/** Bounded exact W^T product-map W executor for a complete spin sector. */
class FullAction {
 public:
  FullAction(Context& context,const StructureAction& action);
  ~FullAction();
  FullAction(const FullAction&)=delete;
  FullAction& operator=(const FullAction&)=delete;

  void apply(ProductMap map,linalg::ConstMatrixView source,
             linalg::MatrixView image,linalg::ConstMatrixView energies,
             double eigenvalue) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xmvb::cuda
