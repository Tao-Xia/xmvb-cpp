#pragma once

#include <functional>

#include <Eigen/Core>

#include "cuda/memory/buffer.hpp"
#include "linalg/views.hpp"

namespace xmvb::cuda {

/** Certified sampled balancing inverse at one frozen accepted point.
 * Full-length basis and H/M images remain on the device. Only small projected
 * matrices and scalar reductions reach the host. Neither observe nor apply
 * evaluates H or M; supplied images must belong to the same accepted point.
 */
class DeviceCoarsePreconditioner {
 public:
  using BaseInverse=std::function<void(linalg::ConstMatrixView,
                                        linalg::MatrixView)>;

  DeviceCoarsePreconditioner(Context& context,int dimension,int capacity,
                             double lindep,double action_accuracy=0.0);
  DeviceCoarsePreconditioner(const DeviceCoarsePreconditioner&)=delete;
  DeviceCoarsePreconditioner& operator=(const DeviceCoarsePreconditioner&)=delete;

  /** Admit a rank-resolved sample only after H and M reciprocity certification. */
  void observe(linalg::ConstMatrixView direction,linalg::ConstMatrixView image,
               linalg::ConstMatrixView metric_image);
  /** Signed symmetric balancing inverse with exactly one base inverse call. */
  void apply(linalg::ConstMatrixView residual,double eigenvalue,
             const BaseInverse& inverse,linalg::MatrixView output) const;

  int size() const noexcept { return size_; }
  int dimension() const noexcept { return dimension_; }

 private:
  void validate_vector(linalg::ConstMatrixView vector) const;
  void ensure_finite(linalg::ConstMatrixView first,
      linalg::ConstMatrixView second={},linalg::ConstMatrixView third={}) const;
  double dot(linalg::ConstMatrixView left,linalg::ConstMatrixView right) const;
  double norm(linalg::ConstMatrixView vector) const;
  double frobenius(linalg::ConstMatrixView panel,int columns) const;
  double projection_roundoff(linalg::ConstMatrixView basis,
      linalg::ConstMatrixView action,const Eigen::MatrixXd& projected) const;
  void copy(linalg::MatrixView out,linalg::ConstMatrixView in) const;
  void axpy(linalg::MatrixView out,double coefficient,
            linalg::ConstMatrixView in) const;
  void scale(linalg::MatrixView out,double coefficient) const;

  Context* context_;
  int dimension_,capacity_,size_=0;
  double lindep_,action_accuracy_;
  Matrix<double> basis_,images_,metrics_;
  mutable Matrix<double> sample_v_,sample_h_,sample_m_;
  mutable Buffer<int> finite_;
  Eigen::MatrixXd hessian_projection_,metric_projection_,sample_factor_;
};

}  // namespace xmvb::cuda
