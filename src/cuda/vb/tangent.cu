#include "cuda/vb/tangent.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/context.hpp"

namespace xmvb::cuda {
namespace {

__global__ void subtract_first(double* vector, double value) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        vector[0] -= value;
}

void device_copy(Context& context, double* destination, const double* source, std::size_t count) {
    check_cuda(cudaMemcpyAsync(destination,
                               source,
                               count * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               context.stream()),
               "CI tangent device copy");
}

} // namespace

SingleStateCiTangent::SingleStateCiTangent(Context& context,
                                           int structures,
                                           linalg::ConstMatrixView selected_state,
                                           linalg::ConstMatrixView selected_hamiltonian_image,
                                           linalg::ConstMatrixView selected_metric_image,
                                           double selected_energy,
                                           double normalized_weight,
                                           SingleStateCiActions actions)
    : context_(&context), structures_(structures), selected_energy_(selected_energy),
      coordinate_scale_(std::sqrt(2.0 * normalized_weight)), reflector_factor_(0.0),
      reflected_first_(0.0), actions_(std::move(actions)),
      selected_hamiltonian_image_(selected_hamiltonian_image),
      selected_metric_image_(selected_metric_image), selected_(context, structures, 1),
      reflector_(context, structures, 1), ambient_(context, structures, 1),
      hessian_(context, structures, 1), metric_(context, structures, 1),
      work_(context, structures, 1) {
    if (structures < 2 || !std::isfinite(selected_energy) || !(normalized_weight > 0.0) ||
        !std::isfinite(normalized_weight) || !std::isfinite(coordinate_scale_) ||
        !actions_.hamiltonian_metric || !actions_.metric)
        throw std::invalid_argument("invalid single-state CI tangent inputs");
    check_ambient(selected_state);
    check_ambient(selected_hamiltonian_image_);
    check_ambient(selected_metric_image_);
    check_cublas(cublasSetPointerMode(context.blas(), CUBLAS_POINTER_MODE_HOST),
                 "CI tangent cublasSetPointerMode");
    device_copy(context, selected_.data(), selected_state.data, structures_);
    device_copy(context, metric_.data(), selected_metric_image_.data, structures_);
    double selected_norm = 0.0;
    check_cublas(
        cublasDdot(
            context.blas(), structures_, selected_.data(), 1, metric_.data(), 1, &selected_norm),
        "CI tangent selected metric norm");
    const double metric_tolerance =
        1.0e3 * std::numeric_limits<double>::epsilon() * std::max(1, structures_);
    if (!std::isfinite(selected_norm) || std::abs(selected_norm - 1.0) > metric_tolerance)
        throw std::invalid_argument("single-state CI tangent requires S-normalized selected state");
    device_copy(context, reflector_.data(), metric_.data(), structures_);
    double metric_image_norm = 0.0;
    check_cublas(cublasDnrm2(context.blas(), structures_, reflector_.data(), 1, &metric_image_norm),
                 "CI tangent constraint norm");
    double first = 0.0;
    context.copy_to_host(&first, reflector_.data(), sizeof(double));
    if (!(metric_image_norm > 0.0) || !std::isfinite(metric_image_norm) || !std::isfinite(first))
        throw std::invalid_argument("single-state CI tangent constraint is invalid");
    const double reflected_first = -std::copysign(metric_image_norm, first);
    reflected_first_ = reflected_first;
    subtract_first<<<1, 1, 0, context.stream()>>>(reflector_.data(), reflected_first);
    check_cuda(cudaGetLastError(), "CI tangent reflector kernel");
    double squared = 0.0;
    check_cublas(
        cublasDdot(
            context.blas(), structures_, reflector_.data(), 1, reflector_.data(), 1, &squared),
        "CI tangent reflector norm");
    if (!(squared > 0.0) || !std::isfinite(squared))
        throw std::runtime_error("single-state CI reflector is invalid");
    reflector_factor_ = 2.0 / squared;
}

void SingleStateCiTangent::check_ambient(linalg::ConstMatrixView view) const {
    if (view.rows != structures_ || view.cols != 1 || view.ld < view.rows || !view.data)
        throw std::invalid_argument("invalid CI tangent ambient vector");
}
void SingleStateCiTangent::check_ambient(linalg::MatrixView view) const {
    check_ambient(static_cast<linalg::ConstMatrixView>(view));
}
void SingleStateCiTangent::check_coordinates(linalg::ConstMatrixView view) const {
    if (view.rows != tangent_size() || view.cols != 1 || view.ld < view.rows || !view.data)
        throw std::invalid_argument("invalid CI tangent coordinates");
}
void SingleStateCiTangent::check_coordinates(linalg::MatrixView view) const {
    check_coordinates(static_cast<linalg::ConstMatrixView>(view));
}

void SingleStateCiTangent::project_coordinates(linalg::ConstMatrixView ambient,
                                               linalg::MatrixView coordinates) const {
    check_ambient(ambient);
    check_coordinates(coordinates);
    device_copy(*context_, work_.data(), ambient.data, structures_);
    double inner = 0.0;
    check_cublas(
        cublasDdot(context_->blas(), structures_, reflector_.data(), 1, work_.data(), 1, &inner),
        "CI tangent project dot");
    const double factor = -reflector_factor_ * inner;
    check_cublas(
        cublasDaxpy(context_->blas(), structures_, &factor, reflector_.data(), 1, work_.data(), 1),
        "CI tangent project reflector");
    device_copy(*context_, coordinates.data, work_.data() + 1, tangent_size());
}

void SingleStateCiTangent::expand_coordinates(linalg::ConstMatrixView coordinates,
                                              linalg::MatrixView ambient) const {
    check_coordinates(coordinates);
    check_ambient(ambient);
    check_cuda(cudaMemsetAsync(ambient.data, 0, structures_ * sizeof(double), context_->stream()),
               "CI tangent expand zero");
    device_copy(*context_, ambient.data + 1, coordinates.data, tangent_size());
    double inner = 0.0;
    check_cublas(
        cublasDdot(context_->blas(), structures_, reflector_.data(), 1, ambient.data, 1, &inner),
        "CI tangent expand dot");
    const double factor = -reflector_factor_ * inner;
    check_cublas(
        cublasDaxpy(context_->blas(), structures_, &factor, reflector_.data(), 1, ambient.data, 1),
        "CI tangent expand reflector");
}

void SingleStateCiTangent::coefficient_response_coordinates(linalg::ConstMatrixView coordinates,
                                                            linalg::MatrixView response) const {
    expand_coordinates(coordinates, response);
    const double factor = 1.0 / coordinate_scale_;
    check_cublas(cublasDscal(context_->blas(), structures_, &factor, response.data, 1),
                 "CI tangent coefficient scale");
}

double SingleStateCiTangent::maximum_coefficient_component_coordinates(
    linalg::ConstMatrixView coordinates) const {
    expand_coordinates(coordinates, ambient_.view());
    const double inverse_scale = 1.0 / coordinate_scale_;
    if (actions_.maximum_component) {
        check_cublas(cublasDscal(context_->blas(), structures_, &inverse_scale, ambient_.data(), 1),
                     "CI tangent maximum coefficient scale");
        return actions_.maximum_component(ambient_.view());
    }
    int one_based = 0;
    check_cublas(cublasIdamax(context_->blas(), structures_, ambient_.data(), 1, &one_based),
                 "CI tangent maximum coefficient");
    double largest = 0.0;
    context_->copy_to_host(&largest, ambient_.data() + one_based - 1, sizeof(double));
    return std::abs(largest) / coordinate_scale_;
}

void SingleStateCiTangent::apply_inverse_augmented_hessian_diagonal(
    linalg::ConstMatrixView covector, double eigenvalue, linalg::MatrixView direction) const {
    check_coordinates(covector);
    check_coordinates(direction);
    if (!actions_.augmented_inverse || !std::isfinite(eigenvalue))
        throw std::invalid_argument("CI tangent augmented inverse is unavailable");
    expand_coordinates(covector, ambient_.view());
    if (actions_.inverse_constraint == CiInverseConstraint::Diagonal) {
        // Bounded selected spaces use Q^T diag(H-(E+shift)M)^-1 Q.
        // Unlike the positive spin model, no Schur inverse is present on CPU.
        actions_.augmented_inverse(ambient_.view(), eigenvalue, hessian_.view());
        project_coordinates(hessian_.view(), direction);
        return;
    }
    // reflector = S c - reflected_first * e_1. Reconstruct the cached selected
    // metric image from the Householder data without another H/M action.
    device_copy(*context_, metric_.data(), reflector_.data(), structures_);
    subtract_first<<<1, 1, 0, context_->stream()>>>(metric_.data(), -reflected_first_);
    check_cuda(cudaGetLastError(), "CI tangent metric constraint reconstruction");
    if (actions_.inverse_constraint == CiInverseConstraint::Carrier) {
        // FULL uses P_L R P_R, not a Schur inverse of R: the signed carrier
        // diagonal can make R indefinite. This is the CPU FULL branch exactly.
        double selected_rhs = 0.0;
        check_cublas(cublasDdot(context_->blas(),
                                structures_,
                                selected_.data(),
                                1,
                                ambient_.data(),
                                1,
                                &selected_rhs),
                     "CI tangent carrier right constraint dot");
        device_copy(*context_, work_.data(), ambient_.data(), structures_);
        const double negative = -selected_rhs;
        check_cublas(
            cublasDaxpy(
                context_->blas(), structures_, &negative, metric_.data(), 1, work_.data(), 1),
            "CI tangent carrier right constraint projection");
        actions_.augmented_inverse(work_.view(), eigenvalue, hessian_.view());
        double left_weight = 0.0;
        check_cublas(
            cublasDdot(
                context_->blas(), structures_, metric_.data(), 1, hessian_.data(), 1, &left_weight),
            "CI tangent carrier left constraint dot");
        const double minus_left = -left_weight;
        check_cublas(cublasDaxpy(context_->blas(),
                                 structures_,
                                 &minus_left,
                                 selected_.data(),
                                 1,
                                 hessian_.data(),
                                 1),
                     "CI tangent carrier left constraint projection");
        project_coordinates(hessian_.view(), direction);
        return;
    }
    // The CPU spin branch directly solves the ambient RHS and S c, then forms
    // the positive one-state Schur complement; it does not use P_R here.
    actions_.augmented_inverse(ambient_.view(), eigenvalue, hessian_.view());
    actions_.augmented_inverse(metric_.view(), eigenvalue, ambient_.view());
    double numerator = 0.0, denominator = 0.0;
    check_cublas(
        cublasDdot(
            context_->blas(), structures_, metric_.data(), 1, hessian_.data(), 1, &numerator),
        "CI tangent Schur numerator");
    check_cublas(
        cublasDdot(
            context_->blas(), structures_, metric_.data(), 1, ambient_.data(), 1, &denominator),
        "CI tangent Schur denominator");
    if (!(denominator > 0.0) || !std::isfinite(denominator) || !std::isfinite(numerator))
        throw std::runtime_error("CI tangent AH constraint model is not positive definite");
    const double factor = -numerator / denominator;
    check_cublas(
        cublasDaxpy(context_->blas(), structures_, &factor, ambient_.data(), 1, hessian_.data(), 1),
        "CI tangent left constraint projection");
    project_coordinates(hessian_.view(), direction);
}

void SingleStateCiTangent::apply_metric_coordinates(linalg::ConstMatrixView coordinates,
                                                    linalg::MatrixView metric_image) const {
    check_coordinates(metric_image);
    expand_coordinates(coordinates, ambient_.view());
    actions_.metric(ambient_.view(), metric_.view());
    project_coordinates(metric_.view(), metric_image);
}

void SingleStateCiTangent::apply_images_coordinates(linalg::ConstMatrixView coordinates,
                                                    linalg::MatrixView hessian_image,
                                                    linalg::MatrixView metric_image) const {
    check_coordinates(hessian_image);
    check_coordinates(metric_image);
    if (hessian_image.data == metric_image.data)
        throw std::invalid_argument("CI tangent Hessian and metric outputs alias");
    expand_coordinates(coordinates, ambient_.view());
    actions_.hamiltonian_metric(ambient_.view(), hessian_.view(), metric_.view());
    const double minus_energy = -selected_energy_;
    check_cublas(
        cublasDaxpy(
            context_->blas(), structures_, &minus_energy, metric_.data(), 1, hessian_.data(), 1),
        "CI tangent shifted Hessian");
    project_coordinates(hessian_.view(), hessian_image);
    project_coordinates(metric_.view(), metric_image);
}

void SingleStateCiTangent::origin_gradient(linalg::MatrixView gradient) const {
    check_coordinates(gradient);
    device_copy(*context_, hessian_.data(), selected_hamiltonian_image_.data, structures_);
    const double minus_energy = -selected_energy_;
    check_cublas(cublasDaxpy(context_->blas(),
                             structures_,
                             &minus_energy,
                             selected_metric_image_.data,
                             1,
                             hessian_.data(),
                             1),
                 "CI tangent origin residual");
    const double scale = coordinate_scale_;
    check_cublas(cublasDscal(context_->blas(), structures_, &scale, hessian_.data(), 1),
                 "CI tangent origin gradient scale");
    project_coordinates(hessian_.view(), gradient);
}

} // namespace xmvb::cuda
