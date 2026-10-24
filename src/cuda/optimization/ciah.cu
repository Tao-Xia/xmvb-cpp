#include "cuda/optimization/ciah.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/memory/context.hpp"
#include "mcscf/optimization/ciah/driver_core.hpp"

namespace xmvb::cuda {
namespace {

__global__ void finite_vector_kernel(const double* values, std::int64_t count, int* finite) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(values[i]))
        atomicExch(finite, 0);
}

__global__ void absolute_products(const double* const* left,
                                  const double* const* right,
                                  std::int64_t count,
                                  mcscf::detail::CiahProduct* products) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    double value = 0.0;
    for (std::int64_t i = threadIdx.x; i < count; i += blockDim.x)
        value += fabs(left[blockIdx.x][i]) * fabs(right[blockIdx.x][i]);
    value = Reduce(scratch).Sum(value);
    if (threadIdx.x == 0)
        products[blockIdx.x].magnitude = value;
}

__global__ void ones_vector_kernel(double* values, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        values[i] = 1.0;
}

void check_vector_view(linalg::ConstMatrixView view, std::int64_t dimension) {
    if (view.rows != dimension || view.cols != 1 || view.ld < view.rows || view.data == nullptr)
        throw std::invalid_argument("invalid CUDA CIAH vector view");
}

Matrix<double>
copy_vector(Context& context, std::int64_t dimension, linalg::ConstMatrixView source) {
    check_vector_view(source, dimension);
    Matrix<double> result(context, dimension, 1);
    check_cuda(cudaMemcpyAsync(result.data(),
                               source.data,
                               static_cast<std::size_t>(dimension) * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               context.stream()),
               "CIAH device vector copy");
    return result;
}

struct DeviceBackend {
    using Vector = Matrix<double>;
    Context& context;
    std::int64_t n;
    DeviceCiahOptions options;
    mutable Buffer<int> finite_flag;

    DeviceBackend(Context& context_, std::int64_t n_, DeviceCiahOptions options_)
        : context(context_), n(n_), options(std::move(options_)), finite_flag(context_, 1) {
        if (n <= 0 || !options.apply_images)
            throw std::invalid_argument("invalid CUDA CIAH dimension or action");
        check_cublas(cublasSetPointerMode(context.blas(), CUBLAS_POINTER_MODE_HOST),
                     "CIAH cublasSetPointerMode");
    }
    DeviceBackend(DeviceBackend&&) = default;
    DeviceBackend(const DeviceBackend&) = delete;

    std::int64_t size() const noexcept {
        return n;
    }
    std::int64_t dimension(const Vector& v) const noexcept {
        return v.cols() == 1 ? v.rows() : -1;
    }
    bool all_finite(const Vector& v) const {
        check_cuda(cudaMemsetAsync(finite_flag.data(), 1, sizeof(int), context.stream()),
                   "CIAH finite flag init");
        constexpr int threads = 256;
        const int blocks = blas_int((n + threads - 1) / threads);
        finite_vector_kernel<<<blocks, threads, 0, context.stream()>>>(
            v.data(), n, finite_flag.data());
        check_cuda(cudaGetLastError(), "CIAH finite kernel");
        int finite = 0;
        context.copy_to_host(&finite, finite_flag.data(), sizeof(int));
        return finite != 0;
    }
    Vector zeros() const {
        Vector v(context, n, 1);
        check_cuda(cudaMemsetAsync(
                       v.data(), 0, static_cast<std::size_t>(n) * sizeof(double), context.stream()),
                   "CIAH vector zero");
        return v;
    }
    Vector ones() const {
        Vector v = zeros();
        constexpr int threads = 256;
        const int blocks = blas_int((n + threads - 1) / threads);
        ones_vector_kernel<<<blocks, threads, 0, context.stream()>>>(v.data(), n);
        check_cuda(cudaGetLastError(), "CIAH ones kernel");
        return v;
    }
    Vector clone(const Vector& v) const {
        return copy_vector(context, n, v.view());
    }
    void axpy(Vector& y, double a, const Vector& x) const {
        check_cublas(cublasDaxpy(context.blas(), blas_int(n), &a, x.data(), 1, y.data(), 1),
                     "CIAH cublasDaxpy");
    }
    void scal(Vector& x, double a) const {
        check_cublas(cublasDscal(context.blas(), blas_int(n), &a, x.data(), 1), "CIAH cublasDscal");
    }
    double dot(const Vector& a, const Vector& b) const {
        double value = 0.0;
        check_cublas(cublasDdot(context.blas(), blas_int(n), a.data(), 1, b.data(), 1, &value),
                     "CIAH cublasDdot");
        return value;
    }
    std::vector<mcscf::detail::CiahProduct>
    products(const std::vector<std::pair<const Vector*, const Vector*>>& requests) const {
        const nvtx3::scoped_range range{"ciah.projection_products"};
        using Product = mcscf::detail::CiahProduct;
        static_assert(sizeof(Product) == 2 * sizeof(double));
        const int count = blas_int(requests.size());
        Buffer<Product> result(context, count);
        Buffer<const double*> left(context, count), right(context, count);
        Buffer<double*> output(context, count);
        std::vector<const double*> a(count), b(count);
        std::vector<double*> out(count);
        for (int i = 0; i < count; ++i) {
            a[i] = requests[i].first->data();
            b[i] = requests[i].second->data();
            out[i] = reinterpret_cast<double*>(result.data() + i);
        }
        context.copy_to_device(left.data(), a.data(), count * sizeof(double*));
        context.copy_to_device(right.data(), b.data(), count * sizeof(double*));
        context.copy_to_device(output.data(), out.data(), count * sizeof(double*));
        const double one = 1.0, zero = 0.0;
        check_cublas(cublasDgemmBatched(context.blas(),
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        1,
                                        1,
                                        blas_int(n),
                                        &one,
                                        left.data(),
                                        1,
                                        right.data(),
                                        blas_int(n),
                                        &zero,
                                        output.data(),
                                        1,
                                        count),
                     "CIAH batched projection dot products");
        absolute_products<<<count, 256, 0, context.stream()>>>(
            left.data(), right.data(), n, result.data());
        check_cuda(cudaGetLastError(), "CIAH batched absolute-product bounds");
        std::vector<Product> host(count);
        context.copy_to_host(host.data(), result.data(), count * sizeof(Product));
        return host;
    }
    double norm(const Vector& v) const {
        double value = 0.0;
        check_cublas(cublasDnrm2(context.blas(), blas_int(n), v.data(), 1, &value),
                     "CIAH cublasDnrm2");
        return value;
    }
    double stable_norm(const Vector& v) const {
        return norm(v);
    }
    bool has_joint_images() const noexcept {
        return true;
    }
    std::pair<Vector, Vector> apply_images(const Vector& v) const {
        Vector h(context, n, 1), g(context, n, 1);
        options.apply_images(v.view(), h.view(), g.view());
        return {std::move(h), std::move(g)};
    }
    Vector apply_metric(const Vector& v) const {
        if (!options.apply_metric)
            throw std::logic_error("CUDA CIAH inverse trace needs a device metric action");
        Vector g(context, n, 1);
        options.apply_metric(v.view(), g.view());
        return g;
    }
    Vector apply_hessian(const Vector& v) const {
        return apply_images(v).first;
    }
    Vector precondition(const Vector& r, double lambda) const {
        if (!options.residual_preconditioner)
            return clone(r);
        Vector out(context, n, 1);
        options.residual_preconditioner(r.view(), lambda, out.view());
        return out;
    }
    Vector residual_direction(const Vector& r) const {
        if (!options.residual_direction)
            return clone(r);
        Vector out(context, n, 1);
        options.residual_direction(r.view(), out.view());
        return out;
    }
    double coordinate_norm(const Vector& v, bool) const {
        return options.coordinate_norm ? options.coordinate_norm(v.view()) : norm(v);
    }
    void observe_images(const Vector& v, const Vector& image) const {
        if (options.observe_images)
            options.observe_images(v.view(), image.view());
    }
};

mcscf::detail::CiahNumerics numerics(const DeviceCiahOptions& options) {
    return {
        options.tolerance, options.lindep, options.max_cycle, options.operator_relative_accuracy};
}

} // namespace

class DeviceCiahWorkspace::Impl {
  public:
    Impl(Context& context,
         std::int64_t dimension,
         DeviceCiahOptions options,
         std::optional<linalg::ConstMatrixView> initial)
        : context_(context), dimension_(dimension),
          core(DeviceBackend(context, dimension, options),
               initial ? std::optional<Matrix<double>>(copy_vector(context, dimension, *initial))
                       : std::nullopt,
               numerics(options)) {}

    Context& context_;
    std::int64_t dimension_;
    mcscf::detail::CiahCore<DeviceBackend> core;
};

DeviceCiahWorkspace::DeviceCiahWorkspace(Context& context,
                                         std::int64_t dimension,
                                         DeviceCiahOptions options,
                                         std::optional<linalg::ConstMatrixView> initial)
    : impl_(std::make_unique<Impl>(context, dimension, std::move(options), initial)) {}

DeviceCiahWorkspace::~DeviceCiahWorkspace() = default;
void DeviceCiahWorkspace::reset_rhs() noexcept {
    impl_->core.reset_rhs();
}
int DeviceCiahWorkspace::basis_size() const noexcept {
    return impl_->core.basis_size();
}
Context& DeviceCiahWorkspace::context() const noexcept {
    return impl_->context_;
}
std::int64_t DeviceCiahWorkspace::dimension() const noexcept {
    return impl_->dimension_;
}

DeviceCiahResult DeviceCiahWorkspace::next(linalg::ConstMatrixView current_gradient) {
    Matrix<double> gradient = copy_vector(impl_->context_, impl_->dimension_, current_gradient);
    return impl_->core.next(gradient);
}

DeviceCiahResult DeviceCiahWorkspace::next(const Matrix<double>& gradient) {
    return impl_->core.next(gradient);
}

double DeviceCiahWorkspace::projected_dual_norm(linalg::ConstMatrixView covector) const {
    Matrix<double> value = copy_vector(impl_->context_, impl_->dimension_, covector);
    return impl_->core.projected_dual_norm(value);
}

double DeviceCiahWorkspace::projected_dual_norm(const Matrix<double>& covector) const {
    return impl_->core.projected_dual_norm(covector);
}

std::optional<DeviceCiahOriginRay>
DeviceCiahWorkspace::origin_ray(linalg::ConstMatrixView origin_gradient) const {
    Matrix<double> value = copy_vector(impl_->context_, impl_->dimension_, origin_gradient);
    return impl_->core.origin_ray(value);
}

std::optional<DeviceCiahOriginRay>
DeviceCiahWorkspace::origin_ray(const Matrix<double>& origin_gradient) const {
    return impl_->core.origin_ray(origin_gradient);
}

DeviceCiahVectorOps::Vector DeviceCiahVectorOps::clone(const Vector& v) const {
    return copy_vector(*context, v.rows(), v.view());
}

DeviceCiahVectorOps::Vector DeviceCiahVectorOps::zeros_like(const Vector& v) const {
    Vector out(*context, v.rows(), 1);
    check_cuda(
        cudaMemsetAsync(
            out.data(), 0, static_cast<std::size_t>(v.rows()) * sizeof(double), context->stream()),
        "CIAH vector zero");
    return out;
}

void DeviceCiahVectorOps::axpy(Vector& y, double factor, const Vector& x) const {
    check_cublas(
        cublasDaxpy(context->blas(), blas_int(y.rows()), &factor, x.data(), 1, y.data(), 1),
        "CIAH path cublasDaxpy");
}

void DeviceCiahVectorOps::scal(Vector& x, double factor) const {
    check_cublas(cublasDscal(context->blas(), blas_int(x.rows()), &factor, x.data(), 1),
                 "CIAH path cublasDscal");
}

double DeviceCiahVectorOps::dot(const Vector& x, const Vector& y) const {
    double result = 0.0;
    check_cublas(cublasDdot(context->blas(), blas_int(x.rows()), x.data(), 1, y.data(), 1, &result),
                 "CIAH path cublasDdot");
    return result;
}

double DeviceCiahVectorOps::norm(const Vector& x) const {
    return segment_norm(x, 0, x.rows());
}

double
DeviceCiahVectorOps::segment_norm(const Vector& x, std::int64_t begin, std::int64_t count) const {
    if (begin < 0 || count < 0 || begin > x.rows() || count > x.rows() - begin)
        throw std::invalid_argument("invalid CUDA CIAH segment");
    if (!count)
        return 0.0;
    double result = 0.0;
    check_cublas(cublasDnrm2(context->blas(), blas_int(count), x.data() + begin, 1, &result),
                 "CIAH path cublasDnrm2");
    return result;
}

bool DeviceCiahVectorOps::is_zero(const Vector& x) const {
    return norm(x) == 0.0;
}

DeviceCiahMicroResult::DeviceCiahMicroResult(Context& context,
                                             double energy,
                                             const Matrix<double>& initial_gradient)
    : displacement(DeviceCiahVectorOps{&context}.zeros_like(initial_gradient)),
      hessian_displacement(DeviceCiahVectorOps{&context}.zeros_like(initial_gradient)),
      predicted_gradient(DeviceCiahVectorOps{&context}.clone(initial_gradient)),
      last_step(DeviceCiahVectorOps{&context}.zeros_like(initial_gradient)),
      energy_model(energy, initial_gradient, DeviceCiahVectorOps{&context}),
      keyframes(energy, DeviceCiahVectorOps{&context}) {}

namespace {

struct DeviceDriverBackend : DeviceCiahVectorOps {
    const DeviceCiahMicroCallbacks& callbacks;
    std::int64_t n;

    DeviceCiahMicroResult make_result(double energy, const Vector& gradient) const {
        return DeviceCiahMicroResult(*context, energy, gradient);
    }
    DeviceCiahResult next(DeviceCiahWorkspace& workspace, const Vector& gradient) const {
        return workspace.next(gradient);
    }
    double projected_dual_norm(DeviceCiahWorkspace& workspace, const Vector& covector) const {
        return workspace.projected_dual_norm(covector);
    }
    double coordinate_norm(const Vector& v) const {
        return callbacks.coordinate_norm(v.view());
    }
    Vector metric_action(const Vector& v) const {
        Vector out(*context, n, 1);
        callbacks.metric_action(v.view(), out.view());
        return out;
    }
    double maximum_component(const Vector& v) const {
        return callbacks.maximum_component(v.view());
    }
    DeviceCiahKeyframe evaluate_keyframe(const Vector& displacement) const {
        DeviceCiahKeyframe point = callbacks.evaluate_keyframe(displacement.view());
        if (point.gradient.rows() != n || point.gradient.cols() != 1)
            throw std::runtime_error("invalid CUDA CIAH keyframe gradient");
        return point;
    }
    double orbital_residual_norm(const Vector& v) const {
        return segment_norm(v, 0, callbacks.orbital_size);
    }
    double structure_residual_norm(const Vector& v) const {
        return segment_norm(v, callbacks.orbital_size, n - callbacks.orbital_size);
    }
    double orbital_norm(const Vector& v) const {
        if (!callbacks.orbital_coordinate_norm)
            return orbital_residual_norm(v);
        const linalg::ConstMatrixView prefix{
            v.data(), callbacks.orbital_size, 1, callbacks.orbital_size};
        return callbacks.orbital_coordinate_norm(prefix);
    }
    void
    audit_step(const char* label, const Vector& displacement, const Vector& step, int index) const {
        if (callbacks.audit_step)
            callbacks.audit_step(label, displacement.view(), step.view(), index);
    }
    void audit_origin(const Vector& displacement,
                      const Vector& step,
                      const Vector& image,
                      const Vector& gradient) const {
        if (callbacks.audit_origin)
            callbacks.audit_origin(displacement.view(), step.view(), image.view(), gradient.view());
    }
    void audit_resumed(const Vector& displacement,
                       const Vector& step,
                       const Vector& image,
                       const Vector& gradient,
                       int basis_size) const {
        if (callbacks.audit_resumed)
            callbacks.audit_resumed(
                displacement.view(), step.view(), image.view(), gradient.view(), basis_size);
    }
    void audit_keyframe(const Vector& displacement,
                        const Vector& image,
                        const DeviceCiahKeyframe& keyframe) const {
        if (callbacks.audit_keyframe)
            callbacks.audit_keyframe(displacement.view(), image.view(), keyframe);
    }
    bool has_audit_resumed() const noexcept {
        return static_cast<bool>(callbacks.audit_resumed);
    }
};

} // namespace

DeviceCiahMicroResult run_device_ciah_microsteps(DeviceCiahWorkspace& workspace,
                                                 linalg::ConstMatrixView initial_gradient,
                                                 double energy,
                                                 double outer_gradient_tolerance,
                                                 const DeviceCiahMicroCallbacks& callbacks) {
    check_vector_view(initial_gradient, workspace.dimension());
    if (!callbacks.coordinate_norm || !callbacks.metric_action || !callbacks.maximum_component ||
        !callbacks.evaluate_keyframe || callbacks.orbital_size < 0 ||
        callbacks.orbital_size > workspace.dimension())
        throw std::invalid_argument("CUDA CIAH microstep callbacks are incomplete");
    Matrix<double> initial =
        copy_vector(workspace.context(), workspace.dimension(), initial_gradient);
    return mcscf::detail::run_ciah_microsteps_core(
        DeviceDriverBackend{{&workspace.context()}, callbacks, workspace.dimension()},
        workspace,
        initial,
        energy,
        outer_gradient_tolerance);
}

} // namespace xmvb::cuda
