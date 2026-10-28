#include <cstdint>
#include "cuda/optimization/coarse.hpp"
#include "linalg/range_inverse.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <cub/block/block_reduce.cuh>
#include <nvtx3/nvtx3.hpp>

#include <Eigen/LU>
#include <Eigen/SVD>

#include "cuda/linalg/blas.hpp"
#include "mcscf/optimization/trace.hpp"

namespace xmvb::cuda {
namespace {

__global__ void finite_vectors(linalg::ConstMatrixView first,
                               linalg::ConstMatrixView second,
                               linalg::ConstMatrixView third,
                               int* valid,
                               std::int64_t dimension) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= dimension)
        return;
    if (!isfinite(first.data[i]) || (second.data && !isfinite(second.data[i])) ||
        (third.data && !isfinite(third.data[i])))
        atomicExch(valid, 0);
}

struct CompactView {
    const double* basis;
    const double* metric;
    std::int64_t begin, length;
};

__global__ void projection_magnitudes(const CompactView* columns,
                                      linalg::ConstMatrixView images,
                                      bool metric,
                                      double* result,
                                      int rank) {
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    const int row = blockIdx.x % rank, column = blockIdx.x / rank;
    const auto left = columns[row], right = columns[column];
    const int first = metric ? max(left.begin, right.begin) : left.begin;
    const int last = metric ? min(left.begin + left.length, right.begin + right.length)
                            : left.begin + left.length;
    double sum = 0.0;
    for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
        const double action =
            metric ? right.metric[i - right.begin] : images.data[column * images.ld + i];
        sum += fabs(left.basis[i - left.begin] * action);
    }
    const double total = Reduce(scratch).Sum(sum);
    if (!threadIdx.x)
        result[blockIdx.x] = total;
}

// The following AXPYs consume this coefficient on the same stream. MGS is
// still sequential, including its second pass; no host scalar is needed.
__global__ void accumulate_coefficient(double* coefficient, double* total) {
    *total += *coefficient;
    *coefficient = -*coefficient;
}

class DeviceScalars {
  public:
    explicit DeviceScalars(cublasHandle_t handle) : handle_(handle) {
        check_cublas(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE),
                     "CUDA coarse device scalar pointer mode");
    }
    ~DeviceScalars() {
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    }
    DeviceScalars(const DeviceScalars&) = delete;
    DeviceScalars& operator=(const DeviceScalars&) = delete;

  private:
    cublasHandle_t handle_;
};

std::size_t reduction_size(int capacity) {
    const auto n = static_cast<std::size_t>(std::max<std::int64_t>(0, capacity));
    return std::max({n * n, 5 * n + 4, n + 3});
}

unsigned int grid(std::int64_t count) {
    constexpr std::int64_t threads = 256;
    const auto blocks = (count + threads - 1) / threads;
    if (blocks <= 0 || blocks > std::numeric_limits<unsigned int>::max())
        throw std::overflow_error("CUDA coarse grid overflow");
    return static_cast<unsigned int>(blocks);
}

} // namespace

DeviceCoarsePreconditioner::DeviceCoarsePreconditioner(Context& context,
                                                       std::int64_t dimension,
                                                       int capacity,
                                                       double lindep,
                                                       double action_accuracy,
                                                       std::vector<SupportRange> supports)
    : context_(&context), dimension_(dimension),
      capacity_(std::min<std::int64_t>(dimension, capacity)), lindep_(lindep),
      action_accuracy_(action_accuracy),
      images_(context, dimension, std::max<std::int64_t>(0, capacity_)),
      sample_v_(context, dimension, 1), sample_h_(context, dimension, 1),
      sample_m_(context, dimension, 1),
      coefficients_(context, std::max<std::int64_t>(1, capacity_), 1),
      reductions_(context, reduction_size(capacity_)), finite_(context, 1),
      hessian_projection_(Eigen::MatrixXd::Zero(std::max<std::int64_t>(0, capacity_),
                                                std::max<std::int64_t>(0, capacity_))),
      metric_projection_(Eigen::MatrixXd::Zero(std::max<std::int64_t>(0, capacity_),
                                               std::max<std::int64_t>(0, capacity_))),
      sample_factor_(Eigen::MatrixXd::Zero(std::max<std::int64_t>(0, capacity_),
                                           std::max<std::int64_t>(0, capacity_))) {
    if (dimension <= 0 || capacity < 0 || !(lindep > 0.0) || !std::isfinite(action_accuracy) ||
        action_accuracy < 0.0)
        throw std::invalid_argument("invalid CUDA coarse preconditioner dimensions");
    if (supports.empty())
        supports.push_back({0, dimension_, capacity_});
    for (int i = 0; i < supports.size(); ++i) {
        const auto& support = supports[i];
        if (support.begin < 0 || support.begin >= dimension_ || support.length <= 0 ||
            support.length > dimension_ - support.begin || support.capacity < 0)
            throw std::invalid_argument("invalid CUDA coarse support contract");
        for (int j = 0; j < i; ++j) {
            const auto& prior = supports[j];
            if (support.begin < prior.begin + prior.length &&
                prior.begin < support.begin + support.length)
                throw std::invalid_argument("overlapping CUDA coarse support contracts");
        }
        panels_.push_back(
            {support.begin, support.length, 0, std::min(support.capacity, capacity_), {}, {}});
    }
    check_cublas(cublasSetPointerMode(context.blas(), CUBLAS_POINTER_MODE_HOST),
                 "CUDA coarse host scalar pointer mode");
}

void DeviceCoarsePreconditioner::validate_vector(linalg::ConstMatrixView vector) const {
    if (vector.rows != dimension_ || vector.cols != 1 || vector.ld < vector.rows || !vector.data)
        throw std::invalid_argument("invalid CUDA coarse vector dimensions");
}

void DeviceCoarsePreconditioner::ensure_finite(linalg::ConstMatrixView first,
                                               linalg::ConstMatrixView second,
                                               linalg::ConstMatrixView third) const {
    validate_vector(first);
    if (second.data)
        validate_vector(second);
    if (third.data)
        validate_vector(third);
    check_cuda(cudaMemsetAsync(finite_.data(), 1, sizeof(int), context_->stream()),
               "CUDA coarse finite flag initialization");
    finite_vectors<<<grid(dimension_), 256, 0, context_->stream()>>>(
        first, second, third, finite_.data(), dimension_);
    check_cuda(cudaGetLastError(), "CUDA coarse finite validation");
    int host = 0;
    context_->copy_to_host(&host, finite_.data(), sizeof(host));
    if (!host)
        throw std::invalid_argument("nonfinite CUDA coarse sample");
}

const double* DeviceCoarsePreconditioner::basis_data(const CompactColumn& column) const {
    const auto& panel = panels_[column.panel];
    return panel.basis.data() + static_cast<std::size_t>(column.slot) * panel.length;
}

const double* DeviceCoarsePreconditioner::metric_data(const CompactColumn& column) const {
    const auto& panel = panels_[column.panel];
    return panel.metric.data() + static_cast<std::size_t>(column.slot) * panel.length;
}

linalg::ConstMatrixView
DeviceCoarsePreconditioner::panel_view(const CompactPanel& panel, bool metric, int used) const {
    return {metric ? panel.metric.data() : panel.basis.data(), panel.length, used, panel.length};
}

Eigen::VectorXd DeviceCoarsePreconditioner::project_basis(linalg::ConstMatrixView vector) const {
    Eigen::VectorXd result(size_);
    if (!size_)
        return result;
    int offset = 0;
    for (const auto& panel : panels_) {
        if (!panel.used)
            continue;
        gemm(*context_,
             {coefficients_.data() + offset, panel.used, 1, panel.used},
             panel_view(panel, false, panel.used),
             {vector.data + panel.begin, panel.length, 1, panel.length},
             1.0,
             0.0,
             Operation::transpose);
        offset += panel.used;
    }
    Eigen::VectorXd packed(size_);
    context_->copy_to_host(packed.data(), coefficients_.data(), size_ * sizeof(double));
    std::vector<int> starts(panels_.size());
    offset = 0;
    for (int p = 0; p < panels_.size(); ++p) {
        starts[p] = offset;
        offset += panels_[p].used;
    }
    for (int i = 0; i < size_; ++i)
        result[i] = packed[starts[columns_[i].panel] + columns_[i].slot];
    return result;
}

void DeviceCoarsePreconditioner::combine(linalg::MatrixView output,
                                         linalg::ConstMatrixView panel,
                                         const Eigen::VectorXd& coefficients,
                                         double scale) const {
    const int columns = static_cast<int>(coefficients.size());
    if (!columns)
        return;
    context_->copy_to_device(coefficients_.data(), coefficients.data(), columns * sizeof(double));
    gemm(*context_,
         output,
         panel.columns(0, columns),
         {coefficients_.data(), columns, 1, columns},
         scale,
         1.0);
}

void DeviceCoarsePreconditioner::combine_compact(linalg::MatrixView output,
                                                 const Eigen::VectorXd& coefficients,
                                                 bool metric,
                                                 double scale) const {
    Eigen::VectorXd packed(coefficients.size());
    std::vector<int> starts(panels_.size());
    int offset = 0;
    for (int p = 0; p < panels_.size(); ++p) {
        starts[p] = offset;
        offset += panels_[p].used;
    }
    for (int i = 0; i < coefficients.size(); ++i)
        packed[starts[columns_[i].panel] + columns_[i].slot] = coefficients[i];
    context_->copy_to_device(coefficients_.data(), packed.data(), packed.size() * sizeof(double));
    for (int p = 0; p < panels_.size(); ++p) {
        const auto& panel = panels_[p];
        if (!panel.used)
            continue;
        gemm(*context_,
             {output.data + panel.begin, panel.length, 1, panel.length},
             panel_view(panel, metric, panel.used),
             {coefficients_.data() + starts[p], panel.used, 1, panel.used},
             scale,
             1.0);
    }
}

void DeviceCoarsePreconditioner::copy(linalg::MatrixView out, linalg::ConstMatrixView in) const {
    check_cuda(cudaMemcpyAsync(out.data,
                               in.data,
                               dimension_ * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               context_->stream()),
               "CUDA coarse vector copy");
}

void DeviceCoarsePreconditioner::scale(linalg::MatrixView out, double coefficient) const {
    check_cublas(cublasDscal_64(context_->blas(), dimension_, &coefficient, out.data, 1),
                 "CUDA coarse scale");
}

double DeviceCoarsePreconditioner::projection_roundoff(bool metric,
                                                       const Eigen::MatrixXd& projected) const {
    const int rank = static_cast<int>(projected.rows());
    Eigen::MatrixXd magnitude(rank, rank);
    std::vector<CompactView> host(rank);
    for (int i = 0; i < rank; ++i) {
        const auto& column = columns_[i];
        host[i] = {basis_data(column), metric_data(column), column.begin, column.length};
    }
    Buffer<CompactView> device(*context_, rank);
    context_->copy_to_device(device.data(), host.data(), host.size() * sizeof(CompactView));
    projection_magnitudes<<<rank * rank, 256, 0, context_->stream()>>>(
        device.data(), images_.view(), metric, reductions_.data(), rank);
    check_cuda(cudaGetLastError(), "CUDA coarse projection magnitudes");
    context_->copy_to_host(magnitude.data(), reductions_.data(), magnitude.size() * sizeof(double));
    const double u = 0.5 * std::numeric_limits<double>::epsilon();
    const double nu = static_cast<double>(dimension_) * u;
    const double gamma = nu / (1.0 - nu);
    const Eigen::MatrixXd error = (gamma / (1.0 - gamma)) * (magnitude + magnitude.transpose()) +
                                  u * (projected.cwiseAbs() + projected.transpose().cwiseAbs());
    return error.stableNorm();
}

void DeviceCoarsePreconditioner::observe(linalg::ConstMatrixView direction,
                                         linalg::ConstMatrixView image,
                                         linalg::ConstMatrixView metric_image) {
    validate_vector(direction);
    validate_vector(metric_image);
    observe_range(direction, image, metric_image, 0);
}

void DeviceCoarsePreconditioner::observe_range(linalg::ConstMatrixView direction,
                                               linalg::ConstMatrixView image,
                                               linalg::ConstMatrixView metric_image,
                                               std::int64_t begin) {
    const nvtx3::scoped_range range{"coarse.observe"};
    if (begin < 0 || begin >= dimension_ || direction.rows <= 0 ||
        direction.rows > dimension_ - begin || direction.cols != 1 ||
        direction.ld < direction.rows || !direction.data || metric_image.rows != direction.rows ||
        metric_image.cols != 1 || metric_image.ld < metric_image.rows || !metric_image.data)
        throw std::invalid_argument("invalid CUDA coarse support range");
    const int length = static_cast<int>(direction.rows);
    validate_vector(image);
    int panel_index = -1;
    for (int p = 0; p < panels_.size(); ++p)
        if (panels_[p].begin == begin && panels_[p].length == length) {
            panel_index = p;
            break;
        }
    if (panel_index < 0)
        throw std::invalid_argument("undeclared CUDA coarse support range");
    check_cuda(cudaMemsetAsync(finite_.data(), 1, sizeof(int), context_->stream()),
               "CUDA coarse finite flag initialization");
    finite_vectors<<<grid(length), 256, 0, context_->stream()>>>(
        direction, {}, metric_image, finite_.data(), length);
    finite_vectors<<<grid(dimension_), 256, 0, context_->stream()>>>(
        image, {}, {}, finite_.data(), dimension_);
    check_cuda(cudaGetLastError(), "CUDA coarse finite validation");
    int valid = 0;
    context_->copy_to_host(&valid, finite_.data(), sizeof(valid));
    if (!valid)
        throw std::invalid_argument("nonfinite CUDA coarse sample");
    if (size_ == capacity_)
        return;
    check_cublas(cublasSetPointerMode(context_->blas(), CUBLAS_POINTER_MODE_HOST),
                 "CUDA coarse observe scalar pointer mode");
    check_cuda(cudaMemcpyAsync(sample_v_.data() + begin,
                               direction.data,
                               length * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               context_->stream()),
               "CUDA coarse compact direction copy");
    copy(sample_h_.view(), image);
    check_cuda(cudaMemcpyAsync(sample_m_.data() + begin,
                               metric_image.data,
                               length * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               context_->stream()),
               "CUDA coarse compact metric copy");
    double* const scalars = reductions_.data();
    check_cuda(cudaMemsetAsync(scalars, 0, size_ * sizeof(double), context_->stream()),
               "CUDA coarse coefficient initialization");
    {
        DeviceScalars mode(context_->blas());
        check_cublas(
            cublasDnrm2_64(context_->blas(), length, sample_v_.data() + begin, 1, scalars + size_),
            "CUDA coarse initial norm");
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < size_; ++i) {
                const auto& column = columns_[i];
                if (column.begin != begin || column.length != length)
                    continue;
                double* const coefficient = scalars + size_ + 2;
                check_cublas(cublasDdot_64(context_->blas(),
                                           length,
                                           basis_data(column),
                                           1,
                                           sample_v_.data() + begin,
                                           1,
                                           coefficient),
                             "CUDA coarse device dot");
                accumulate_coefficient<<<1, 1, 0, context_->stream()>>>(coefficient, scalars + i);
                check_cuda(cudaGetLastError(), "CUDA coarse coefficient accumulation");
                check_cublas(cublasDaxpy_64(context_->blas(),
                                            length,
                                            coefficient,
                                            basis_data(column),
                                            1,
                                            sample_v_.data() + begin,
                                            1),
                             "CUDA coarse compact basis axpy");
                check_cublas(cublasDaxpy_64(context_->blas(),
                                            dimension_,
                                            coefficient,
                                            images_.view().column(i).data,
                                            1,
                                            sample_h_.data(),
                                            1),
                             "CUDA coarse image axpy");
                check_cublas(cublasDaxpy_64(context_->blas(),
                                            length,
                                            coefficient,
                                            metric_data(column),
                                            1,
                                            sample_m_.data() + begin,
                                            1),
                             "CUDA coarse compact metric axpy");
            }
        check_cublas(
            cublasDnrm2_64(
                context_->blas(), length, sample_v_.data() + begin, 1, scalars + size_ + 1),
            "CUDA coarse remaining norm");
    }
    Eigen::VectorXd samples(size_ + 2);
    context_->copy_to_host(samples.data(), scalars, samples.size() * sizeof(double));
    const double initial = samples[size_], remaining = samples[size_ + 1];
    if (!(initial > 0.0))
        return;
    Eigen::MatrixXd factor = Eigen::MatrixXd::Zero(size_ + 1, size_ + 1);
    factor.topLeftCorner(size_, size_) = sample_factor_.topLeftCorner(size_, size_);
    factor.col(size_).head(size_) = samples.head(size_) / initial;
    factor(size_, size_) = remaining / initial;
    const Eigen::JacobiSVD<Eigen::MatrixXd> spectrum(factor);
    const double ratio =
        std::pow(spectrum.singularValues().tail(1)[0] / spectrum.singularValues()[0], 2);
    if (!(ratio > lindep_))
        return;
    auto& panel = panels_[panel_index];
    if (panel.used == panel.capacity)
        throw std::runtime_error("CUDA coarse support capacity exceeded");
    const int candidate = size_;
    copy(images_.view().column(candidate), sample_h_.view());
    const double inverse = 1.0 / remaining;
    scale(images_.view().column(candidate), inverse);
    const bool new_panel = panel.used == 0;
    if (new_panel) {
        Buffer<double> basis_buffer(*context_, static_cast<std::size_t>(length) * panel.capacity);
        Buffer<double> metric_buffer(*context_, static_cast<std::size_t>(length) * panel.capacity);
        panel.basis = std::move(basis_buffer);
        panel.metric = std::move(metric_buffer);
    }
    const int slot = panel.used;
    const auto rollback = [&] {
        if (static_cast<int>(columns_.size()) > candidate) {
            columns_.pop_back();
            --panel.used;
        }
        if (new_panel) {
            panel.basis = Buffer<double>();
            panel.metric = Buffer<double>();
        }
    };
    try {
        double* const basis = panel.basis.data() + static_cast<std::size_t>(slot) * length;
        double* const metric = panel.metric.data() + static_cast<std::size_t>(slot) * length;
        check_cuda(cudaMemcpyAsync(basis,
                                   sample_v_.data() + begin,
                                   length * sizeof(double),
                                   cudaMemcpyDeviceToDevice,
                                   context_->stream()),
                   "CUDA coarse compact basis copy");
        check_cuda(cudaMemcpyAsync(metric,
                                   sample_m_.data() + begin,
                                   length * sizeof(double),
                                   cudaMemcpyDeviceToDevice,
                                   context_->stream()),
                   "CUDA coarse compact metric copy");
        check_cublas(cublasDscal_64(context_->blas(), length, &inverse, basis, 1),
                     "CUDA coarse compact basis scale");
        check_cublas(cublasDscal_64(context_->blas(), length, &inverse, metric, 1),
                     "CUDA coarse compact metric scale");
        columns_.push_back({begin, length, panel_index, slot});
        ++panel.used;
        int offset = 0;
        std::vector<int> hcol(panels_.size(), -1), mcol(panels_.size(), -1),
            mrow(panels_.size(), -1);
        for (int p = 0; p < panels_.size(); ++p) {
            const auto& current = panels_[p];
            if (!current.used)
                continue;
            hcol[p] = offset;
            gemm(*context_,
                 {scalars + offset, current.used, 1, current.used},
                 panel_view(current, false, current.used),
                 {images_.view().column(candidate).data + current.begin,
                  current.length,
                  1,
                  current.length},
                 1.0,
                 0.0,
                 Operation::transpose);
            offset += current.used;
        }
        const int hrow = offset;
        if (candidate)
            gemm(*context_,
                 {scalars + offset, candidate, 1, candidate},
                 {images_.data() + begin, length, candidate, dimension_},
                 {basis, length, 1, length},
                 1.0,
                 0.0,
                 Operation::transpose);
        offset += candidate;
        for (int p = 0; p < panels_.size(); ++p) {
            const auto& current = panels_[p];
            if (!current.used || current.begin != begin || current.length != length)
                continue;
            mcol[p] = offset;
            gemm(*context_,
                 {scalars + offset, current.used, 1, current.used},
                 panel_view(current, false, current.used),
                 {metric, length, 1, length},
                 1.0,
                 0.0,
                 Operation::transpose);
            offset += current.used;
        }
        for (int p = 0; p < panels_.size(); ++p) {
            const auto& current = panels_[p];
            if (!current.used || current.begin != begin || current.length != length)
                continue;
            const int count = current.used - (p == panel_index ? 1 : 0);
            if (!count)
                continue;
            mrow[p] = offset;
            gemm(*context_,
                 {scalars + offset, count, 1, count},
                 panel_view(current, true, count),
                 {basis, length, 1, length},
                 1.0,
                 0.0,
                 Operation::transpose);
            offset += count;
        }
        const int norm_begin = offset;
        {
            DeviceScalars mode(context_->blas());
            if (action_accuracy_ > 0.0) {
                for (int p = 0; p < panels_.size(); ++p) {
                    const auto& current = panels_[p];
                    if (!current.used)
                        continue;
                    check_cublas(
                        cublasDnrm2_64(context_->blas(),
                                       (static_cast<std::int64_t>(current.length) * current.used),
                                       current.basis.data(),
                                       1,
                                       scalars + offset++),
                        "CUDA coarse compact panel norm");
                }
                const std::int64_t count =
                    (static_cast<std::int64_t>(dimension_) * (candidate + 1));
                check_cublas(
                    cublasDnrm2_64(context_->blas(), count, images_.data(), 1, scalars + offset),
                    "CUDA coarse image norm");
                ++offset;
            }
        }
        Eigen::VectorXd projections(offset);
        context_->copy_to_host(projections.data(), scalars, projections.size() * sizeof(double));
        for (int i = 0; i <= candidate; ++i) {
            const auto& column = columns_[i];
            hessian_projection_(i, candidate) = projections[hcol[column.panel] + column.slot];
            metric_projection_(i, candidate) =
                mcol[column.panel] >= 0 ? projections[mcol[column.panel] + column.slot] : 0.0;
            if (i == candidate)
                continue;
            hessian_projection_(candidate, i) = projections[hrow + i];
            metric_projection_(candidate, i) =
                mrow[column.panel] >= 0 ? projections[mrow[column.panel] + column.slot] : 0.0;
        }
        double basis_norm = 0.0;
        if (action_accuracy_ > 0.0)
            basis_norm = projections.segment(norm_begin, offset - norm_begin - 1).stableNorm();
        const auto certified =
            [&](const Eigen::MatrixXd& projection, bool metric, double action_accuracy) {
                const Eigen::MatrixXd raw = projection.topLeftCorner(candidate + 1, candidate + 1);
                const double skew = (raw - raw.transpose()).norm();
                double bound = 256 * std::numeric_limits<double>::epsilon() * (candidate + 1) *
                               std::max(1.0, raw.norm());
                if (action_accuracy > 0.0)
                    bound += action_accuracy * basis_norm * projections[offset - 1];
                if (skew > bound)
                    bound += projection_roundoff(metric, raw);
                return std::isfinite(skew) && std::isfinite(bound) && skew <= bound;
            };
        if (!certified(hessian_projection_, false, action_accuracy_) ||
            !certified(metric_projection_, true, 0.0)) {
            rollback();
            return;
        }
        for (int i = 0; i < candidate; ++i) {
            const double hessian =
                0.5 * (hessian_projection_(i, candidate) + hessian_projection_(candidate, i));
            const double metric =
                0.5 * (metric_projection_(i, candidate) + metric_projection_(candidate, i));
            hessian_projection_(i, candidate) = hessian_projection_(candidate, i) = hessian;
            metric_projection_(i, candidate) = metric_projection_(candidate, i) = metric;
        }
        sample_factor_.topLeftCorner(size_ + 1, size_ + 1) = factor;
        ++size_;
    } catch (...) {
        rollback();
        throw;
    }
}

void DeviceCoarsePreconditioner::apply(linalg::ConstMatrixView residual,
                                       double eigenvalue,
                                       const BaseInverse& inverse,
                                       linalg::MatrixView output) const {
    const nvtx3::scoped_range range{"coarse.apply"};
    validate_vector(residual);
    validate_vector(output);
    if (!std::isfinite(eigenvalue) || !inverse)
        throw std::invalid_argument("invalid CUDA coarse inverse request");
    ensure_finite(residual);
    check_cublas(cublasSetPointerMode(context_->blas(), CUBLAS_POINTER_MODE_HOST),
                 "CUDA coarse apply scalar pointer mode");
    if (!size_) {
        copy(sample_v_.view(), residual);
        inverse(sample_v_.view(), sample_h_.view());
        check_cublas(cublasSetPointerMode(context_->blas(), CUBLAS_POINTER_MODE_HOST),
                     "CUDA coarse restore scalar pointer mode");
        ensure_finite(sample_h_.view());
        copy(output, sample_h_.view());
        return;
    }
    const linalg::RangeInverse factor(hessian_projection_.topLeftCorner(size_, size_) -
                                      eigenvalue * metric_projection_.topLeftCorner(size_, size_));
    if (factor.rank() != size_)
        mcscf::diagnostics::ciah_trace("coarse_shift_range",
                                       "size=%d,rank=%ld,shift=%.17g",
                                       size_,
                                       static_cast<long>(factor.rank()),
                                       eigenvalue);
    Eigen::VectorXd projected = project_basis(residual);
    const Eigen::VectorXd a = factor.solve(projected);
    copy(sample_v_.view(), residual);
    combine(sample_v_.view(), images_.view(), a, -1.0);
    combine_compact(sample_v_.view(), a, true, eigenvalue);
    inverse(sample_v_.view(), sample_h_.view());
    check_cublas(cublasSetPointerMode(context_->blas(), CUBLAS_POINTER_MODE_HOST),
                 "CUDA coarse restore scalar pointer mode");
    ensure_finite(sample_h_.view());
    gemm(*context_,
         {reductions_.data(), size_, 1, size_},
         images_.view().columns(0, size_),
         sample_h_.view(),
         1.0,
         0.0,
         Operation::transpose);
    std::vector<int> starts(panels_.size());
    int offset = size_;
    for (int p = 0; p < panels_.size(); ++p) {
        const auto& panel = panels_[p];
        starts[p] = offset;
        if (!panel.used)
            continue;
        gemm(*context_,
             {reductions_.data() + offset, panel.used, 1, panel.used},
             panel_view(panel, true, panel.used),
             {sample_h_.data() + panel.begin, panel.length, 1, panel.length},
             1.0,
             0.0,
             Operation::transpose);
        offset += panel.used;
    }
    Eigen::VectorXd packed(offset);
    context_->copy_to_host(packed.data(), reductions_.data(), offset * sizeof(double));
    for (int i = 0; i < size_; ++i)
        projected[i] =
            packed[i] - eigenvalue * packed[starts[columns_[i].panel] + columns_[i].slot];
    const Eigen::VectorXd b = factor.solve(projected);
    copy(output, sample_h_.view());
    combine_compact(output, a - b, false);
}

} // namespace xmvb::cuda
