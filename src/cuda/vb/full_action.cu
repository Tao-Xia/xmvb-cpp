#include "cuda/vb/full_action.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include "cuda/linalg/blas.hpp"
#include "cuda/vb/action.hpp"
#include "cuda/vb/product.hpp"
#include "cuda/vb/panel.hpp"
#include "cuda/vb/transport.hpp"
#include "cuda/vb/transform.hpp"

namespace xmvb::cuda {
namespace {

__global__ void identity_matrix(linalg::MatrixView matrix) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < matrix.rows * matrix.cols) {
        const int row = index % matrix.rows, column = index / matrix.rows;
        matrix.data[row + matrix.ld * column] = row == column ? 1.0 : 0.0;
    }
}

__global__ void divide_diagonal_column(linalg::MatrixView source,
                                       linalg::ConstMatrixView diagonal,
                                       linalg::ConstMatrixView energy,
                                       double shift) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < source.rows * source.cols) {
        const auto row = index % source.rows, column = index / source.rows;
        double denominator = diagonal.data[row + diagonal.ld * column] - energy.data[0] - shift;
        if (fabs(denominator) < 1.0e-8)
            denominator = 1.0e-8;
        source.data[row + source.ld * column] /= denominator;
    }
}

__global__ void beta_identity_columns(linalg::MatrixView values, int first) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < values.cols)
        values.data[first + column + values.ld * column] = 1.0;
}

void clear(Context& context, linalg::MatrixView image) {
    check_cuda(cudaMemset2DAsync(image.data,
                                 image.ld * sizeof(double),
                                 0,
                                 image.rows * sizeof(double),
                                 image.cols,
                                 context.stream()),
               "clear FULL selected product map");
}

/** Each group owns one raw spin-pair value; tiles never alias. */
template <bool Project>
__global__ void selected_tiles(const std::int64_t* offsets,
                               const std::int64_t* products,
                               const std::int64_t* structures,
                               const double* weights,
                               std::int64_t groups,
                               int alpha,
                               int extent,
                               const linalg::MatrixView* tiles,
                               linalg::ConstMatrixView source,
                               linalg::MatrixView image) {
    for (auto g = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < groups;
         g += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const auto a = products[g] % alpha, b = products[g] / alpha;
        const auto tile = tiles[b / extent];
        auto& value = tile.data[a + tile.ld * (b % extent)];
        if constexpr (Project) {
            for (auto t = offsets[g]; t < offsets[g + 1]; ++t)
                atomicAdd(image.data + structures[t], weights[t] * value);
        } else {
            double sum = 0.0;
            for (auto t = offsets[g]; t < offsets[g + 1]; ++t)
                sum += weights[t] * source.data[structures[t]];
            value = sum;
        }
    }
}

} // namespace

struct FullAction::Impl {
    Context& context;
    const StructureAction& action;
    StructureSourceFactors sources;
    ExteriorTransform identity_alpha, identity_beta;
    ProductContraction identity_product;

    Impl(Context& ctx, const StructureAction& source)
        : context(ctx), action(source), sources(ctx, source),
          identity_alpha(ctx, source.alpha_transform()),
          identity_beta(ctx, source.beta_transform()),
          identity_product(ctx,
                           source.orthogonal_view(),
                           identity_alpha,
                           identity_beta,
                           source.orthogonal_view().support) {
        const int active = identity_alpha.n_orbitals();
        if (active != identity_beta.n_orbitals() || sources.source_count() <= 0)
            throw std::invalid_argument("CUDA FULL action dimensions disagree");
        Matrix<double> orbital_identity(ctx, active, active);
        identity_matrix<<<(active * active + 255) / 256, 256, 0, ctx.stream()>>>(
            orbital_identity.view());
        check_cuda(cudaGetLastError(), "CUDA FULL identity frame");
        identity_alpha.update_upper(orbital_identity.view());
        identity_beta.update_upper(orbital_identity.view());
    }

    /** Same ordered maps as the former carrier implementation, on independent
   * panels. If the collection cannot fit, the exact source-factor schedule
   * below provides the same map with a smaller live working set.
   */
    bool apply_ordered(ProductMap map,
                       linalg::ConstMatrixView source,
                       linalg::MatrixView image,
                       linalg::ConstMatrixView energies,
                       double shift) const {
        const auto alpha = action.n_alpha_strings(), beta = action.n_beta_strings();
        const auto rows = static_cast<std::size_t>(alpha);
        if (rows > std::numeric_limits<std::size_t>::max() / sizeof(double) / beta)
            return false;
        const auto payload = rows * beta * sizeof(double);
        // Leave room for the transform stripe, diagonal panel and shared sigma
        // workspace. This changes residency only, not the physical operator.
        const int extent = std::min(product_panel_width(), std::max(1, beta - 1));
        const auto panels = (static_cast<std::size_t>(beta) + extent - 1) / extent;
        const auto metadata =
            panels * sizeof(linalg::MatrixView) + static_cast<std::size_t>(beta) * sizeof(double*);
        const auto allowance = context.available_device_bytes() / 4;
        if (metadata > allowance || payload > allowance - metadata)
            return false;
        std::vector<Matrix<double>> owners;
        std::vector<linalg::MatrixView> views;
        owners.reserve((beta + extent - 1) / extent);
        views.reserve(owners.capacity());
        for (int first = 0; first < beta; first += extent) {
            owners.emplace_back(context, alpha, std::min(extent, beta - first));
            views.push_back(owners.back().view());
        }
        Buffer<linalg::MatrixView> descriptors(context, views.size());
        context.copy_to_device(
            descriptors.data(), views.data(), views.size() * sizeof(views.front()));
        const auto groups = action.n_product_groups();
        const auto blocks =
            static_cast<unsigned>(std::min<std::int64_t>(65535, (groups + 255) / 256));
        const auto transform = [&](bool adjoint, bool inverse) {
            for (auto tile : views)
                action.alpha_transform().apply_left(tile, adjoint, inverse);
            action.beta_transform().apply_right_tiles(views, adjoint, inverse);
        };
        const bool divide = map == ProductMap::IdentityDiagonalInverse ||
                            map == ProductMap::InverseOrbitalDiagonalInverseAdjointOrbital;
        Matrix<double> diagonal;
        if (divide)
            diagonal = Matrix<double>(context, alpha, extent);
        for (std::int64_t rhs = 0; rhs < source.cols; ++rhs) {
            for (auto tile : views)
                clear(context, tile);
            selected_tiles<false>
                <<<blocks, 256, 0, context.stream()>>>(action.term_offsets_data(),
                                                       action.product_indices_data(),
                                                       action.term_structures_data(),
                                                       action.term_weights_data(),
                                                       groups,
                                                       alpha,
                                                       extent,
                                                       descriptors.data(),
                                                       source.column(rhs),
                                                       {});
            check_cuda(cudaGetLastError(), "FULL selected tile expansion");
            switch (map) {
                case ProductMap::Orbital:
                    transform(false, false);
                    break;
                case ProductMap::InverseOrbital:
                    transform(false, true);
                    break;
                case ProductMap::AdjointOrbital:
                    transform(true, false);
                    break;
                case ProductMap::InverseAdjointOrbital:
                    transform(true, true);
                    break;
                case ProductMap::InverseMetric:
                case ProductMap::InverseOrbitalDiagonalInverseAdjointOrbital:
                    transform(true, true);
                    break;
                case ProductMap::IdentityDiagonalInverse:
                    break;
                default:
                    throw std::invalid_argument("invalid ordered FULL map");
            }
            if (divide) {
                int first = 0;
                for (auto tile : views) {
                    auto values = diagonal.view().columns(0, tile.cols);
                    action.carrier_sigma().diagonal_columns(first, values);
                    const auto elements = tile.rows * tile.cols;
                    divide_diagonal_column<<<(elements + 255) / 256, 256, 0, context.stream()>>>(
                        tile, values, energies.column(rhs), shift);
                    check_cuda(cudaGetLastError(), "FULL ordered tile diagonal inverse");
                    first += tile.cols;
                }
            }
            if (map == ProductMap::InverseMetric ||
                map == ProductMap::InverseOrbitalDiagonalInverseAdjointOrbital)
                transform(false, true);
            clear(context, image.column(rhs));
            selected_tiles<true>
                <<<blocks, 256, 0, context.stream()>>>(action.term_offsets_data(),
                                                       action.product_indices_data(),
                                                       action.term_structures_data(),
                                                       action.term_weights_data(),
                                                       groups,
                                                       alpha,
                                                       extent,
                                                       descriptors.data(),
                                                       {},
                                                       image.column(rhs));
            check_cuda(cudaGetLastError(), "FULL selected tile projection");
        }
        return true;
    }

    void
    apply_orbital(ProductMap map, linalg::ConstMatrixView source, linalg::MatrixView image) const {
        const int extent = sources.source_extent();
        Matrix<double> alpha(context, action.n_alpha_strings(), extent);
        Matrix<double> beta(context, action.n_beta_strings(), extent);
        const bool adjoint =
            map == ProductMap::AdjointOrbital || map == ProductMap::InverseAdjointOrbital;
        const bool inverse =
            map == ProductMap::InverseOrbital || map == ProductMap::InverseAdjointOrbital;
        const auto& alpha_map = action.alpha_transform();
        const auto& beta_map = action.beta_transform();
        for (std::int64_t column = 0; column < source.cols; ++column) {
            auto target = image.column(column);
            clear(context, target);
            for (int first = 0; first < sources.source_count(); first += extent) {
                const int count = std::min(extent, sources.source_count() - first);
                auto a = alpha.view().columns(0, count), b = beta.view().columns(0, count);
                sources.load(first, count, source.column(column), a, b);
                alpha_map.apply_left(a, adjoint, inverse);
                beta_map.apply_left(b, adjoint, inverse);
                sources.contract(a, b, target);
            }
        }
    }

    void apply_diagonal(linalg::ConstMatrixView source,
                        linalg::MatrixView image,
                        linalg::ConstMatrixView energies,
                        double shift) const {
        const int extent = sources.source_extent();
        Matrix<double> alpha(context, action.n_alpha_strings(), extent);
        Matrix<double> beta(context, action.n_beta_strings(), extent);
        Matrix<double> diagonal(context, action.n_alpha_strings(), extent);
        for (std::int64_t column = 0; column < source.cols; ++column) {
            auto target = image.column(column);
            clear(context, target);
            for (int first = 0; first < sources.source_count();) {
                const int first_beta = sources.source_column(first);
                int count = 1;
                while (count < extent && first + count < sources.source_count() &&
                       sources.source_column(first + count) == first_beta + count)
                    ++count;
                auto a = alpha.view().columns(0, count);
                auto b = beta.view().columns(0, count);
                auto values = diagonal.view().columns(0, count);
                sources.load(first, count, source.column(column), a, b);
                action.carrier_sigma().diagonal_columns(first_beta, values);
                const auto elements = static_cast<std::int64_t>(a.rows) * count;
                divide_diagonal_column<<<(elements + 255) / 256, 256, 0, context.stream()>>>(
                    a, values, energies.column(column), shift);
                check_cuda(cudaGetLastError(), "CUDA FULL tiled diagonal inverse");
                sources.contract_raw_sources(first, a, target);
                first += count;
            }
        }
    }

    void apply_inverse_metric(linalg::ConstMatrixView source, linalg::MatrixView image) const {
        const int extent = sources.source_extent();
        Matrix<double> alpha(context, action.n_alpha_strings(), extent);
        Matrix<double> beta(context, action.n_beta_strings(), extent);
        for (std::int64_t column = 0; column < source.cols; ++column) {
            auto target = image.column(column);
            clear(context, target);
            for (int first = 0; first < sources.source_count(); first += extent) {
                const int count = std::min(extent, sources.source_count() - first);
                auto a = alpha.view().columns(0, count), b = beta.view().columns(0, count);
                sources.load(first, count, source.column(column), a, b);
                action.alpha_transform().apply_left(a, true, true);
                action.beta_transform().apply_left(b, true, true);
                action.alpha_transform().apply_left(a, false, true);
                action.beta_transform().apply_left(b, false, true);
                sources.contract(a, b, target);
            }
        }
    }

    void apply_compound(linalg::ConstMatrixView source,
                        linalg::MatrixView image,
                        linalg::ConstMatrixView energies,
                        double shift,
                        bool diagonal_inverse) const {
        const int extent = sources.source_extent();
        const int alpha_count = action.n_alpha_strings();
        const int beta_count = action.n_beta_strings();
        Matrix<double> source_alpha(context, alpha_count, extent);
        Matrix<double> source_beta(context, beta_count, extent);
        Matrix<double> target_alpha(context, alpha_count, extent);
        Matrix<double> target_beta(context, beta_count, extent);
        Matrix<double> diagonal;
        if (diagonal_inverse)
            diagonal = Matrix<double>(context, alpha_count, extent);
        for (std::int64_t rhs = 0; rhs < source.cols; ++rhs) {
            auto target = image.column(rhs);
            clear(context, target);
            for (int first_beta = 0; first_beta < beta_count; first_beta += extent) {
                const int target_count = std::min(extent, beta_count - first_beta);
                auto carrier = target_alpha.view().columns(0, target_count);
                clear(context, carrier);
                for (int first_source = 0; first_source < sources.source_count();
                     first_source += extent) {
                    const int source_count =
                        std::min(extent, sources.source_count() - first_source);
                    auto a = source_alpha.view().columns(0, source_count);
                    auto b = source_beta.view().columns(0, source_count);
                    sources.load(first_source, source_count, source.column(rhs), a, b);
                    action.alpha_transform().apply_left(a, true, true);
                    action.beta_transform().apply_left(b, true, true);
                    const linalg::ConstMatrixView beta_rows{
                        b.data + first_beta, target_count, source_count, b.ld};
                    gemm(context,
                         carrier,
                         a,
                         beta_rows,
                         1.0,
                         1.0,
                         Operation::none,
                         Operation::transpose);
                }
                if (diagonal_inverse) {
                    auto values = diagonal.view().columns(0, target_count);
                    action.carrier_sigma().diagonal_columns(first_beta, values);
                    const auto elements = static_cast<std::int64_t>(alpha_count) * target_count;
                    divide_diagonal_column<<<(elements + 255) / 256, 256, 0, context.stream()>>>(
                        carrier, values, energies.column(rhs), shift);
                    check_cuda(cudaGetLastError(), "CUDA FULL tiled compound diagonal inverse");
                }
                action.alpha_transform().apply_left(carrier, false, true);
                auto beta_basis = target_beta.view().columns(0, target_count);
                clear(context, beta_basis);
                beta_identity_columns<<<(target_count + 255) / 256, 256, 0, context.stream()>>>(
                    beta_basis, first_beta);
                check_cuda(cudaGetLastError(), "CUDA FULL beta identity panel");
                action.beta_transform().apply_left(beta_basis, false, true);
                sources.contract(carrier, beta_basis, target);
            }
        }
    }

    void apply_hamiltonian(linalg::ConstMatrixView source, linalg::MatrixView image) const {
        const ProductIncidence incidence{action.term_offsets_data(),
                                         action.term_structures_data(),
                                         action.term_weights_data(),
                                         action.size()};
        for (std::int64_t column = 0; column < source.cols; ++column)
            identity_product.apply_selected(incidence,
                                            source.column(column),
                                            action.carrier_sigma(),
                                            image.column(column),
                                            {},
                                            context.available_device_bytes());
    }
};

FullAction::FullAction(Context& context, const StructureAction& action)
    : impl_(std::make_unique<Impl>(context, action)) {}
FullAction::~FullAction() = default;

void FullAction::apply(ProductMap map,
                       linalg::ConstMatrixView source,
                       linalg::MatrixView image,
                       linalg::ConstMatrixView energies,
                       double eigenvalue) const {
    const auto& p = *impl_;
    if (source.rows != p.action.size() || source.cols <= 0 || source.ld < source.rows ||
        image.rows != source.rows || image.cols != source.cols || image.ld < image.rows ||
        !source.data || !image.data || source.data == image.data || !std::isfinite(eigenvalue))
        throw std::invalid_argument("CUDA FULL selected product map dimensions");
    const nvtx3::scoped_range range{"ci.full_product_map"};
    const bool diagonal = map == ProductMap::IdentityDiagonalInverse ||
                          map == ProductMap::InverseOrbitalDiagonalInverseAdjointOrbital;
    if (diagonal &&
        (energies.rows != 1 || energies.cols != source.cols || energies.ld < 1 || !energies.data))
        throw std::invalid_argument("CUDA FULL inverse energies dimensions");
    if (map != ProductMap::IdentityHamiltonian &&
        p.apply_ordered(map, source, image, energies, eigenvalue))
        return;
    switch (map) {
        case ProductMap::Orbital:
        case ProductMap::InverseOrbital:
        case ProductMap::AdjointOrbital:
        case ProductMap::InverseAdjointOrbital:
            p.apply_orbital(map, source, image);
            return;
        case ProductMap::IdentityHamiltonian:
            p.apply_hamiltonian(source, image);
            return;
        case ProductMap::IdentityDiagonalInverse:
            p.apply_diagonal(source, image, energies, eigenvalue);
            return;
        case ProductMap::InverseMetric:
            p.apply_inverse_metric(source, image);
            return;
        case ProductMap::InverseOrbitalDiagonalInverseAdjointOrbital:
            p.apply_compound(source, image, energies, eigenvalue, true);
            return;
    }
    throw std::invalid_argument("CUDA FULL product map type");
}

} // namespace xmvb::cuda
