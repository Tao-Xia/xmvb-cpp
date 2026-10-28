#include <cmath>
#include <array>
#include <numeric>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include "core/pair_index.hpp"
#include "cuda/vb/operator_first.hpp"
#include "cuda/ci/raw.hpp"
#include "cuda/ci/similarity.hpp"
#include "mcscf/ci/raw_action.hpp"
#include "mcscf/ci/sigma.hpp"
#include "cuda/vb/action.hpp"
#include "vbscf/structures/assembly/operator_first.hpp"
#include "vbscf/structures/orthogonal_ci/integrals.hpp"
#include "vbscf/determinants/pairs/same_spin_cache.hpp"
namespace {
using Matrix = Eigen::MatrixXd;
using Device = xmvb::cuda::Matrix<double>;
using Strings = std::vector<std::vector<int>>;
void near(const Matrix& a, const Matrix& b, const char* label, double tol = 3e-10) {
    if (a.rows() != b.rows() || a.cols() != b.cols() || !a.allFinite() ||
        (a - b).norm() > tol * std::max(1.0, b.norm())) {
        std::cerr << label << "\nactual:\n" << a << "\nexpected:\n" << b << '\n';
        throw std::runtime_error(label);
    }
}
Device upload(xmvb::cuda::Context& c, const Matrix& a) {
    Device d(c, a.rows(), a.cols());
    xmvb::cuda::copy_to_device(c, d.view(), a);
    return d;
}
Matrix download(xmvb::cuda::Context& c, xmvb::linalg::ConstMatrixView a) {
    Matrix out(a.rows, a.cols);
    xmvb::cuda::copy_to_host(c, out, a);
    return out;
}
Matrix upper(Matrix x) {
    x.triangularView<Eigen::StrictlyLower>().setZero();
    return x;
}
/** Repeated beta columns make every reduction partition nonzero in a small case. */
void check_reduction(xmvb::cuda::Context& c,
                     const xmvb::vb::StructureAction& action,
                     const xmvb::vb::OrthogonalActiveIntegrals& integrals,
                     const Matrix& coefficients) {
    const auto& v = action.orthogonal_view();
    const int n = v.alpha.n_orbitals(), na = v.alpha.dimension(), nb = v.beta.dimension();
    constexpr int count = 128;
    std::vector<std::int64_t> alpha(v.products.size()), ids(na), columns(count);
    std::vector<std::int64_t> beta(v.products.size());
    Matrix x = Matrix::Zero(v.products.size(), 1);
    for (std::size_t p = 0; p < v.products.size(); ++p) {
        alpha[p] = v.products[p] / nb;
        beta[p] = v.products[p] % nb;
        for (auto t = v.offsets[p]; t < v.offsets[p + 1]; ++t)
            x(p, 0) += v.terms[t].coefficient * coefficients(v.terms[t].structure, 0);
    }
    std::iota(ids.begin(), ids.end(), 0);
    const auto& masks = v.alpha.occupations();
    std::sort(ids.begin(), ids.end(), [&](int a, int b) { return masks[a] < masks[b]; });
    std::vector<std::uint64_t> sorted(na);
    for (int a = 0; a < na; ++a)
        sorted[a] = masks[ids[a]];
    for (int j = 0; j < count; ++j)
        columns[j] = j % nb;
    const auto upload_vector = [&](const auto& values) {
        using T = typename std::decay_t<decltype(values)>::value_type;
        xmvb::cuda::Buffer<T> buffer(c, values.size());
        c.copy_to_device(buffer.data(), values.data(), values.size() * sizeof(T));
        return buffer;
    };
    auto a = upload_vector(alpha), indices = upload_vector(ids), cols = upload_vector(columns);
    auto b = upload_vector(beta);
    auto am = upload_vector(v.alpha.occupations()), bm = upload_vector(v.beta.occupations()),
         sm = upload_vector(sorted);
    int device, processors;
    xmvb::cuda::check_cuda(cudaGetDevice(&device), "test CUDA device");
    xmvb::cuda::check_cuda(
        cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device),
        "test CUDA multiprocessors");
    xmvb::mcscf::RawCiPlan plan(
        v.sigma.alpha_connections(), v.sigma.beta_connections_view(), v.products);
    const xmvb::cuda::RawCiTopology topology{a.data(),
                                             b.data(),
                                             static_cast<std::int64_t>(v.products.size()),
                                             am.data(),
                                             bm.data(),
                                             sm.data(),
                                             indices.data(),
                                             na,
                                             n,
                                             v.sigma.alpha_connections().n_occupied(),
                                             v.sigma.beta_connections_view().n_occupied(),
                                             processors,
                                             plan.complete_covalent_support()};
    Matrix seed(na, count), dseed(na, count), dsource = 0.013 * x.array().cos().matrix();
    for (int j = 0; j < count; ++j)
        for (int i = 0; i < na; ++i) {
            seed(i, j) = 0.021 * std::sin(i + 0.17 * j);
            dseed(i, j) = 0.017 * std::cos(0.13 * i - j);
        }
    const xmvb::mcscf::ActiveIntegrals physical{integrals.one_electron, integrals.pair_kernel};
    const xmvb::mcscf::SimilarityHamiltonian h(physical, integrals.orbital_transform);
    const Matrix zero = Matrix::Zero(n, n);
    const xmvb::mcscf::ActiveIntegrals delta{0.02 * physical.one_electron,
                                             -0.01 * physical.pair_kernel};
    const auto dh = h.direction(delta, zero);
    Matrix images = Matrix::Zero(na, count), image0 = images, image1 = images;
    std::array<Matrix, 3> ones, twos;
    for (int l = 0; l < 3; ++l) {
        ones[l] = zero;
        twos[l] = Matrix::Zero(n * n, n * n);
    }
    const xmvb::mcscf::RawCiAdjoint adjoint{seed, {&ones[0], &twos[0]}};
    const std::array<xmvb::mcscf::RawCiDirection, 2> directions{
        {{&dh, &dsource, 0, image0, dseed, {&ones[1], &twos[1]}},
         {nullptr, &dsource, 0, image1, dseed, {&ones[2], &twos[2]}}}};
    plan.apply_columns(h, x, columns, images, &adjoint, directions);
    xmvb::cuda::SimilarityHamiltonian gpu(c, n);
    auto ph = upload(c, physical.one_electron), pg = upload(c, physical.pair_kernel),
         r = upload(c, integrals.orbital_transform);
    gpu.update(ph.view(), pg.view(), r.view());
    auto delta_h = upload(c, delta.one_electron), delta_g = upload(c, delta.pair_kernel),
         zr = upload(c, zero);
    const auto direction = gpu.direction(delta_h.view(), delta_g.view(), zr.view());
    auto values = upload(c, x), dx = upload(c, dsource), seeds = upload(c, seed),
         dseeds = upload(c, dseed);
    Device out(c, na, count), out0(c, na, count), out1(c, na, count);
    std::array<Device, 3> one, two;
    for (int l = 0; l < 3; ++l) {
        one[l] = upload(c, zero);
        two[l] = upload(c, Matrix::Zero(n * n, n * n));
    }
    xmvb::cuda::RawCiBatch batch{};
    batch.count = 2;
    batch.lanes[0] = {direction.one.view(),
                      direction.two.view(),
                      dx.view(),
                      dseeds.view(),
                      out0.view(),
                      one[1].view(),
                      two[1].view()};
    batch.lanes[1] = {{}, {}, dx.view(), dseeds.view(), out1.view(), one[2].view(), two[2].view()};
    for (int repeat = 1; repeat <= 2; ++repeat) {
        xmvb::cuda::apply_raw_ci(c,
                                 topology,
                                 values.view(),
                                 cols.data(),
                                 count,
                                 gpu.one(),
                                 gpu.two(),
                                 out.view(),
                                 seeds.view(),
                                 one[0].view(),
                                 two[0].view(),
                                 batch);
        near(download(c, out.view()), images, "partitioned raw image");
        near(download(c, out0.view()), image0, "partitioned raw integral direction");
        near(download(c, out1.view()), image1, "partitioned raw CI direction");
        for (int l = 0; l < 3; ++l) {
            near(download(c, one[l].view()), repeat * ones[l], "partitioned raw one adjoint");
            near(download(c, two[l].view()), repeat * twos[l], "partitioned raw two adjoint");
        }
    }
}
void check(int ea, int eb, bool covalent = false, int n = 4) {
    std::cout << "spin sector " << ea << ',' << eb << ", covalent=" << covalent << std::endl;
    constexpr int ns = 3;
    const int pairs = n * (n + 1) / 2;
    const auto as = xmvb::mcscf::SpinStrings(n, ea).decode(),
               bs = xmvb::mcscf::SpinStrings(n, eb).decode();
    Strings a, b;
    std::vector<std::vector<xmvb::vb::StructureExpansionTerm>> terms;
    const int products = covalent ? as.size() : 5;
    for (int k = 0; k < products; ++k) {
        const int ai = covalent ? k : (k * 3) % as.size();
        a.push_back(as[ai]);
        if (covalent) {
            std::vector<int> other;
            for (int i = 0; i < n; ++i)
                if (std::find(a.back().begin(), a.back().end(), i) == a.back().end())
                    other.push_back(i);
            b.push_back(other);
        } else
            b.push_back(bs[(k * 2 + 1) % bs.size()]);
        terms.push_back({{k % ns, 0.7 + 0.1 * k}, {(k + 1) % ns, -0.13}});
    }
    Matrix r = Matrix::Identity(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            r(i, j) += 0.07 * std::sin(i + 2 * j + 1.);
    const Matrix overlap = r.transpose() * r;
    Matrix h(n, n), g(pairs, pairs);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            h(i, j) = std::cos(0.19 * (i + j + 1));
    for (int i = 0; i < pairs; ++i)
        for (int j = 0; j < pairs; ++j)
            g(i, j) = 0.03 * std::cos(0.13 * (i + j + 1));
    xmvb::integrals::ActiveSpaceTwoElectronResult eri;
    eri.packed_active_two_electron_integrals.resize(pairs * (pairs + 1) / 2);
    for (int i = 0; i < pairs; ++i)
        for (int j = 0; j <= i; ++j)
            eri.packed_active_two_electron_integrals
                [xmvb::core::PairIndex::packed_pair_of_pairs_index(i, j)] = g(i, j);
    std::vector<double> sv(overlap.data(), overlap.data() + overlap.size());
    const auto topology = xmvb::vb::build_orthogonal_spin_topology(a, b, n);
    const xmvb::vb::StructureAction cpu(
        terms, ns, topology, sv, h, eri, n, nullptr, xmvb::vb::StructureActionMode::ActionOnly);
    const auto integrals = xmvb::vb::orthogonalize_active_integrals(sv, h, eri, n);
    const xmvb::vb::OperatorFirstAction reference(
        cpu.orthogonal_view(), ns, integrals, integrals.orbital_transform);
    xmvb::cuda::Context c(64ULL * 1024 * 1024);
    xmvb::cuda::StructureAction action(c, cpu);
    xmvb::cuda::OperatorFirstAction gpu(c, action);
    auto dh = upload(c, integrals.one_electron), dg = upload(c, integrals.pair_kernel),
         dr = upload(c, integrals.orbital_transform);
    gpu.update(dh.view(), dg.view(), dr.view());
    Matrix block(ns, 5);
    for (int j = 0; j < block.cols(); ++j)
        for (int i = 0; i < block.rows(); ++i)
            block(i, j) = std::sin(0.31 * (i + 1) * (j + 2));
    auto dc = upload(c, block);
    Device out_h(c, ns, block.cols()), out_m(c, ns, block.cols());
    gpu.apply(dc.view(), out_h.view(), out_m.view(), 1 << 20);
    auto expected = cpu.apply(block);
    near(download(c, out_h.view()), expected.hamiltonian, "device source H block");
    near(download(c, out_m.view()), expected.overlap, "device source M block");
    gpu.apply(dc.view(), {}, out_m.view(), 1 << 20);
    near(download(c, out_m.view()), expected.overlap, "device source M only");
    gpu.apply(dc.view(), out_h.view(), {}, 1 << 20);
    near(download(c, out_h.view()), expected.hamiltonian, "device source H only");
    // Fit exactly two RHS and one target column, exercising both batch bounds.
    const auto& view = cpu.orthogonal_view();
    const std::size_t limited =
        sizeof(double) *
        (view.beta.dimension() + 2 * (3 * view.products.size() + 2 * view.alpha.dimension()));
    gpu.apply(dc.view(), out_h.view(), out_m.view(), limited);
    near(download(c, out_h.view()), expected.hamiltonian, "bounded device source H block");
    near(download(c, out_m.view()), expected.overlap, "bounded device source M block");
    Matrix zero_first = block;
    zero_first.col(0).setZero();
    auto dzero = upload(c, zero_first);
    gpu.apply(dzero.view(), out_h.view(), out_m.view(), 1 << 20);
    const auto zero_expected = cpu.apply(zero_first);
    near(download(c, out_h.view()), zero_expected.hamiltonian, "zero-first device source H block");
    near(download(c, out_m.view()), zero_expected.overlap, "zero-first device source M block");
    const Matrix coeff = block.leftCols(1);
    if (n == 4 && ea == 2 && eb == 2)
        check_reduction(c, cpu, integrals, coeff);
    auto state = cpu.prepare_active_adjoint(coeff, {1.0}, {0.0});
    auto base = gpu.jet(action.carrier_sigma(), dc.view().column(0), {}, {}, {}, {}, 1 << 20);
    const auto base_cpu = reference.jet(state, nullptr, nullptr, nullptr, true, 1 << 20);
    near(download(c, base.one.view()), base_cpu.one_electron, "device source one adjoint");
    near(download(c, base.pair.view()), base_cpu.pair_kernel, "device source pair adjoint");
    near(download(c, base.generator.view()), upper(base_cpu.generator), "device source generator");
    const double energy = -0.8;
    gpu.accept(energy);
    state.energies[0] = energy;
    Matrix delta_r = Matrix::Zero(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j)
            delta_r(i, j) = 0.009 * std::cos(i + 3 * j + 0.3);
    const Matrix q = 0.013 * coeff.array().cos().matrix();
    xmvb::mcscf::ActiveIntegrals delta{0.017 * integrals.one_electron,
                                       -0.023 * integrals.pair_kernel};
    auto dq = upload(c, q), drr = upload(c, delta_r), dhi = upload(c, delta.one_electron),
         dgi = upload(c, delta.pair_kernel);
    xmvb::cuda::ProductJet ci;
    const auto actual = gpu.jet(action.carrier_sigma(),
                                dc.view().column(0),
                                dq.view(),
                                drr.view(),
                                dhi.view(),
                                dgi.view(),
                                1 << 20,
                                &ci);
    const auto jet = reference.jet(state, &delta, &delta_r, &q, true, 1 << 20);
    const auto ci_ref = reference.jet(state, nullptr, nullptr, &q, true, 1 << 20);
    near(download(c, actual.hamiltonian.view()),
         jet.images.hamiltonian,
         "device source directional H");
    near(download(c, actual.overlap.view()), jet.images.overlap, "device source directional M");
    near(2 * download(c, actual.one.view()), jet.one_electron, "device source one HVP");
    near(2 * download(c, actual.pair.view()), jet.pair_kernel, "device source pair HVP");
    near(download(c, actual.generator.view()), upper(jet.generator), "device source generator HVP");
    near(download(c, ci.hamiltonian.view()), ci_ref.images.hamiltonian, "device independent CI H");
    near(download(c, ci.overlap.view()), ci_ref.images.overlap, "device independent CI M");
    near(2 * download(c, ci.one.view()), ci_ref.one_electron, "device independent CI one");
    near(2 * download(c, ci.pair.view()), ci_ref.pair_kernel, "device independent CI pair");
    near(download(c, ci.generator.view()),
         upper(ci_ref.generator),
         "device independent CI generator");
}
} // namespace
int main() {
    try {
        check(2, 2);
        check(3, 1);
        check(0, 2);
        check(2, 0);
        check(2, 2, true);
        check(3, 1, true);
        check(3, 3, true, 6);
        check(4, 2, true, 6);
        std::cout << "operator-first CUDA action and jet passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
