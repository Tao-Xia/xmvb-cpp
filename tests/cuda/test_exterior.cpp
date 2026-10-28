#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>

#include "cuda/memory/upload.hpp"
#include "cuda/vb/transform.hpp"

namespace {
using Host = Eigen::MatrixXd;
using namespace xmvb;

void near(const Host& actual, const Host& expected, const char* where) {
    const double error = (actual - expected).norm() / std::max(1.0, expected.norm());
    if (!actual.allFinite() || error > 4e-13)
        throw std::runtime_error(std::string(where) + " relative error=" + std::to_string(error));
}

Host values(int rows, int cols) {
    Host out(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            out(i, j) = std::sin(0.07 * (i + 1) + 0.19 * j);
    return out;
}

void check(int n, int electrons, bool shuffled, std::size_t shared_bytes) {
    mcscf::SpinStrings strings(n, electrons);
    if (shuffled) {
        auto masks = strings.masks();
        std::reverse(masks.begin(), masks.end());
        strings = mcscf::SpinStrings(n, masks);
    }
    Host upper = Host::Zero(n, n), du = Host::Zero(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) {
            upper(i, j) = i == j ? 1.0 + 0.003 * (j + 1) : 0.017 * std::sin(i + 2 * j + 1.0);
            du(i, j) = 0.013 * std::cos(2 * i + j + 0.3);
        }
    vb::ExteriorOrbitalTransform cpu(strings, upper);
    cuda::Context context(256ULL * 1024 * 1024);
    cuda::ExteriorTransform gpu(context, cpu, shared_bytes);
    cuda::Matrix<double> u(context, n, n), d(context, n, n);
    cuda::copy_to_device(context, u.view(), upper);
    cuda::copy_to_device(context, d.view(), du);
    const auto direction = gpu.direction(u.view(), d.view());
    const auto dc = cpu.direction(du);
    const int dim = strings.size();
    const Host initial = values(dim, 3), delta = values(dim, 3) * 0.23;
    // Padded leading dimensions exercise actual panel addressing.
    cuda::Matrix<double> storage(context, dim + 2, 3), tangent(context, dim + 2, 3);
    const linalg::MatrixView p{storage.data(), dim, 3, dim + 2}, t{tangent.data(), dim, 3, dim + 2};
    const int words = (dim + 63) / 64;
    std::vector<std::uint64_t> masks(3 * words);
    std::vector<Eigen::Index> offsets{0}, selected;
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < dim; ++row)
            if ((row + 2 * col) % 4 != 0) {
                selected.push_back(row);
                masks[col * words + row / 64] |= std::uint64_t{1} << (row % 64);
            }
        offsets.push_back(selected.size());
    }
    cuda::Buffer<std::uint64_t> allowed(context, masks.size());
    context.copy_to_device(allowed.data(), masks.data(), masks.size() * sizeof(std::uint64_t));
    for (bool adjoint : {false, true}) {
        Host input(dim, 6), expected, actual(dim, 6);
        input.leftCols(3) = initial;
        input.rightCols(3) = 0.8 * initial;
        expected = input;
        cpu.apply_supported_left(&expected, offsets, selected, adjoint);
        cuda::Matrix<double> panel(context, dim, 6);
        cuda::copy_to_device(context, panel.view(), input);
        gpu.apply_supported_left(panel.view(), allowed.data(), 3, adjoint);
        cuda::copy_to_host(context, actual, panel.view());
        near(actual, expected, "supported repeated RHS");
    }
    for (bool adjoint : {false, true}) {
        for (bool inverse : {false, true}) {
            Host expected = initial, actual(dim, 3);
            if (inverse) {
                if (adjoint)
                    cpu.apply_inverse_adjoint_left(&expected);
                else
                    cpu.apply_inverse_left(&expected);
            } else if (adjoint)
                cpu.apply_adjoint_left(&expected);
            else
                cpu.apply_left(&expected);
            cuda::copy_to_device(context, p, initial);
            gpu.apply_left(p, adjoint, inverse);
            cuda::copy_to_host(context, actual, p);
            near(actual, expected, "left");
            // Two RHS blocks, independent right actions.
            Host right_initial(3, 2 * dim), right_expected(3, 2 * dim), right_actual(3, 2 * dim);
            right_initial.leftCols(dim) = initial.transpose();
            right_initial.rightCols(dim) = 0.7 * initial.transpose();
            right_expected.leftCols(dim) = expected.transpose();
            right_expected.rightCols(dim) = 0.7 * expected.transpose();
            cuda::Matrix<double> right(context, 3, 2 * dim);
            cuda::copy_to_device(context, right.view(), right_initial);
            gpu.apply_right(right.view(), adjoint, inverse);
            cuda::copy_to_host(context, right_actual, right.view());
            near(right_actual, right_expected, "right");
            // Independent column tiles, including the one-row case.
            for (int rows : {1, 3}) {
                const int first = dim > 1 ? dim / 2 : dim;
                cuda::Matrix<double> a(context, rows, first), b(context, rows, dim - first);
                Host input = initial.transpose().topRows(rows);
                Host left = input.leftCols(first), right_host = input.rightCols(dim - first);
                cuda::copy_to_device(context, a.view(), left);
                std::vector<linalg::MatrixView> tiles{a.view()};
                if (dim > first) {
                    cuda::copy_to_device(context, b.view(), right_host);
                    tiles.push_back(b.view());
                }
                gpu.apply_right_tiles(tiles, adjoint, inverse);
                Host actual_left(rows, first), actual_right(rows, dim - first), out(rows, dim);
                cuda::copy_to_host(context, actual_left, a.view());
                out.leftCols(first) = actual_left;
                if (dim > first) {
                    cuda::copy_to_host(context, actual_right, b.view());
                    out.rightCols(dim - first) = actual_right;
                }
                near(out, Host(expected.transpose().topRows(rows)), "indirect");
            }
        }
        Host expected = initial, expected_delta = delta, actual(dim, 3), actual_delta(dim, 3);
        if (adjoint)
            cpu.apply_directional_adjoint_left(dc, &expected, &expected_delta);
        else
            cpu.apply_directional_left(dc, &expected, &expected_delta);
        cuda::copy_to_device(context, p, initial);
        cuda::copy_to_device(context, t, delta);
        gpu.apply_directional_left(direction, p, t, adjoint);
        cuda::copy_to_host(context, actual, p);
        cuda::copy_to_host(context, actual_delta, t);
        near(actual, expected, "directional primal");
        near(actual_delta, expected_delta, "directional tangent");
        const int first = dim > 1 ? dim / 2 : dim;
        cuda::Matrix<double> a(context, 3, first), b(context, 3, dim - first);
        cuda::Matrix<double> da(context, 3, first), db(context, 3, dim - first);
        Host left = initial.transpose().leftCols(first),
             right = initial.transpose().rightCols(dim - first);
        Host dleft = delta.transpose().leftCols(first),
             dright = delta.transpose().rightCols(dim - first);
        cuda::copy_to_device(context, a.view(), left);
        cuda::copy_to_device(context, da.view(), dleft);
        std::vector<linalg::MatrixView> tiles{a.view()}, dt{da.view()};
        if (dim > first) {
            cuda::copy_to_device(context, b.view(), right);
            cuda::copy_to_device(context, db.view(), dright);
            tiles.push_back(b.view());
            dt.push_back(db.view());
        }
        gpu.apply_directional_right_tiles(direction, tiles, dt, adjoint);
        cuda::copy_to_host(context, left, a.view());
        cuda::copy_to_host(context, dleft, da.view());
        Host out(3, dim), dout(3, dim);
        out.leftCols(first) = left;
        dout.leftCols(first) = dleft;
        if (dim > first) {
            cuda::copy_to_host(context, right, b.view());
            cuda::copy_to_host(context, dright, db.view());
            out.rightCols(dim - first) = right;
            dout.rightCols(dim - first) = dright;
        }
        near(out, Host(expected.transpose()), "indirect directional primal");
        near(dout, Host(expected_delta.transpose()), "indirect directional tangent");
    }
}
} // namespace
int main(int argc, char**) {
    try {
        for (const std::size_t bytes :
             {std::size_t{0}, std::size_t{1024}, std::numeric_limits<std::size_t>::max()}) {
            for (const auto [n, e] : {std::pair{4, 0}, {4, 4}, {6, 2}, {8, 5}}) {
                check(n, e, false, bytes);
                check(n, e, true, bytes);
            }
            if (argc == 1)
                for (int n : {14, 16, 18})
                    check(n, n / 2, false, bytes);
        }
        std::cout << "Exterior column fusion: all orders, tangents, shuffled strings and tiled "
                     "columns pass\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
