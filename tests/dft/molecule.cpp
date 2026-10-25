#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>

#include "dft/grid/atomic.hpp"
#include "dft/integration/potential.hpp"
#include "dft/xc/functional.hpp"
#include "input/deck/model.hpp"
#include "input/loading/molecule.hpp"
#include "integrals/basis/values.hpp"
#include "integrals/jk/contraction.hpp"
#include "linalg/eigensolver.hpp"

namespace {
void write_matrix(const std::filesystem::path& path, const Eigen::MatrixXd& matrix) {
    std::ofstream stream(path, std::ios::binary);
    const std::int64_t shape[] = {matrix.rows(), matrix.cols()};
    stream.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    stream.write(reinterpret_cast<const char*>(matrix.data()), matrix.size() * sizeof(double));
    if (!stream)
        throw std::runtime_error("cannot write DFT oracle matrix");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 5)
            throw std::invalid_argument(
                "expected input.xmi, oracle directory and optional radial/angular counts");
        const auto deck = xmvb::vb::parse_input_deck_model(argv[1]);
        const auto input = xmvb::input::load_molecule(deck);
        const std::filesystem::path out(argv[2]);
        std::filesystem::create_directories(out);
        const int n_bf = input.ao.n_basis_functions;
        xmvb::dft::GridOptions options;
        options.radial_points = deck.metadata.grid_radial_points;
        options.angular_points = deck.metadata.grid_angular_points;
        if (argc == 5) {
            options.radial_points = std::stoi(argv[3]);
            options.angular_points = std::stoi(argv[4]);
        }
        const auto grid = xmvb::dft::make_grid(input.libcint_input, options);
        const xmvb::integrals::BasisEvaluator basis(input.libcint_input);
        xmvb::integrals::BasisValues values;
        Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(n_bf, n_bf);
        for (Eigen::Index begin = 0; begin < grid.points.rows(); begin += 4096) {
            const Eigen::Index rows = std::min<Eigen::Index>(4096, grid.points.rows() - begin);
            basis.evaluate(grid.points.middleRows(begin, rows), 0, &values);
            const Eigen::MatrixXd weighted =
                grid.weights.segment(begin, rows).asDiagonal() * values.value;
            overlap.noalias() += values.value.transpose() * weighted;
        }
        const double overlap_error = (overlap - input.overlap).cwiseAbs().maxCoeff();
        std::cout << std::setprecision(17) << "overlap_error=" << overlap_error << '\n';

        xmvb::linalg::GeneralizedEigensolver solver;
        const auto& core = input.ao.ao_core_hamiltonian_matrix;
        const auto eigen =
            solver.solve_dense(std::span<const double>(core.data(), core.size()),
                               std::span<const double>(input.overlap.data(), input.overlap.size()),
                               n_bf);
        const Eigen::Map<const Eigen::MatrixXd> orbitals(
            eigen.eigenvector_matrix.data(), n_bf, n_bf);
        const int twice_ms = deck.metadata.declared_spin_multiplicity - 1;
        const int n_alpha = (input.n_electrons + twice_ms) / 2;
        const int n_beta = input.n_electrons - n_alpha;
        const bool unrestricted =
            deck.metadata.ks_spin == xmvb::vb::KohnShamSpin::Unrestricted ||
            (deck.metadata.ks_spin == xmvb::vb::KohnShamSpin::Auto && twice_ms != 0);
        const int n_spin = unrestricted ? 2 : 1;
        const xmvb::dft::Functional functional(deck.metadata.xc_functional, n_spin);
        const xmvb::dft::XcIntegrator integrator(input.libcint_input, grid, functional);
        std::array<Eigen::MatrixXd, 2> density;
        density[0] = (unrestricted ? 1.0 : 2.0) * orbitals.leftCols(n_alpha) *
                     orbitals.leftCols(n_alpha).transpose();
        if (unrestricted)
            density[1] = orbitals.leftCols(n_beta) * orbitals.leftCols(n_beta).transpose();
        const std::array<Eigen::Ref<const Eigen::MatrixXd>, 2> channels{density[0], density[1]};
        const auto xc = integrator.evaluate(std::span(channels).first(n_spin));
        Eigen::MatrixXd probes(31, 3);
        for (int row = 0; row < probes.rows(); ++row)
            for (int axis = 0; axis < 3; ++axis)
                probes(row, axis) = 0.047 * ((row * (axis + 3)) % 29 - 14);
        basis.evaluate(probes, 1, &values);
        write_matrix(out / "probes.bin", probes);
        write_matrix(out / "ao.bin", values.value);
        for (int axis = 0; axis < 3; ++axis)
            write_matrix(out / ("ao-d" + std::to_string(axis) + ".bin"), values.gradient[axis]);
        write_matrix(out / "grid.bin", grid.points);
        write_matrix(out / "weights.bin", grid.weights);
        write_matrix(out / "overlap.bin", input.overlap);
        write_matrix(out / "core.bin", core);
        for (int spin = 0; spin < n_spin; ++spin) {
            const std::string suffix = unrestricted ? (spin == 0 ? "-a" : "-b") : "";
            write_matrix(out / ("density" + suffix + ".bin"), density[spin]);
            write_matrix(out / ("vxc" + suffix + ".bin"), xc.potential[spin]);
        }
        Eigen::MatrixXd total = density[0];
        if (unrestricted)
            total += density[1];
        write_matrix(out / "coulomb.bin", xmvb::integrals::contract_jk(total, input.ao));
        std::ofstream summary(out / "summary.txt");
        summary << std::setprecision(17) << "overlap_error=" << overlap_error << '\n'
                << "xc_energy=" << xc.energy << '\n'
                << "grid_electrons=" << xc.electron_count[0] + xc.electron_count[1] << '\n'
                << "grid_alpha=" << xc.electron_count[0] << '\n'
                << "grid_beta=" << xc.electron_count[1] << '\n'
                << "n_spin=" << n_spin << '\n'
                << "nuclear_energy=" << input.nuclear_energy << '\n';
        if (overlap_error > 1e-6)
            throw std::runtime_error("DFT quadrature does not reproduce AO overlap");
        std::cout << "DFT molecular quadrature and oracle export: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
