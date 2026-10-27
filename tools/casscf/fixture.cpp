#include "fixture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/ao/graph.hpp"

namespace xmvb::casscf::fixture {
namespace {

template <typename T> T get(std::ifstream& stream, const char* field) {
    T value{};
    if (!stream.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error(std::string("truncated fixture: ") + field);
    return value;
}

Eigen::MatrixXd get_matrix(std::ifstream& stream, int nbf, const char* field) {
    const std::size_t count = static_cast<std::size_t>(nbf) * nbf;
    std::vector<double> row_major(count);
    if (!stream.read(reinterpret_cast<char*>(row_major.data()),
                     static_cast<std::streamsize>(count * sizeof(double))))
        throw std::runtime_error(std::string("truncated fixture: ") + field);
    Eigen::MatrixXd result(nbf, nbf);
    for (int i = 0; i < nbf; ++i)
        for (int j = 0; j < nbf; ++j) {
            const double x = row_major[static_cast<std::size_t>(i) * nbf + j];
            if (!std::isfinite(x))
                throw std::runtime_error(std::string("nonfinite fixture matrix: ") + field);
            result(i, j) = x;
        }
    return result;
}

} // namespace

Input read(const std::filesystem::path& path) {
    static_assert(sizeof(double) == 8 && sizeof(std::uint32_t) == 4);
    if (std::endian::native != std::endian::little)
        throw std::runtime_error("XMCASAO1 requires a little-endian host");
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("cannot open fixture: " + path.string());
    std::array<char, 8> magic{};
    if (!stream.read(magic.data(), magic.size()) ||
        magic != std::array<char, 8>{'X', 'M', 'C', 'A', 'S', 'A', 'O', '1'})
        throw std::runtime_error("invalid XMCASAO1 fixture magic");
    const auto nbf_u = get<std::uint32_t>(stream, "nbf");
    const auto ncore_u = get<std::uint32_t>(stream, "ncore");
    const auto nactive_u = get<std::uint32_t>(stream, "nactive");
    const auto nelec_u = get<std::uint32_t>(stream, "active electrons");
    const auto spin_u = get<std::uint32_t>(stream, "twice spin");
    const double nuclear = get<double>(stream, "nuclear energy");
    if (nbf_u == 0 || nbf_u > 1000 || ncore_u > nbf_u || nactive_u == 0 ||
        nactive_u > nbf_u - ncore_u || nelec_u > 2 * nactive_u || spin_u > nelec_u ||
        nelec_u + spin_u > 2 * nactive_u || (nelec_u - spin_u) % 2 != 0 || !std::isfinite(nuclear))
        throw std::runtime_error("invalid XMCASAO1 dimensions or spin");
    const int nbf = static_cast<int>(nbf_u);
    const std::uint64_t npair = static_cast<std::uint64_t>(nbf) * (nbf + 1) / 2;
    const std::uint64_t neri = npair * (npair + 1) / 2;
    const std::uint64_t expected = 36 + 3ull * nbf * nbf * 8 + neri * 8;
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("XMCASAO1 fixture byte count is inconsistent");
    if (neri > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("AO ERI stream exceeds addressable size");

    Input input;
    input.ncore = static_cast<int>(ncore_u);
    input.nactive = static_cast<int>(nactive_u);
    input.active_electrons = static_cast<int>(nelec_u);
    input.twice_spin = static_cast<int>(spin_u);
    input.nuclear_energy = nuclear;
    input.overlap = get_matrix(stream, nbf, "overlap");
    input.ao.ao_core_hamiltonian_matrix = get_matrix(stream, nbf, "hcore");
    input.initial_orbitals = get_matrix(stream, nbf, "initial orbitals");
    input.ao.n_basis_functions = nbf;
    if (!input.overlap.isApprox(input.overlap.transpose(), 1e-11) ||
        !input.ao.ao_core_hamiltonian_matrix.isApprox(
            input.ao.ao_core_hamiltonian_matrix.transpose(), 1e-11) ||
        !(input.initial_orbitals.transpose() * input.overlap * input.initial_orbitals)
             .isApprox(Eigen::MatrixXd::Identity(nbf, nbf), 1e-8))
        throw std::runtime_error("invalid AO symmetry or initial MO orthonormality");

    std::vector<std::int64_t> left;
    std::vector<std::int64_t> right;
    std::vector<double> values;
    left.reserve(static_cast<std::size_t>(neri));
    right.reserve(static_cast<std::size_t>(neri));
    values.reserve(static_cast<std::size_t>(neri));
    constexpr std::size_t chunk_size = 32768;
    std::array<double, chunk_size> chunk{};
    std::size_t filled = 0, current = 0;
    for (int p = 0; p < static_cast<int>(npair); ++p)
        for (int q = 0; q <= p; ++q) {
            if (current == filled) {
                filled = static_cast<std::size_t>(
                    std::min<std::uint64_t>(chunk_size, neri - values.size()));
                if (!stream.read(reinterpret_cast<char*>(chunk.data()),
                                 static_cast<std::streamsize>(filled * sizeof(double))))
                    throw std::runtime_error("truncated AO ERI stream");
                current = 0;
            }
            const double value = chunk[current++];
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite AO ERI");
            left.push_back(p);
            right.push_back(q);
            values.push_back(value);
        }
    input.ao.pair_graph =
        integrals::build_ao_pair_graph(std::move(left), std::move(right), std::move(values), nbf);
    return input;
}

} // namespace xmvb::casscf::fixture
