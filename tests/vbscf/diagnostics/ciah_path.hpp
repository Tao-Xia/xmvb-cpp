#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "vbscf/orbitals/preparation/preparer.hpp"
#include "vbscf/structures/assembly/action.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

namespace xmvb::test {

/** @brief Export tiny actual CIAH snapshots for independent physical overlaps.
 * No optimizer callback, strategy or threshold is modified. Full spin-product
 * columns are permitted only in this explicitly bounded diagnostic reference.
 */
inline void export_ciah_path(const vb::OrbitalPreparationInput& input,
                             const vb::StructureAction& action,
                             const std::filesystem::path& directory) {
    const auto view = action.orthogonal_view();
    const auto na = input.n_active_orbitals;
    const auto nc = (input.n_total_electrons - input.n_active_electrons) / 2;
    const auto ua = view.alpha.strings().size(), ub = view.beta.strings().size();
    const auto ns = action.n_structures();
    if (na > 8 || ns > 8 || ua > 64 || ub > 64 || ua * ub > 512)
        throw std::length_error("CIAH path reference requires a tiny selected space");
    const auto output = directory / "prepared";
    std::filesystem::create_directories(output);
    const auto write = [](std::ofstream& file, const auto* data, std::size_t size) {
        file.write(reinterpret_cast<const char*>(data), sizeof(*data) * size);
        if (!file)
            throw std::runtime_error("cannot write CIAH path reference");
    };
    std::ofstream basis(output / "basis.bin", std::ios::binary);
    const std::array<std::uint64_t, 8> header = {1,
                                                 input.n_basis_functions,
                                                 nc,
                                                 na,
                                                 static_cast<std::uint64_t>(ns),
                                                 static_cast<std::uint64_t>(ua),
                                                 static_cast<std::uint64_t>(ub),
                                                 1};
    write(basis, header.data(), header.size());
    write(basis, view.alpha.strings().masks().data(), ua);
    write(basis, view.beta.strings().masks().data(), ub);
    Eigen::MatrixXd w = Eigen::MatrixXd::Zero(ua * ub, ns);
    for (std::size_t product = 0; product < view.products.size(); ++product) {
        const auto row = view.products[product] / ub + ua * (view.products[product] % ub);
        for (auto term = view.offsets[product]; term < view.offsets[product + 1]; ++term)
            w(row, view.terms[term].structure) += view.terms[term].coefficient;
    }
    write(basis, input.ao_overlap_matrix.data(), input.ao_overlap_matrix.size());
    write(basis, w.data(), w.size());
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (!entry.is_regular_file() || !name.starts_with("state-") ||
            entry.path().extension() != ".bin")
            continue;
        std::ifstream file(entry.path(), std::ios::binary);
        std::array<std::uint64_t, 4> dimensions{};
        file.read(reinterpret_cast<char*>(dimensions.data()), sizeof(dimensions));
        if (!file || dimensions[0] != input.n_basis_functions ||
            dimensions[1] != input.n_orbitals || dimensions[2] != ns || dimensions[3] != 1)
            throw std::runtime_error("CIAH path snapshot dimensions disagree");
        Eigen::MatrixXd raw(dimensions[0], dimensions[1]), c(ns, 1);
        file.read(reinterpret_cast<char*>(raw.data()), raw.size() * sizeof(double));
        file.read(reinterpret_cast<char*>(c.data()), c.size() * sizeof(double));
        if (!file)
            throw std::runtime_error("truncated CIAH state snapshot");
        auto point = input;
        for (std::size_t i = 0; i < input.n_orbitals; ++i)
            for (int j = 0; j < vb::stored_sparse_orbital_coefficient_count(input, i); ++j) {
                const auto slot = i * input.n_basis_functions + j;
                point.orbital_value_table[slot] = raw(input.orbital_basis_index_table[slot] - 1, i);
            }
        const auto prepared = vb::ActiveSpaceOrbitalPreparer{}.prepare(point);
        const Eigen::MatrixXd core =
            prepared.physical_orbital_frame.normalized_orbital_matrix.leftCols(nc);
        const Eigen::MatrixXd active = prepared.auxiliary_orbital_matrix.middleCols(nc, na);
        std::ofstream destination(output / name, std::ios::binary);
        write(destination, core.data(), core.size());
        write(destination, active.data(), active.size());
        write(destination, c.data(), c.size());
        ++count;
    }
    std::cout << "PATH_EXPORT,states=" << count << ",directory=" << output << '\n';
}

} // namespace xmvb::test
