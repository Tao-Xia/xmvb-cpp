#include "output/orbitals/writer.hpp"

#include <fstream>
#include <iomanip>
#include <locale>
#include <span>
#include <stdexcept>
#include <vector>

namespace xmvb::output {
namespace {

std::filesystem::path write_orbitals(const std::filesystem::path& input_path,
                                     std::size_t n_bf,
                                     std::span<const int> counts,
                                     std::span<const double> coefficients,
                                     std::span<const int> indices) {
    auto path = std::filesystem::absolute(input_path);
    path.replace_extension(".orb");
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error("cannot open orbital output: " + path.string());
    out.imbue(std::locale::classic());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        out << std::setw(6) << counts[i];
        if ((i + 1) % 20 == 0 || i + 1 == counts.size())
            out << '\n';
    }
    // Match the XMVB GUS layout: ten decimal places, four coefficient/index pairs.
    out << std::fixed << std::setprecision(10);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const int count = counts[i];
        out << "# ORBITAL " << i + 1 << " NAO = " << count << '\n';
        for (int j = 0; j < count; ++j) {
            const auto offset = i * n_bf + j;
            out << std::setw(14) << coefficients[offset] << ' ' << std::setw(6)
                << (indices.empty() ? j + 1 : indices[offset]);
            if ((j + 1) % 4 == 0 || j + 1 == count)
                out << '\n';
            else
                out << ' ';
        }
    }
    out.close();
    if (!out)
        throw std::runtime_error("cannot write orbital output: " + path.string());
    return path;
}

} // namespace

std::filesystem::path write_orbital_file(const std::filesystem::path& input_path,
                                         const vb::OrbitalPreparationInput& orbitals) {
    std::vector<int> counts(orbitals.n_orbitals);
    for (std::size_t i = 0; i < counts.size(); ++i)
        counts[i] = vb::stored_sparse_orbital_coefficient_count(orbitals, i);
    return write_orbitals(input_path,
                          orbitals.n_basis_functions,
                          counts,
                          orbitals.orbital_value_table,
                          orbitals.orbital_basis_index_table);
}

std::filesystem::path write_orbital_file(const std::filesystem::path& input_path,
                                         const Eigen::MatrixXd& orbitals) {
    if (orbitals.rows() <= 0 || orbitals.cols() <= 0)
        throw std::invalid_argument("cannot write an empty orbital frame");
    const std::vector<int> counts(orbitals.cols(), orbitals.rows());
    return write_orbitals(input_path,
                          orbitals.rows(),
                          counts,
                          std::span<const double>(orbitals.data(), orbitals.size()),
                          {});
}

} // namespace xmvb::output
