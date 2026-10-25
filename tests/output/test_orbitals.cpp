#include "output/orbitals/writer.hpp"
#include "guess/initial.hpp"

#include <filesystem>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    xmvb::vb::OrbitalPreparationInput input;
    input.n_basis_functions = 5;
    input.n_orbitals = 3;
    input.orbital_basis_counts = {3, 5, 1};
    input.orbital_basis_index_table = {1, 3, 5, 0, 0, 1, 2, 3, 4, 5, 4, 0, 0, 0, 0};
    input.orbital_value_table = {
        1.1234567890123457, 0., -1e-24, 0., 0., -0.5, 0.1, 0.2, 0.3, 0.4, 1., 0., 0., 0., 0.};
    const auto path = xmvb::output::write_orbital_file("orbital.writer.test.xmi", input);
    if (path.filename() != "orbital.writer.test.orb")
        throw std::runtime_error("incorrect orbital filename");
    std::ifstream stream(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(stream, line);) {
        if (line.find('$') != std::string::npos)
            throw std::runtime_error("unexpected GUS delimiter");
        if (!line.empty() && line.front() != '#' && line.find_first_of("eEdD") != std::string::npos)
            throw std::runtime_error("unexpected scientific notation");
        lines.push_back(line);
    }
    auto restored = input;
    restored.orbital_value_table.assign(15, 0.);
    // GUESS=READ uses the production GUS parser, including sparse AO indices.
    xmvb::vb::build_initial_orbital_guess("", 2, {}, {}, nullptr, &lines, &restored);
    for (std::size_t i = 0; i < input.orbital_value_table.size(); ++i)
        if (std::abs(restored.orbital_value_table[i] - input.orbital_value_table[i]) > 5.1e-11)
            throw std::runtime_error("GUS orbital round trip exceeds decimal rounding");
    std::filesystem::remove(path);
}
