#pragma once

#include <string>

#include "input/deck/keywords.hpp"
#include "vbscf/core/contracts/eigensolver.hpp"
#include "vbscf/core/contracts/tnvb.hpp"

namespace xmvb::vb {

/** @brief Orbital optimizer selected by the `.xmi` `ISCF` keyword. */
enum class InputScfOptimizer {
    Unspecified,
    Tnhvp,
    Plbfgs,
    Ciah,
};

/** Electronic wavefunction selected by a bare `$CTRL` method flag. */
enum class ElectronicMethod { Vbscf, Casscf, Dft };

/** KS spin channels; Auto selects UKS when NMUL is greater than one. */
enum class KohnShamSpin { Auto, Restricted, Unrestricted };

/**
 * @brief Pure C++ summary of the `$CTRL` metadata needed by the standalone loader.
 *
 * This parser intentionally targets the subset of historical XMVB deck syntax
 * that the front-end consumes directly: basis selection,
 * guess/orbital chart types, charge/unit metadata, SCF iteration limits, and
 * a few standalone flags. Keeping these fields in a dedicated C++ object
 * keeps the pure loader independent of external parser state.
 */
struct InputDeckMetadata {
    ElectronicMethod electronic_method = ElectronicMethod::Vbscf;
    bool electronic_method_explicit = false;
    std::string xc_functional = "LDA_X,LDA_C_VWN";
    KohnShamSpin ks_spin = KohnShamSpin::Auto;
    int grid_radial_points = 80;
    int grid_angular_points = 590;
    double scf_tolerance = 1.0e-8;
    std::string basis_name;
    /** @brief Optional explicit RI fitting basis; defaults to `BASIS-jkfit`. */
    std::string auxiliary_basis_name;
    std::string structure_class_keyword;
    int guess_type = kGuessTypeAuto;
    /** @brief Whether `GUESS=` was explicitly present in `$CTRL`. */
    bool has_explicit_guess_type = false;
    int orbital_type = kOrbitalTypeGen;
    int fragment_type = kFragmentTypeAtom;
    int wavefunction_type = kWavefunctionTypeStructure;
    int vb_function_type = kVbFunctionTypeDeterminant;
    int declared_active_orbitals = 0;
    int declared_active_electrons = 0;
    int declared_spin_multiplicity = 1;
    int total_charge = 0;
    int requested_scf_max_iterations = 0;
    bool has_explicit_scf_max_iterations = false;
    TnVbConfiguration tnvb;
    /** @brief Number of consecutive lowest states in an equal-weight average. */
    int state_average_count = 1;
    InputScfOptimizer scf_optimizer = InputScfOptimizer::Unspecified;
    StructureEigensolver structure_eigensolver = StructureEigensolver::Davidson;
    bool has_explicit_structure_eigensolver = false;
    bool geometry_coordinates_in_bohr = false;
    bool request_ri_two_electron_mode = false;
    bool request_molden_output = false;
};

InputDeckMetadata parse_input_deck_metadata(const std::string& input_file_path);

} // namespace xmvb::vb
