#pragma once

#include <chrono>
#include <iosfwd>
#include <string>

#include "input/loading/loader.hpp"
#include "vbscf/optimization/driver/result.hpp"

namespace xmvb::output {

void print_program_preamble(std::ostream& output,
                            const std::string& input_path,
                            const std::chrono::system_clock::time_point& start_time,
                            int n_threads);

void print_input_sections(std::ostream& output,
                          const std::string& input_path,
                          const vb::VbScfInputLoadResult& load_result,
                          const char* optimizer_name,
                          int max_iterations);

/** @brief Common scalar-only SCF report formatting for CPU and GPU drivers. */
void print_iteration_header(std::ostream& output);
void print_iteration_row(
    std::ostream& output, int iteration, double energy, double energy_change, double gradient_inf);
void print_scf_status(std::ostream& output, bool converged, int iterations, double energy);

/** @brief Print supplied coefficients without constructing norms or matrices. */
void print_structure_coefficients(std::ostream& output,
                                  const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                  const vb::RawStructureData& structures);

/** @brief Print the supplied final orbitals using the common XMVB AO table. */
void print_orbitals(std::ostream& output,
                    const vb::VbScfStaticMoleculeMetadata& metadata,
                    const vb::OrbitalPreparationInput& orbitals);

/** @brief Print the matrix-free CC weight table in the common XMVB layout. */
void print_coulson_chirgwin_weights(std::ostream& output,
                                    const Eigen::Ref<const Eigen::VectorXd>& coefficients,
                                    const Eigen::Ref<const Eigen::VectorXd>& overlap_product,
                                    const vb::RawStructureData& structures);

/** @brief Prints requested final analyses; returns the exact mean structure overlap. */
double print_final_state_sections(std::ostream& output,
                                  const vb::VbScfInputLoadResult& load_result,
                                  const vb::VbScfOptimizerResult& result);

} // namespace xmvb::output
