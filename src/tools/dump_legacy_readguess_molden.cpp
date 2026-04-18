#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "runtime_c/local_runtime_api_internal.h"
#include "cint.h"

extern "C" {
int xmvb_cpp_vb_readguess(inp_info inp_str, vb_info vb_str);
}

#include "runtime/molden_file_writer.hpp"

namespace fs = std::filesystem;

namespace {

void print_usage() {
  std::cerr
      << "usage: dump_legacy_readguess_molden <input.xmi> <output_stem.xmi>\n";
}

xmvb::vb::CppVbInput build_molden_export_input(
    const XmvbCppRuntimeHandle& runtime_handle) {
  if (runtime_handle.molecule == NULL ||
      runtime_handle.molecule->atm == NULL ||
      runtime_handle.molecule->bas == NULL ||
      runtime_handle.vb_wavefunction == NULL ||
      runtime_handle.vb_wavefunction->dv == NULL ||
      runtime_handle.vb_wavefunction->nv == NULL ||
      runtime_handle.vb_wavefunction->ma == NULL ||
      runtime_handle.vb_wavefunction->ma0 == NULL ||
      runtime_handle.vb_wavefunction->atm == NULL ||
      runtime_handle.vb_wavefunction->bas == NULL ||
      runtime_handle.vb_wavefunction->basidx == NULL ||
      runtime_handle.vb_wavefunction->env == NULL) {
    throw std::runtime_error(
        "runtime state is incomplete for Molden export after legacy readguess");
  }

  xmvb::vb::CppVbInput input;
  const int n_atoms = runtime_handle.molecule->atm->natm;
  const int n_shells = runtime_handle.molecule->bas->nbas;
  const int n_basis_functions = runtime_handle.vb_wavefunction->nb;
  const int n_orbitals = runtime_handle.vb_wavefunction->nor;

  input.libcint_input.n_atoms = n_atoms;
  input.libcint_input.n_shells = n_shells;
  input.libcint_input.n_gaussian_primitives =
      runtime_handle.vb_wavefunction->ngto;
  input.libcint_input.atm.assign(
      runtime_handle.vb_wavefunction->atm,
      runtime_handle.vb_wavefunction->atm + xmvb::to_size(n_atoms) * ATM_SLOTS);
  input.libcint_input.bas.assign(
      runtime_handle.vb_wavefunction->bas,
      runtime_handle.vb_wavefunction->bas + xmvb::to_size(n_shells) * BAS_SLOTS);
  input.libcint_input.basidx.assign(
      runtime_handle.vb_wavefunction->basidx,
      runtime_handle.vb_wavefunction->basidx + xmvb::to_size(n_shells) * 2);
  input.libcint_input.env.assign(
      runtime_handle.vb_wavefunction->env,
      runtime_handle.vb_wavefunction->env +
          xmvb::to_size(runtime_handle.vb_wavefunction->ngto) * 2 +
              xmvb::to_size(n_atoms) * 3 + PTR_ENV_START);

  input.orbital_preparation_input.n_basis_functions = n_basis_functions;
  input.orbital_preparation_input.n_orbitals = n_orbitals;
  input.orbital_preparation_input.n_active_orbitals =
      runtime_handle.vb_wavefunction->nao;
  input.orbital_preparation_input.n_total_electrons =
      runtime_handle.vb_wavefunction->nel;
  input.orbital_preparation_input.n_active_electrons =
      runtime_handle.vb_wavefunction->nae;
  input.orbital_preparation_input.spin_multiplicity =
      runtime_handle.vb_wavefunction->nmul;
  input.orbital_preparation_input.orbital_type =
      runtime_handle.vb_wavefunction->orbtyp;
  input.orbital_preparation_input.orbital_value_table.assign(
      runtime_handle.vb_wavefunction->dv,
      runtime_handle.vb_wavefunction->dv +
          xmvb::to_size(n_basis_functions) * n_orbitals);
  input.orbital_preparation_input.orbital_basis_index_table.assign(
      runtime_handle.vb_wavefunction->nv,
      runtime_handle.vb_wavefunction->nv +
          xmvb::to_size(n_basis_functions) * n_orbitals);
  input.orbital_preparation_input.orbital_basis_counts.assign(
      runtime_handle.vb_wavefunction->ma,
      runtime_handle.vb_wavefunction->ma + n_orbitals);
  input.orbital_preparation_input.original_orbital_basis_counts.assign(
      runtime_handle.vb_wavefunction->ma0,
      runtime_handle.vb_wavefunction->ma0 + n_orbitals);
  return input;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      print_usage();
      return 1;
    }

    const std::string input_path = argv[1];
    const fs::path output_stem_path = argv[2];
    const fs::path runtime_output_stem =
        fs::temp_directory_path() /
        (output_stem_path.stem().string() + "_legacy_readguess_runtime");

    XmvbCppRuntimeHandle* runtime_handle = NULL;
    char error_message[1024] = {0};
    if (xmvb_cpp_runtime_create(
            &runtime_handle,
            error_message,
            sizeof(error_message)) != 0) {
      throw std::runtime_error(
          error_message[0] != '\0' ? error_message : "failed to create runtime");
    }

    try {
      if (xmvb_cpp_runtime_load_input(
              runtime_handle,
              const_cast<char*>(input_path.c_str()),
              argv[0],
              error_message,
              sizeof(error_message)) != 0) {
        throw std::runtime_error(
            error_message[0] != '\0' ? error_message : "failed to load input");
      }
      if (xmvb_cpp_runtime_initialize_wavefunction(
              runtime_handle,
              runtime_output_stem.c_str(),
              error_message,
              sizeof(error_message)) != 0) {
        throw std::runtime_error(
            error_message[0] != '\0' ? error_message
                                     : "failed to initialize runtime wavefunction");
      }
      if (xmvb_cpp_runtime_prepare_libcint_buffers(
              runtime_handle,
              error_message,
              sizeof(error_message)) != 0) {
        throw std::runtime_error(
            error_message[0] != '\0' ? error_message
                                     : "failed to prepare runtime libcint buffers");
      }

      // This diagnostic intentionally bypasses the optional legacy `vbguess`
      // wrapper and calls the legacy-style `vb_readguess` implementation
      // directly. That isolates the `$GUS` parsing/remapping semantics from
      // HF-dependent guess paths so TiCl/FeCl open-shell cases can be compared
      // against the standalone C++ `GUESS=READ` builder one-to-one.
      if (xmvb_cpp_vb_readguess(
              runtime_handle->input_info,
              runtime_handle->vb_wavefunction) != 0) {
        throw std::runtime_error("xmvb_cpp_vb_readguess failed");
      }

      const xmvb::vb::CppVbInput input =
          build_molden_export_input(*runtime_handle);
      const fs::path molden_path =
          xmvb::vb::write_molden_file(output_stem_path, input);
      std::cout << molden_path.string() << '\n';
    } catch (...) {
      xmvb_cpp_runtime_destroy(runtime_handle);
      throw;
    }

    xmvb_cpp_runtime_destroy(runtime_handle);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "dump_legacy_readguess_molden: " << error.what() << '\n';
    return 1;
  }
}
