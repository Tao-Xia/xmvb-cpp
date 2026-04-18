#include "runtime_c/local_runtime_api_internal.h"

#include <stdio.h>
#include <string.h>

hf_info xmvb_cpp_init_hf(const mol_info mol);
void xmvb_cpp_load_hf_jaux(const char* aux_fname, hf_info hf);
void xmvb_cpp_load_hf_kgrids(const char* kgrids_fname, const char* kfgrids_fname, hf_info hf);
void xmvb_cpp_set_hf_coulomb(Jbuilder_type jtype, hf_info hf);
void xmvb_cpp_set_hf_exchange(Kbuilder_type ktype, hf_info hf);
void xmvb_cpp_load_hf_dft(
    double hf_frac,
    GRIDS_TYPE gtype,
    const int dft_id[],
    const double dft_frac[],
    int num,
    int disp_type,
    int dft_name,
    hf_info hf);
void xmvb_cpp_del_hf(hf_info hf);

static void set_error_message(
    char* error_message,
    size_t error_message_capacity,
    const char* message) {
  if (error_message == NULL || error_message_capacity == 0) {
    return;
  }
  snprintf(error_message, error_message_capacity, "%s", message);
}

int xmvb_cpp_runtime_setup_hf(
    XmvbCppRuntimeHandle* runtime_handle,
    char* error_message,
    size_t error_message_capacity) {
  if (runtime_handle == NULL || runtime_handle->input_info == NULL ||
      runtime_handle->molecule == NULL) {
    set_error_message(error_message, error_message_capacity, "runtime state is incomplete for hf setup");
    return 1;
  }

  if (runtime_handle->hf_wavefunction != NULL) {
    return 0;
  }

  runtime_handle->hf_wavefunction = xmvb_cpp_init_hf(runtime_handle->molecule);
  if (runtime_handle->hf_wavefunction == NULL) {
    set_error_message(error_message, error_message_capacity, "failed to initialize hf runtime");
    return 1;
  }
  runtime_handle->destroy_hf_fn = xmvb_cpp_del_hf;

  xmvb_cpp_load_hf_jaux(runtime_handle->input_info->aux_name, runtime_handle->hf_wavefunction);
  xmvb_cpp_load_hf_kgrids(
      runtime_handle->input_info->k_grid_file,
      runtime_handle->input_info->k_grid_file_final,
      runtime_handle->hf_wavefunction);
  xmvb_cpp_set_hf_coulomb(RI_jbuilder, runtime_handle->hf_wavefunction);
  xmvb_cpp_set_hf_exchange(COSX_kbuilder, runtime_handle->hf_wavefunction);

  if (runtime_handle->input_info->ndft > 0) {
    xmvb_cpp_load_hf_dft(
        runtime_handle->input_info->hf_frac,
        runtime_handle->input_info->grid_type,
        runtime_handle->input_info->dft_id,
        runtime_handle->input_info->dft_frac,
        runtime_handle->input_info->ndft,
        runtime_handle->input_info->disp_type,
        runtime_handle->input_info->dft_name,
        runtime_handle->hf_wavefunction);
  }

  runtime_handle->hf_wavefunction->open_type = runtime_handle->input_info->ihf_type;
  return 0;
}
