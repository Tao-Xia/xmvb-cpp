#include "vb/pdft/libcint_ao_grid_evaluator.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "cint.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace xmvb::vb::pdft {

namespace {

constexpr int kElementVal = 0;
constexpr int kCenterIndex = 1;
constexpr int kAtomIndex = 0;
constexpr int kAngularValue = 1;
constexpr int kExponentIndex = 2;
constexpr int kPrimitiveCountValue = 3;
constexpr int kCoefficientIndex = 4;

struct AtmInfo {
  int natm;
  int charge;
  int mult;
  int alpha_num;
  int beta_num;
  int* atm;
  double* value;
  char* mol_fname;
};

struct BasInfo {
  AtmInfo* atm;
  int nbas;
  int msize;
  int* bas;
  int* shls_p;
  double* value;
  char* bas_fname;
  int qoff;
};

using atm_info = AtmInfo*;
using bas_info = BasInfo*;

extern "C" {
int eval_ao_psi(
    double* xyz,
    int shls,
    double* buf,
    const atm_info atm,
    const bas_info bas);
void eval_ao(
    double* xyz,
    int shls,
    double* psi,
    double* psix,
    double* psiy,
    double* psiz,
    double* psixx,
    double* psiyy,
    double* psizz,
    const atm_info atm,
    const bas_info bas);
}

void validate_libcint_input_shape(const LibcintInput& libcint_input) {
  if (libcint_input.n_atoms <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one atom");
  }
  if (libcint_input.n_shells <= 0) {
    throw std::invalid_argument("LibcintInput must contain at least one shell");
  }
  if (libcint_input.atm.size() != xmvb::to_size(libcint_input.n_atoms) * ATM_SLOTS) {
    throw std::invalid_argument("LibcintInput atom table size mismatch");
  }
  if (libcint_input.bas.size() != xmvb::to_size(libcint_input.n_shells) * BAS_SLOTS) {
    throw std::invalid_argument("LibcintInput basis table size mismatch");
  }
  if (libcint_input.basidx.size() != xmvb::to_size(libcint_input.n_shells) * 2) {
    throw std::invalid_argument("LibcintInput basidx size mismatch");
  }
  if (libcint_input.env.empty()) {
    throw std::invalid_argument("LibcintInput env must not be empty");
  }
}

int infer_n_basis_functions(const LibcintInput& libcint_input) {
  validate_libcint_input_shape(libcint_input);
  const int last_shell_offset =
      libcint_input.basidx[xmvb::to_size(libcint_input.n_shells - 1) * 2];
  const int last_shell_count =
      libcint_input.basidx[xmvb::to_size(libcint_input.n_shells - 1) * 2 + 1];
  const int n_basis_functions = last_shell_offset + last_shell_count;
  if (n_basis_functions <= 0) {
    throw std::invalid_argument("LibcintInput basis-function count must be positive");
  }
  return n_basis_functions;
}

}  // namespace

struct LibcintAoGridEvaluator::LegacyAoBasisAdapter {
  AtmInfo atom_info_storage{};
  BasInfo basis_info_storage{};
  std::vector<int> atom_table;
  std::vector<double> atom_values;
  std::vector<int> basis_table;
  std::vector<int> shell_offsets;
  std::vector<double> basis_values;

  atm_info mutable_atom_info() { return &atom_info_storage; }
  bas_info mutable_basis_info() { return &basis_info_storage; }
  const AtmInfo* atom_info() const { return &atom_info_storage; }
  const BasInfo* basis_info() const { return &basis_info_storage; }
};

namespace {

std::unique_ptr<LibcintAoGridEvaluator::LegacyAoBasisAdapter> build_legacy_ao_basis_adapter(
    const LibcintInput& libcint_input) {
  auto adapter = std::make_unique<LibcintAoGridEvaluator::LegacyAoBasisAdapter>();
  adapter->atom_table.assign(xmvb::to_size(libcint_input.n_atoms) * 2, 0);
  adapter->atom_values.assign(xmvb::to_size(libcint_input.n_atoms) * 3, 0.0);
  adapter->basis_table.assign(xmvb::to_size(libcint_input.n_shells) * 5, 0);
  adapter->shell_offsets.assign(xmvb::to_size(libcint_input.n_shells), 0);

  for (int atom_index = 0; atom_index < libcint_input.n_atoms; ++atom_index) {
    const int atom_offset = xmvb::to_size(atom_index) * ATM_SLOTS;
    const int coordinate_offset = libcint_input.atm[atom_offset + PTR_COORD];
    if (coordinate_offset < 0 ||
        coordinate_offset + 2 >= static_cast<int>(libcint_input.env.size())) {
      throw std::invalid_argument("LibcintInput coordinate pointer is out of range");
    }
    adapter->atom_table[xmvb::to_size(atom_index) * 2 + kElementVal] =
        libcint_input.atm[atom_offset + CHARGE_OF];
    adapter->atom_table[xmvb::to_size(atom_index) * 2 + kCenterIndex] =
        atom_index * 3;
    adapter->atom_values[xmvb::to_size(atom_index) * 3] =
        libcint_input.env[xmvb::to_size(coordinate_offset)];
    adapter->atom_values[xmvb::to_size(atom_index) * 3 + 1] =
        libcint_input.env[xmvb::to_size(coordinate_offset + 1)];
    adapter->atom_values[xmvb::to_size(atom_index) * 3 + 2] =
        libcint_input.env[xmvb::to_size(coordinate_offset + 2)];
  }

  int basis_value_offset = 0;
  for (int shell_index = 0; shell_index < libcint_input.n_shells; ++shell_index) {
    const int basis_offset = xmvb::to_size(shell_index) * BAS_SLOTS;
    const int local_basis_offset = xmvb::to_size(shell_index) * 5;
    const int n_primitives = libcint_input.bas[basis_offset + NPRIM_OF];
    const int exponent_offset = libcint_input.bas[basis_offset + PTR_EXP];
    const int coefficient_offset = libcint_input.bas[basis_offset + PTR_COEFF];
    if (n_primitives <= 0) {
      throw std::invalid_argument("LibcintInput shell primitive count must be positive");
    }
    if (exponent_offset < 0 ||
        exponent_offset + n_primitives > static_cast<int>(libcint_input.env.size()) ||
        coefficient_offset < 0 ||
        coefficient_offset + n_primitives > static_cast<int>(libcint_input.env.size())) {
      throw std::invalid_argument("LibcintInput shell exponent/coefficient pointer is out of range");
    }

    adapter->basis_table[local_basis_offset + kAtomIndex] =
        libcint_input.bas[basis_offset + ATOM_OF];
    adapter->basis_table[local_basis_offset + kAngularValue] =
        libcint_input.bas[basis_offset + ANG_OF];
    adapter->basis_table[local_basis_offset + kExponentIndex] =
        basis_value_offset;
    adapter->basis_table[local_basis_offset + kPrimitiveCountValue] =
        n_primitives;
    adapter->basis_table[local_basis_offset + kCoefficientIndex] =
        basis_value_offset + n_primitives;
    adapter->shell_offsets[xmvb::to_size(shell_index)] =
        libcint_input.basidx[xmvb::to_size(shell_index) * 2];
    adapter->basis_values.insert(
        adapter->basis_values.end(),
        libcint_input.env.data() + exponent_offset,
        libcint_input.env.data() + exponent_offset + n_primitives);
    adapter->basis_values.insert(
        adapter->basis_values.end(),
        libcint_input.env.data() + coefficient_offset,
        libcint_input.env.data() + coefficient_offset + n_primitives);
    basis_value_offset += 2 * n_primitives;
  }

  adapter->atom_info_storage.natm = libcint_input.n_atoms;
  adapter->atom_info_storage.atm = adapter->atom_table.data();
  adapter->atom_info_storage.value = adapter->atom_values.data();

  adapter->basis_info_storage.atm = &adapter->atom_info_storage;
  adapter->basis_info_storage.nbas = libcint_input.n_shells;
  adapter->basis_info_storage.msize = infer_n_basis_functions(libcint_input);
  adapter->basis_info_storage.bas = adapter->basis_table.data();
  adapter->basis_info_storage.shls_p = adapter->shell_offsets.data();
  adapter->basis_info_storage.value = adapter->basis_values.data();
  adapter->basis_info_storage.qoff = basis_value_offset;

  return adapter;
}

void validate_grid_points(const Eigen::Ref<const Eigen::MatrixXd>& grid_points) {
  if (grid_points.cols() != 3) {
    throw std::invalid_argument("grid_points must have 3 columns");
  }
}

}  // namespace

LibcintAoGridEvaluator::LibcintAoGridEvaluator(const LibcintInput& libcint_input)
    : libcint_input_(libcint_input),
      legacy_basis_adapter_(build_legacy_ao_basis_adapter(libcint_input)),
      n_basis_functions_(infer_n_basis_functions(libcint_input)) {}

LibcintAoGridEvaluator::~LibcintAoGridEvaluator() = default;

LibcintAoGridEvaluator::LibcintAoGridEvaluator(LibcintAoGridEvaluator&&) noexcept = default;

LibcintAoGridEvaluator& LibcintAoGridEvaluator::operator=(
    LibcintAoGridEvaluator&&) noexcept = default;

AoGridValues LibcintAoGridEvaluator::evaluate_values(
    const Eigen::MatrixXd& grid_points) const {
  validate_grid_points(grid_points);
  AoGridValues result;
  result.values = Eigen::MatrixXd::Zero(grid_points.rows(), n_basis_functions_);

  // The legacy AO evaluator works shell-by-shell on Cartesian CGTOs.  This
  // wrapper keeps the mathematics explicit in the clean Eigen-facing API while
  // reusing the repository's validated primitive/contracted polynomial code.
#pragma omp parallel if(grid_points.rows() > 32)
  {
    std::vector<double> shell_values;
#pragma omp for schedule(static)
    for (int point_index = 0; point_index < grid_points.rows(); ++point_index) {
      const double xyz[3] = {
          grid_points(point_index, 0),
          grid_points(point_index, 1),
          grid_points(point_index, 2),
      };
      for (int shell_index = 0; shell_index < libcint_input_.n_shells; ++shell_index) {
        const int ao_offset = libcint_input_.basidx[xmvb::to_size(shell_index) * 2];
        const int ao_count = libcint_input_.basidx[xmvb::to_size(shell_index) * 2 + 1];
        shell_values.assign(xmvb::to_size(ao_count), 0.0);
        eval_ao_psi(
            const_cast<double*>(xyz),
            shell_index,
            shell_values.data(),
            legacy_basis_adapter_->mutable_atom_info(),
            legacy_basis_adapter_->mutable_basis_info());
        for (int local_ao = 0; local_ao < ao_count; ++local_ao) {
          result.values(point_index, ao_offset + local_ao) =
              shell_values[xmvb::to_size(local_ao)];
        }
      }
    }
  }
  return result;
}

AoGridValues LibcintAoGridEvaluator::evaluate_values_and_gradients(
    const Eigen::MatrixXd& grid_points) const {
  validate_grid_points(grid_points);
  AoGridValues result;
  result.values = Eigen::MatrixXd::Zero(grid_points.rows(), n_basis_functions_);
  result.gradients = Eigen::MatrixXd::Zero(grid_points.rows(), 3 * n_basis_functions_);

  // `eval_ao` returns Cartesian AO values together with first and second
  // derivatives for one shell at one point.  VB-PDFT currently needs only the
  // first derivatives, but we still pass scratch buffers for the second
  // derivatives because the legacy kernel computes them in one fused routine.
#pragma omp parallel if(grid_points.rows() > 32)
  {
    std::vector<double> shell_values;
    std::vector<double> shell_grad_x;
    std::vector<double> shell_grad_y;
    std::vector<double> shell_grad_z;
    std::vector<double> shell_grad_xx;
    std::vector<double> shell_grad_yy;
    std::vector<double> shell_grad_zz;
#pragma omp for schedule(static)
    for (int point_index = 0; point_index < grid_points.rows(); ++point_index) {
      const double xyz[3] = {
          grid_points(point_index, 0),
          grid_points(point_index, 1),
          grid_points(point_index, 2),
      };
      for (int shell_index = 0; shell_index < libcint_input_.n_shells; ++shell_index) {
        const int ao_offset = libcint_input_.basidx[xmvb::to_size(shell_index) * 2];
        const int ao_count = libcint_input_.basidx[xmvb::to_size(shell_index) * 2 + 1];
        shell_values.assign(xmvb::to_size(ao_count), 0.0);
        shell_grad_x.assign(xmvb::to_size(ao_count), 0.0);
        shell_grad_y.assign(xmvb::to_size(ao_count), 0.0);
        shell_grad_z.assign(xmvb::to_size(ao_count), 0.0);
        shell_grad_xx.assign(xmvb::to_size(ao_count), 0.0);
        shell_grad_yy.assign(xmvb::to_size(ao_count), 0.0);
        shell_grad_zz.assign(xmvb::to_size(ao_count), 0.0);
        eval_ao(
            const_cast<double*>(xyz),
            shell_index,
            shell_values.data(),
            shell_grad_x.data(),
            shell_grad_y.data(),
            shell_grad_z.data(),
            shell_grad_xx.data(),
            shell_grad_yy.data(),
            shell_grad_zz.data(),
            legacy_basis_adapter_->mutable_atom_info(),
            legacy_basis_adapter_->mutable_basis_info());
        for (int local_ao = 0; local_ao < ao_count; ++local_ao) {
          const int global_ao = ao_offset + local_ao;
          result.values(point_index, global_ao) =
              shell_values[xmvb::to_size(local_ao)];
          result.gradients(point_index, 3 * global_ao) =
              shell_grad_x[xmvb::to_size(local_ao)];
          result.gradients(point_index, 3 * global_ao + 1) =
              shell_grad_y[xmvb::to_size(local_ao)];
          result.gradients(point_index, 3 * global_ao + 2) =
              shell_grad_z[xmvb::to_size(local_ao)];
        }
      }
    }
  }
  return result;
}

}  // namespace xmvb::vb::pdft
