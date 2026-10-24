#include "libcint/direct_shell.hpp"
#include "libcint/shell_normalization.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

extern "C" {
#include "cint.h"
#include "cint_funcs.h"

extern CINTIntegralFunction int2c2e_cart;
extern CINTIntegralFunction int3c2e_cart;

FINT cint1e_ovlp_cart(
    double* buf, FINT* shls, FINT* atm, FINT natm, FINT* bas, FINT nbas, double* env);
FINT cint1e_kin_cart(
    double* buf, FINT* shls, FINT* atm, FINT natm, FINT* bas, FINT nbas, double* env);
FINT cint1e_nuc_cart(
    double* buf, FINT* shls, FINT* atm, FINT natm, FINT* bas, FINT nbas, double* env);
FINT cint1e_r_cart(
    double* buf, FINT* shls, FINT* atm, FINT natm, FINT* bas, FINT nbas, double* env);
void cint2c2e_cart_optimizer(
    CINTOpt** opt, FINT* atm, FINT natm, FINT* bas, FINT nbas, double* env);
void cint3c2e_cart_optimizer(
    CINTOpt** opt, FINT* atm, FINT natm, FINT* bas, FINT nbas, double* env);
}

#include "integrals/ao/libcint.hpp"

namespace xmvb::libcint {

namespace {

/** @brief Evaluate with caller scratch; capacity grows only for larger shells. */
CACHE_SIZE_T evaluate_cached(CINTIntegralFunction* integral,
                             std::vector<double>& values,
                             FINT* shells,
                             const int* atm,
                             int n_atoms,
                             const int* bas,
                             int n_shells,
                             const double* env,
                             void* optimizer,
                             LibcintShellWorkspace& workspace) {
    auto* opt = reinterpret_cast<CINTOpt*>(optimizer);
    const CACHE_SIZE_T size = integral(nullptr,
                                       nullptr,
                                       shells,
                                       const_cast<int*>(atm),
                                       n_atoms,
                                       const_cast<int*>(bas),
                                       n_shells,
                                       const_cast<double*>(env),
                                       opt,
                                       nullptr);
    if (size <= 0) {
        throw std::runtime_error("invalid libcint scratch size");
    }
    if (workspace.cache.size() < static_cast<std::size_t>(size)) {
        workspace.cache.resize(static_cast<std::size_t>(size));
    }
    return integral(values.data(),
                    nullptr,
                    shells,
                    const_cast<int*>(atm),
                    n_atoms,
                    const_cast<int*>(bas),
                    n_shells,
                    const_cast<double*>(env),
                    opt,
                    workspace.cache.data());
}

enum class OneElectronIntegralKind {
    Overlap,
    CoreHamiltonian,
    Kinetic,
};

void validate_shared_atom_tables(const xmvb::integrals::LibcintInput& primary_input,
                                 const xmvb::integrals::LibcintInput& auxiliary_input) {
    if (auxiliary_input.n_atoms != primary_input.n_atoms) {
        throw std::invalid_argument("primary and auxiliary LibcintInput atom counts differ");
    }
    for (int atom = 0; atom < primary_input.n_atoms; ++atom) {
        const int primary_atom_offset = atom * ATM_SLOTS;
        const int auxiliary_atom_offset = atom * ATM_SLOTS;
        if (primary_input.atm[primary_atom_offset + CHARGE_OF] !=
            auxiliary_input.atm[auxiliary_atom_offset + CHARGE_OF]) {
            throw std::invalid_argument("primary and auxiliary atom charges differ");
        }
        const int primary_coordinate_offset = primary_input.atm[primary_atom_offset + PTR_COORD];
        const int auxiliary_coordinate_offset =
            auxiliary_input.atm[auxiliary_atom_offset + PTR_COORD];
        for (int component = 0; component < 3; ++component) {
            if (primary_input.env[primary_coordinate_offset + component] !=
                auxiliary_input.env[auxiliary_coordinate_offset + component]) {
                throw std::invalid_argument("primary and auxiliary atom coordinates differ");
            }
        }
    }
}

std::vector<double> build_combined_env(const xmvb::integrals::LibcintInput& primary_input,
                                       const xmvb::integrals::LibcintInput& auxiliary_input) {
    std::vector<double> combined_env;
    combined_env.reserve(primary_input.env.size() + auxiliary_input.env.size());
    combined_env.insert(combined_env.end(), primary_input.env.begin(), primary_input.env.end());
    combined_env.insert(combined_env.end(), auxiliary_input.env.begin(), auxiliary_input.env.end());
    return combined_env;
}

std::vector<int> build_combined_basis(const xmvb::integrals::LibcintInput& primary_input,
                                      const xmvb::integrals::LibcintInput& auxiliary_input) {
    std::vector<int> combined_bas;
    combined_bas.reserve(primary_input.bas.size() + auxiliary_input.bas.size());
    combined_bas.insert(combined_bas.end(), primary_input.bas.begin(), primary_input.bas.end());
    const int auxiliary_env_offset = static_cast<int>(primary_input.env.size());
    for (int shell = 0; shell < auxiliary_input.n_shells; ++shell) {
        const std::size_t shell_offset = static_cast<std::size_t>(shell) * BAS_SLOTS;
        const std::size_t combined_offset = combined_bas.size();
        combined_bas.insert(combined_bas.end(),
                            auxiliary_input.bas.begin() + shell_offset,
                            auxiliary_input.bas.begin() + shell_offset + BAS_SLOTS);
        combined_bas[combined_offset + PTR_EXP] += auxiliary_env_offset;
        combined_bas[combined_offset + PTR_COEFF] += auxiliary_env_offset;
    }
    return combined_bas;
}

} // namespace

LibcintDirectShellEvaluator::LibcintDirectShellEvaluator(const xmvb::integrals::LibcintInput& input)
    : input_(input), n_basis_functions_(xmvb::integrals::infer_n_basis_functions(input)) {
    xmvb::integrals::validate_libcint_input_shape(input_);
    ao_normalization_ = xmvb::integrals::build_cartesian_ao_normalization(input_);

    cint2e_cart_optimizer(reinterpret_cast<CINTOpt**>(&two_electron_optimizer_),
                          const_cast<int*>(input_.atm.data()),
                          input_.n_atoms,
                          const_cast<int*>(input_.bas.data()),
                          input_.n_shells,
                          const_cast<double*>(input_.env.data()));
}

LibcintDirectShellEvaluator::LibcintDirectShellEvaluator(
    const xmvb::integrals::LibcintInput& primary_input,
    const xmvb::integrals::LibcintInput& auxiliary_input)
    : input_(primary_input), auxiliary_input_(&auxiliary_input),
      n_basis_functions_(xmvb::integrals::infer_n_basis_functions(primary_input)),
      n_auxiliary_basis_functions_(xmvb::integrals::infer_n_basis_functions(auxiliary_input)) {
    xmvb::integrals::validate_libcint_input_shape(input_);
    xmvb::integrals::validate_libcint_input_shape(*auxiliary_input_);
    validate_shared_atom_tables(input_, *auxiliary_input_);

    ao_normalization_ = xmvb::integrals::build_cartesian_ao_normalization(input_);
    auxiliary_ao_normalization_ =
        xmvb::integrals::build_cartesian_ao_normalization(*auxiliary_input_);
    // Auxiliary metric truncation is defined in unit-radial Cartesian AOs,
    // matching PySCF. Primary orbitals remain component-wise unit normalized.
    for (int shell = 0; shell < auxiliary_input_->n_shells; ++shell) {
        const int l = auxiliary_input_->bas[shell * BAS_SLOTS + ANG_OF];
        const int n_cart = cartesian_ao_count(l);
        const int offset = xmvb::integrals::shell_ao_offset(*auxiliary_input_, shell);
        const int n_contractions = auxiliary_input_->bas[shell * BAS_SLOTS + NCTR_OF];
        for (int contraction = 0; contraction < n_contractions; ++contraction) {
            int component = 0;
            for (int lx = l; lx >= 0; --lx) {
                for (int ly = l - lx; ly >= 0; --ly) {
                    auxiliary_ao_normalization_[offset + contraction * n_cart + component++] *=
                        radial_cartesian_scale(lx, ly, l - lx - ly);
                }
            }
        }
    }
    combined_atm_.assign(input_.atm.begin(), input_.atm.end());
    combined_bas_ = build_combined_basis(input_, *auxiliary_input_);
    combined_env_ = build_combined_env(input_, *auxiliary_input_);

    cint2e_cart_optimizer(reinterpret_cast<CINTOpt**>(&two_electron_optimizer_),
                          const_cast<int*>(input_.atm.data()),
                          input_.n_atoms,
                          const_cast<int*>(input_.bas.data()),
                          input_.n_shells,
                          const_cast<double*>(input_.env.data()));
    cint2c2e_cart_optimizer(reinterpret_cast<CINTOpt**>(&auxiliary_metric_optimizer_),
                            const_cast<int*>(auxiliary_input_->atm.data()),
                            auxiliary_input_->n_atoms,
                            const_cast<int*>(auxiliary_input_->bas.data()),
                            auxiliary_input_->n_shells,
                            const_cast<double*>(auxiliary_input_->env.data()));
    cint3c2e_cart_optimizer(reinterpret_cast<CINTOpt**>(&three_center_optimizer_),
                            combined_atm_.data(),
                            input_.n_atoms,
                            combined_bas_.data(),
                            input_.n_shells + auxiliary_input_->n_shells,
                            combined_env_.data());
}

LibcintDirectShellEvaluator::~LibcintDirectShellEvaluator() {
    CINTdel_optimizer(reinterpret_cast<CINTOpt**>(&two_electron_optimizer_));
    CINTdel_optimizer(reinterpret_cast<CINTOpt**>(&auxiliary_metric_optimizer_));
    CINTdel_optimizer(reinterpret_cast<CINTOpt**>(&three_center_optimizer_));
}

void LibcintDirectShellEvaluator::validate_shell_index(int shell_index) const {
    if (shell_index < 0 || shell_index >= input_.n_shells) {
        throw std::out_of_range("shell index out of range");
    }
}

void LibcintDirectShellEvaluator::validate_auxiliary_shell_index(int shell_index) const {
    if (auxiliary_input_ == nullptr) {
        throw std::logic_error("auxiliary shell access requires an auxiliary LibcintInput");
    }
    if (shell_index < 0 || shell_index >= auxiliary_input_->n_shells) {
        throw std::out_of_range("auxiliary shell index out of range");
    }
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_one_electron_shell_pair(
    int left_shell, int right_shell, int integral_kind) const {
    validate_shell_index(left_shell);
    validate_shell_index(right_shell);

    const int left_ao_offset = xmvb::integrals::shell_ao_offset(input_, left_shell);
    const int right_ao_offset = xmvb::integrals::shell_ao_offset(input_, right_shell);
    const int left_ao_count = xmvb::integrals::shell_ao_count(input_, left_shell);
    const int right_ao_count = xmvb::integrals::shell_ao_count(input_, right_shell);
    std::vector<double> buffer(left_ao_count * right_ao_count, 0.0);
    FINT shell_pair[2] = {left_shell, right_shell};

    FINT status = 0;
    switch (static_cast<OneElectronIntegralKind>(integral_kind)) {
        case OneElectronIntegralKind::Overlap:
            status = cint1e_ovlp_cart(buffer.data(),
                                      shell_pair,
                                      const_cast<int*>(input_.atm.data()),
                                      input_.n_atoms,
                                      const_cast<int*>(input_.bas.data()),
                                      input_.n_shells,
                                      const_cast<double*>(input_.env.data()));
            if (status != 0) {
                // The libcint adapter uses the same AO normalization convention as
                // Every AO matrix lives in the
                // diagonally normalized Cartesian AO basis.  Core-H and ERI shell
                // blocks were already converted at this boundary; overlap must be
                // scaled the same way or the generalized eigenproblems in the C++
                // guess path become inconsistent.
                for (std::size_t index = 0; index < buffer.size(); ++index) {
                    const int row = static_cast<int>(index % left_ao_count);
                    const int column = static_cast<int>(index / left_ao_count);
                    const double normalization = ao_normalization_[left_ao_offset + row] *
                                                 ao_normalization_[right_ao_offset + column];
                    buffer[index] *= normalization;
                }
            }
            break;
        case OneElectronIntegralKind::CoreHamiltonian: {
            std::vector<double> nuclear_buffer(buffer.size(), 0.0);
            const FINT kinetic_status = cint1e_kin_cart(buffer.data(),
                                                        shell_pair,
                                                        const_cast<int*>(input_.atm.data()),
                                                        input_.n_atoms,
                                                        const_cast<int*>(input_.bas.data()),
                                                        input_.n_shells,
                                                        const_cast<double*>(input_.env.data()));
            const FINT nuclear_status = cint1e_nuc_cart(nuclear_buffer.data(),
                                                        shell_pair,
                                                        const_cast<int*>(input_.atm.data()),
                                                        input_.n_atoms,
                                                        const_cast<int*>(input_.bas.data()),
                                                        input_.n_shells,
                                                        const_cast<double*>(input_.env.data()));
            status = kinetic_status || nuclear_status;
            if (status != 0) {
                for (std::size_t index = 0; index < buffer.size(); ++index) {
                    const int row = static_cast<int>(index % left_ao_count);
                    const int column = static_cast<int>(index / left_ao_count);
                    const double normalization = ao_normalization_[left_ao_offset + row] *
                                                 ao_normalization_[right_ao_offset + column];
                    buffer[index] = (buffer[index] + nuclear_buffer[index]) * normalization;
                }
            }
            break;
        }
        case OneElectronIntegralKind::Kinetic:
            status = cint1e_kin_cart(buffer.data(),
                                     shell_pair,
                                     const_cast<int*>(input_.atm.data()),
                                     input_.n_atoms,
                                     const_cast<int*>(input_.bas.data()),
                                     input_.n_shells,
                                     const_cast<double*>(input_.env.data()));
            if (status != 0) {
                for (std::size_t index = 0; index < buffer.size(); ++index) {
                    const int row = static_cast<int>(index % left_ao_count);
                    const int column = static_cast<int>(index / left_ao_count);
                    buffer[index] *= ao_normalization_[left_ao_offset + row] *
                                     ao_normalization_[right_ao_offset + column];
                }
            }
            break;
    }

    if (status == 0) {
        std::fill(buffer.begin(), buffer.end(), 0.0);
    }

    LibcintShellBlock result;
    result.left_shell = left_shell;
    result.right_shell = right_shell;
    result.left_ao_offset = left_ao_offset;
    result.right_ao_offset = right_ao_offset;
    result.left_ao_count = left_ao_count;
    result.right_ao_count = right_ao_count;
    result.values = std::move(buffer);
    return result;
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_overlap_shell_pair(int left_shell,
                                                                           int right_shell) const {
    return evaluate_one_electron_shell_pair(
        left_shell, right_shell, static_cast<int>(OneElectronIntegralKind::Overlap));
}

LibcintShellBlock
LibcintDirectShellEvaluator::evaluate_core_hamiltonian_shell_pair(int left_shell,
                                                                  int right_shell) const {
    return evaluate_one_electron_shell_pair(
        left_shell, right_shell, static_cast<int>(OneElectronIntegralKind::CoreHamiltonian));
}

LibcintShellBlock LibcintDirectShellEvaluator::evaluate_kinetic_shell_pair(int left_shell,
                                                                           int right_shell) const {
    return evaluate_one_electron_shell_pair(
        left_shell, right_shell, static_cast<int>(OneElectronIntegralKind::Kinetic));
}

LibcintVectorShellBlock
LibcintDirectShellEvaluator::evaluate_position_shell_pair(int left_shell, int right_shell) const {
    validate_shell_index(left_shell);
    validate_shell_index(right_shell);

    const int left_ao_offset = xmvb::integrals::shell_ao_offset(input_, left_shell);
    const int right_ao_offset = xmvb::integrals::shell_ao_offset(input_, right_shell);
    const int left_ao_count = xmvb::integrals::shell_ao_count(input_, left_shell);
    const int right_ao_count = xmvb::integrals::shell_ao_count(input_, right_shell);
    const std::size_t component_size = left_ao_count * right_ao_count;
    std::vector<double> buffer(3 * component_size, 0.0);
    FINT shell_pair[2] = {left_shell, right_shell};
    const FINT status = cint1e_r_cart(buffer.data(),
                                      shell_pair,
                                      const_cast<int*>(input_.atm.data()),
                                      input_.n_atoms,
                                      const_cast<int*>(input_.bas.data()),
                                      input_.n_shells,
                                      const_cast<double*>(input_.env.data()));
    if (status != 0) {
        for (int component = 0; component < 3; ++component) {
            for (std::size_t index = 0; index < component_size; ++index) {
                const int row = static_cast<int>(index % left_ao_count);
                const int column = static_cast<int>(index / left_ao_count);
                buffer[component * component_size + index] *=
                    ao_normalization_[left_ao_offset + row] *
                    ao_normalization_[right_ao_offset + column];
            }
        }
    }

    LibcintVectorShellBlock result;
    result.left_shell = left_shell;
    result.right_shell = right_shell;
    result.left_ao_offset = left_ao_offset;
    result.right_ao_offset = right_ao_offset;
    result.left_ao_count = left_ao_count;
    result.right_ao_count = right_ao_count;
    result.values = std::move(buffer);
    return result;
}

const LibcintShellQuartet& LibcintDirectShellEvaluator::evaluate_two_electron_shell_quartet(
    int shell_i, int shell_j, int shell_k, int shell_l, LibcintShellWorkspace& workspace) const {
    validate_shell_index(shell_i);
    validate_shell_index(shell_j);
    validate_shell_index(shell_k);
    validate_shell_index(shell_l);

    const int ao_offset_i = xmvb::integrals::shell_ao_offset(input_, shell_i);
    const int ao_offset_j = xmvb::integrals::shell_ao_offset(input_, shell_j);
    const int ao_offset_k = xmvb::integrals::shell_ao_offset(input_, shell_k);
    const int ao_offset_l = xmvb::integrals::shell_ao_offset(input_, shell_l);
    const int ao_count_i = xmvb::integrals::shell_ao_count(input_, shell_i);
    const int ao_count_j = xmvb::integrals::shell_ao_count(input_, shell_j);
    const int ao_count_k = xmvb::integrals::shell_ao_count(input_, shell_k);
    const int ao_count_l = xmvb::integrals::shell_ao_count(input_, shell_l);

    auto& result = workspace.quartet;
    auto& buffer = result.values;
    buffer.resize(static_cast<std::size_t>(ao_count_i) * ao_count_j * ao_count_k * ao_count_l);
    FINT shell_quartet[4] = {shell_i, shell_j, shell_k, shell_l};
    const auto status = evaluate_cached(int2e_cart,
                                        buffer,
                                        shell_quartet,
                                        input_.atm.data(),
                                        input_.n_atoms,
                                        input_.bas.data(),
                                        input_.n_shells,
                                        input_.env.data(),
                                        two_electron_optimizer_,
                                        workspace);

    if (status != 0) {
        for (int s = 0; s < ao_count_l; ++s) {
            const double normalization_l = ao_normalization_[ao_offset_l + s];
            for (int r = 0; r < ao_count_k; ++r) {
                const double normalization_kl =
                    normalization_l * ao_normalization_[ao_offset_k + r];
                for (int q = 0; q < ao_count_j; ++q) {
                    const double normalization_jkl =
                        normalization_kl * ao_normalization_[ao_offset_j + q];
                    for (int p = 0; p < ao_count_i; ++p) {
                        const std::size_t index = p + q * ao_count_i + r * ao_count_i * ao_count_j +
                                                  s * ao_count_i * ao_count_j * ao_count_k;
                        buffer[index] *= normalization_jkl * ao_normalization_[ao_offset_i + p];
                    }
                }
            }
        }
    } else {
        std::fill(buffer.begin(), buffer.end(), 0.0);
    }

    result.shell_i = shell_i;
    result.shell_j = shell_j;
    result.shell_k = shell_k;
    result.shell_l = shell_l;
    result.ao_offset_i = ao_offset_i;
    result.ao_offset_j = ao_offset_j;
    result.ao_offset_k = ao_offset_k;
    result.ao_offset_l = ao_offset_l;
    result.ao_count_i = ao_count_i;
    result.ao_count_j = ao_count_j;
    result.ao_count_k = ao_count_k;
    result.ao_count_l = ao_count_l;
    return result;
}

const LibcintShellBlock& LibcintDirectShellEvaluator::evaluate_auxiliary_metric_shell_pair(
    int left_auxiliary_shell, int right_auxiliary_shell, LibcintShellWorkspace& workspace) const {
    validate_auxiliary_shell_index(left_auxiliary_shell);
    validate_auxiliary_shell_index(right_auxiliary_shell);

    const int left_ao_offset =
        xmvb::integrals::shell_ao_offset(*auxiliary_input_, left_auxiliary_shell);
    const int right_ao_offset =
        xmvb::integrals::shell_ao_offset(*auxiliary_input_, right_auxiliary_shell);
    const int left_ao_count =
        xmvb::integrals::shell_ao_count(*auxiliary_input_, left_auxiliary_shell);
    const int right_ao_count =
        xmvb::integrals::shell_ao_count(*auxiliary_input_, right_auxiliary_shell);

    auto& result = workspace.metric;
    auto& buffer = result.values;
    buffer.resize(static_cast<std::size_t>(left_ao_count) * right_ao_count);
    FINT shell_pair[2] = {left_auxiliary_shell, right_auxiliary_shell};
    const auto status = evaluate_cached(int2c2e_cart,
                                        buffer,
                                        shell_pair,
                                        auxiliary_input_->atm.data(),
                                        auxiliary_input_->n_atoms,
                                        auxiliary_input_->bas.data(),
                                        auxiliary_input_->n_shells,
                                        auxiliary_input_->env.data(),
                                        auxiliary_metric_optimizer_,
                                        workspace);

    if (status != 0) {
        for (int column = 0; column < right_ao_count; ++column) {
            const double right_normalization =
                auxiliary_ao_normalization_[right_ao_offset + column];
            for (int row = 0; row < left_ao_count; ++row) {
                const std::size_t index = row + column * left_ao_count;
                buffer[index] *=
                    auxiliary_ao_normalization_[left_ao_offset + row] * right_normalization;
            }
        }
    } else {
        std::fill(buffer.begin(), buffer.end(), 0.0);
    }

    result.left_shell = left_auxiliary_shell;
    result.right_shell = right_auxiliary_shell;
    result.left_ao_offset = left_ao_offset;
    result.right_ao_offset = right_ao_offset;
    result.left_ao_count = left_ao_count;
    result.right_ao_count = right_ao_count;
    return result;
}

const LibcintThreeCenterShellBlock& LibcintDirectShellEvaluator::evaluate_three_center_shell_block(
    int primary_left_shell,
    int primary_right_shell,
    int auxiliary_shell,
    LibcintShellWorkspace& workspace) const {
    validate_shell_index(primary_left_shell);
    validate_shell_index(primary_right_shell);
    validate_auxiliary_shell_index(auxiliary_shell);

    const int combined_auxiliary_shell = input_.n_shells + auxiliary_shell;
    const int primary_left_ao_offset = xmvb::integrals::shell_ao_offset(input_, primary_left_shell);
    const int primary_right_ao_offset =
        xmvb::integrals::shell_ao_offset(input_, primary_right_shell);
    const int auxiliary_ao_offset =
        xmvb::integrals::shell_ao_offset(*auxiliary_input_, auxiliary_shell);
    const int primary_left_ao_count = xmvb::integrals::shell_ao_count(input_, primary_left_shell);
    const int primary_right_ao_count = xmvb::integrals::shell_ao_count(input_, primary_right_shell);
    const int auxiliary_ao_count =
        xmvb::integrals::shell_ao_count(*auxiliary_input_, auxiliary_shell);

    auto& result = workspace.three_center;
    auto& buffer = result.values;
    buffer.resize(static_cast<std::size_t>(primary_left_ao_count) * primary_right_ao_count *
                  auxiliary_ao_count);
    FINT shell_triple[3] = {
        primary_left_shell,
        primary_right_shell,
        combined_auxiliary_shell,
    };
    const auto status = evaluate_cached(int3c2e_cart,
                                        buffer,
                                        shell_triple,
                                        combined_atm_.data(),
                                        input_.n_atoms,
                                        combined_bas_.data(),
                                        input_.n_shells + auxiliary_input_->n_shells,
                                        combined_env_.data(),
                                        three_center_optimizer_,
                                        workspace);

    if (status != 0) {
        for (int auxiliary_local = 0; auxiliary_local < auxiliary_ao_count; ++auxiliary_local) {
            const double auxiliary_normalization =
                auxiliary_ao_normalization_[auxiliary_ao_offset + auxiliary_local];
            for (int right_local = 0; right_local < primary_right_ao_count; ++right_local) {
                const double right_normalization =
                    ao_normalization_[primary_right_ao_offset + right_local];
                for (int left_local = 0; left_local < primary_left_ao_count; ++left_local) {
                    const std::size_t index =
                        left_local + right_local * primary_left_ao_count +
                        auxiliary_local * primary_left_ao_count * primary_right_ao_count;
                    buffer[index] *= ao_normalization_[primary_left_ao_offset + left_local] *
                                     right_normalization * auxiliary_normalization;
                }
            }
        }
    } else {
        std::fill(buffer.begin(), buffer.end(), 0.0);
    }

    result.primary_left_shell = primary_left_shell;
    result.primary_right_shell = primary_right_shell;
    result.auxiliary_shell = auxiliary_shell;
    result.primary_left_ao_offset = primary_left_ao_offset;
    result.primary_right_ao_offset = primary_right_ao_offset;
    result.auxiliary_ao_offset = auxiliary_ao_offset;
    result.primary_left_ao_count = primary_left_ao_count;
    result.primary_right_ao_count = primary_right_ao_count;
    result.auxiliary_ao_count = auxiliary_ao_count;
    return result;
}

double LibcintDirectShellEvaluator::evaluate_two_electron_shell_pair_bound(
    int shell_i, int shell_j, LibcintShellWorkspace& workspace) const {
    validate_shell_index(shell_i);
    validate_shell_index(shell_j);

    const int ao_count_i = xmvb::integrals::shell_ao_count(input_, shell_i);
    const int ao_count_j = xmvb::integrals::shell_ao_count(input_, shell_j);
    const int offset_i = xmvb::integrals::shell_ao_offset(input_, shell_i);
    const int offset_j = xmvb::integrals::shell_ao_offset(input_, shell_j);
    auto& buffer = workspace.quartet.values;
    buffer.resize(static_cast<std::size_t>(ao_count_i) * ao_count_j * ao_count_i * ao_count_j);
    FINT shell_quartet[4] = {shell_i, shell_j, shell_i, shell_j};
    const auto status = evaluate_cached(int2e_cart,
                                        buffer,
                                        shell_quartet,
                                        input_.atm.data(),
                                        input_.n_atoms,
                                        input_.bas.data(),
                                        input_.n_shells,
                                        input_.env.data(),
                                        two_electron_optimizer_,
                                        workspace);
    if (status == 0) {
        return 0.0;
    }

    double bound = 0.0;
    for (int j = 0; j < ao_count_j; ++j) {
        for (int i = 0; i < ao_count_i; ++i) {
            const std::size_t index =
                static_cast<std::size_t>(i) +
                static_cast<std::size_t>(ao_count_i) *
                    (static_cast<std::size_t>(j) +
                     static_cast<std::size_t>(ao_count_j) *
                         (static_cast<std::size_t>(i) + static_cast<std::size_t>(ao_count_i) * j));
            const double scale = ao_normalization_[offset_i + i] * ao_normalization_[offset_j + j];
            bound = std::max(bound, std::abs(buffer[index]) * scale * scale);
        }
    }
    return bound;
}

} // namespace xmvb::libcint
