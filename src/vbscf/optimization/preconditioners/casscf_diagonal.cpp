#include "vbscf/optimization/preconditioners/casscf_diagonal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <Eigen/Cholesky>

#include "vbscf/core/contracts/input.hpp"
#include "vbscf/core/contracts/orbital_type.hpp"
#include "vbscf/derivatives/hessian/context/accepted_point.hpp"
#include "vbscf/integrals/active/two_electron/construction/indexer.hpp"
#include "vbscf/integrals/active/two_electron/response/types.hpp"
#include "vbscf/integrals/active/two_electron/transformation/ao_pair_operator.hpp"
#include "vbscf/integrals/active/two_electron/transformation/packed_pair_map.hpp"
#include "vbscf/integrals/ao/ri/factorization.hpp"

namespace xmvb::vb {
namespace {

std::size_t tensor4_index(
    int p, int q, int r, int s, int dimension) noexcept {
  return (((static_cast<std::size_t>(p) * dimension + q) * dimension + r) *
          dimension + s);
}

std::size_t ppaa_index(
    int p, int q, int a, int b, int n_mo, int n_active) noexcept {
  return (((static_cast<std::size_t>(p) * n_mo + q) * n_active + a) *
          n_active + b);
}

std::size_t papa_index(
    int p, int a, int q, int b, int n_mo, int n_active) noexcept {
  return (((static_cast<std::size_t>(p) * n_active + a) * n_mo + q) *
          n_active + b);
}

std::size_t packed_eri_count(int n_active) noexcept {
  const std::size_t n_pairs =
      static_cast<std::size_t>(n_active) * (n_active + 1) / 2;
  return n_pairs * (n_pairs + 1) / 2;
}

void validate(
    const CasscfDiagonalIntermediates& x,
    const Eigen::Ref<const Eigen::MatrixXd>& dm1,
    const std::vector<double>& packed_gradient) {
  const int n_mo = static_cast<int>(x.h_core.rows());
  if (x.n_core < 0 || x.n_active <= 0 || n_mo <= x.n_core + x.n_active ||
      x.h_core.cols() != n_mo || x.v_core.rows() != n_mo ||
      x.v_core.cols() != n_mo || x.j_pc.rows() != n_mo ||
      x.j_pc.cols() != x.n_core || x.k_pc.rows() != n_mo ||
      x.k_pc.cols() != x.n_core || dm1.rows() != x.n_active ||
      dm1.cols() != x.n_active ||
      x.ppaa.size() != static_cast<std::size_t>(n_mo) * n_mo *
                             x.n_active * x.n_active ||
      x.papa.size() != static_cast<std::size_t>(n_mo) * x.n_active *
                             n_mo * x.n_active ||
      packed_gradient.size() != packed_eri_count(x.n_active) ||
      !x.h_core.allFinite() || !x.v_core.allFinite() ||
      !x.j_pc.allFinite() || !x.k_pc.allFinite() || !dm1.allFinite()) {
    throw std::invalid_argument("invalid CASSCF Hessian-diagonal input");
  }
  if (!std::all_of(x.ppaa.begin(), x.ppaa.end(), [](double value) {
        return std::isfinite(value);
      }) ||
      !std::all_of(x.papa.begin(), x.papa.end(), [](double value) {
        return std::isfinite(value);
      }) ||
      !std::all_of(packed_gradient.begin(), packed_gradient.end(), [](double value) {
        return std::isfinite(value);
      })) {
    throw std::invalid_argument("non-finite CASSCF Hessian-diagonal tensor");
  }
}

ExactCtxPairMatrix build_mixed_pair_map(
    const Eigen::Ref<const Eigen::MatrixXd>& left,
    const Eigen::Ref<const Eigen::MatrixXd>& right) {
  if (left.rows() != right.rows()) {
    throw std::invalid_argument("mixed pair maps require common AO rows");
  }
  const int n_bf = static_cast<int>(left.rows());
  const int n_left = static_cast<int>(left.cols());
  const int n_right = static_cast<int>(right.cols());
  ExactCtxPairMatrix result(
      static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2,
      n_left * n_right);
  Eigen::Index pair = 0;
  for (int mu = 0; mu < n_bf; ++mu) {
    for (int nu = 0; nu <= mu; ++nu, ++pair) {
      for (int p = 0; p < n_left; ++p) {
        for (int q = 0; q < n_right; ++q) {
          double value = left(mu, p) * right(nu, q);
          if (mu != nu) value += left(nu, p) * right(mu, q);
          result(pair, p * n_right + q) = value;
        }
      }
    }
  }
  return result;
}

Eigen::MatrixXd build_pair_transform(
    const Eigen::Ref<const Eigen::MatrixXd>& orbital_transform) {
  const int n = static_cast<int>(orbital_transform.rows());
  if (n <= 0 || orbital_transform.cols() != n) {
    throw std::invalid_argument("orbital pair transform must be square");
  }
  const int n_pairs = n * (n + 1) / 2;
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_pairs, n_pairs);
  for (int new_p = 0; new_p < n; ++new_p) {
    for (int new_q = 0; new_q <= new_p; ++new_q) {
      const int new_pair = TwoElectronIndexer::packed_pair_index(new_p, new_q);
      for (int old_p = 0; old_p < n; ++old_p) {
        for (int old_q = 0; old_q <= old_p; ++old_q) {
          const int old_pair = TwoElectronIndexer::packed_pair_index(old_p, old_q);
          double value =
              orbital_transform(old_p, new_p) *
              orbital_transform(old_q, new_q);
          if (old_p != old_q) {
            value +=
                orbital_transform(old_q, new_p) *
                orbital_transform(old_p, new_q);
          }
          result(old_pair, new_pair) = value;
        }
      }
    }
  }
  return result;
}

ExactCtxPairMatrix apply_ao_pair_operator(
    const VbScfInput& input,
    const Eigen::Ref<const ExactCtxPairMatrix>& coefficients) {
  ExactCtxPairMatrix result;
  const int n_bf = input.orbital_preparation_input.n_basis_functions;
  if (input.standard_two_electron_mode ==
      StandardTwoElectronMode::ResolutionOfIdentity) {
    if (!input.ri_factorization) {
      throw std::invalid_argument("RI diagonal requires accepted AO factors");
    }
    const auto& factors = input.ri_factorization->metric_whitened_ao_pair_factors;
    result.noalias() = factors.transpose() * (factors * coefficients);
  } else {
    detail::apply_exact_ao_pair_kernel(
        input.ao_integral_input,
        coefficients,
        n_bf,
        static_cast<std::size_t>(coefficients.cols()),
        &result);
  }
  return result;
}

std::vector<double> transform_dm2(
    const std::vector<double>& source,
    const Eigen::Ref<const Eigen::MatrixXd>& transform) {
  const int n = static_cast<int>(transform.rows());
  std::vector<double> current = source;
  std::vector<double> next(current.size());
  for (int axis = 0; axis < 4; ++axis) {
    std::fill(next.begin(), next.end(), 0.0);
    for (int p = 0; p < n; ++p)
      for (int q = 0; q < n; ++q)
        for (int r = 0; r < n; ++r)
          for (int s = 0; s < n; ++s) {
            const int output[4] = {p, q, r, s};
            double value = 0.0;
            for (int old = 0; old < n; ++old) {
              int input_index[4] = {p, q, r, s};
              input_index[axis] = old;
              value += transform(output[axis], old) * current[tensor4_index(
                  input_index[0], input_index[1], input_index[2],
                  input_index[3], n)];
            }
            next[tensor4_index(p, q, r, s, n)] = value;
          }
    current.swap(next);
  }
  return current;
}

std::vector<double> pack_symmetric_dm2(
    const std::vector<double>& dm2,
    int n) {
  std::vector<double> packed(packed_eri_count(n), 0.0);
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q <= p; ++q) {
      const int pq = TwoElectronIndexer::packed_pair_index(p, q);
      for (int r = 0; r < n; ++r) {
        for (int s = 0; s <= r; ++s) {
          const int rs = TwoElectronIndexer::packed_pair_index(r, s);
          if (rs > pq) continue;
          const int multiplicity =
              (p == q ? 1 : 2) * (r == s ? 1 : 2) * (pq == rs ? 1 : 2);
          packed[TwoElectronIndexer::packed_pair_of_pairs_index(pq, rs)] =
              0.5 * multiplicity * dm2[tensor4_index(p, q, r, s, n)];
        }
      }
    }
  }
  return packed;
}

}  // namespace

std::vector<double> unpack_symmetric_active_two_rdm(
    const std::vector<double>& packed_eri_gradient,
    int n_active) {
  if (n_active <= 0 || packed_eri_gradient.size() != packed_eri_count(n_active)) {
    throw std::invalid_argument("invalid packed active ERI gradient");
  }
  std::vector<double> dm2(
      static_cast<std::size_t>(n_active) * n_active * n_active * n_active,
      0.0);
  for (int p = 0; p < n_active; ++p) {
    for (int q = 0; q < n_active; ++q) {
      const int pq = TwoElectronIndexer::packed_pair_index(p, q);
      for (int r = 0; r < n_active; ++r) {
        for (int s = 0; s < n_active; ++s) {
          const int rs = TwoElectronIndexer::packed_pair_index(r, s);
          const int packed =
              TwoElectronIndexer::packed_pair_of_pairs_index(pq, rs);
          const int multiplicity =
              (p == q ? 1 : 2) * (r == s ? 1 : 2) * (pq == rs ? 1 : 2);
          dm2[tensor4_index(p, q, r, s, n_active)] =
              2.0 * packed_eri_gradient[packed] / multiplicity;
        }
      }
    }
  }
  return dm2;
}

Eigen::MatrixXd build_casscf_orbital_hessian_diagonal(
    const CasscfDiagonalIntermediates& x,
    const Eigen::Ref<const Eigen::MatrixXd>& dm1_active,
    const std::vector<double>& packed_gradient) {
  validate(x, dm1_active, packed_gradient);
  const int n_mo = static_cast<int>(x.h_core.rows());
  const int n_core = x.n_core;
  const int n_active = x.n_active;
  const int n_occ = n_core + n_active;
  const std::vector<double> dm2 =
      unpack_symmetric_active_two_rdm(packed_gradient, n_active);

  Eigen::MatrixXd dm1 = Eigen::MatrixXd::Zero(n_mo, n_mo);
  dm1.diagonal().head(n_core).setConstant(2.0);
  dm1.block(n_core, n_core, n_active, n_active) = dm1_active;

  Eigen::MatrixXd v_active = Eigen::MatrixXd::Zero(n_mo, n_mo);
  for (int p = 0; p < n_mo; ++p) {
    for (int q = 0; q < n_mo; ++q) {
      double value = 0.0;
      for (int a = 0; a < n_active; ++a) {
        for (int b = 0; b < n_active; ++b) {
          value +=
              x.ppaa[ppaa_index(p, q, a, b, n_mo, n_active)] *
                  dm1_active(a, b) -
              0.5 * x.papa[papa_index(p, a, q, b, n_mo, n_active)] *
                  dm1_active(a, b);
        }
      }
      v_active(p, q) = value;
    }
  }
  const Eigen::MatrixXd v_total = x.v_core + v_active;

  Eigen::MatrixXd gpq = Eigen::MatrixXd::Zero(n_mo, n_mo);
  gpq.leftCols(n_core) =
      2.0 * (x.h_core.leftCols(n_core) + v_total.leftCols(n_core));
  gpq.middleCols(n_core, n_active).noalias() =
      (x.h_core.middleCols(n_core, n_active) +
       x.v_core.middleCols(n_core, n_active)) * dm1_active;

  Eigen::MatrixXd jkcaa = Eigen::MatrixXd::Zero(n_occ, n_active);
  Eigen::MatrixXd v_dm2 = Eigen::MatrixXd::Zero(n_mo, n_active);
  for (int p = 0; p < n_mo; ++p) {
    for (int a = 0; a < n_active; ++a) {
      double gradient_dm2 = 0.0;
      double diagonal_dm2 = 0.0;
      for (int u = 0; u < n_active; ++u) {
        for (int w = 0; w < n_active; ++w) {
          for (int z = 0; z < n_active; ++z) {
            gradient_dm2 +=
                x.ppaa[ppaa_index(
                    p, n_core + u, w, z, n_mo, n_active)] *
                dm2[tensor4_index(w, z, u, a, n_active)];
          }
        }
      }
      gpq(p, n_core + a) += gradient_dm2;

      for (int w = 0; w < n_active; ++w) {
        for (int z = 0; z < n_active; ++z) {
          diagonal_dm2 +=
              x.ppaa[ppaa_index(p, p, w, z, n_mo, n_active)] *
              dm2[tensor4_index(w, z, a, a, n_active)];
          diagonal_dm2 +=
              x.papa[papa_index(p, w, p, z, n_mo, n_active)] *
              (dm2[tensor4_index(a, w, z, a, n_active)] +
               dm2[tensor4_index(w, a, z, a, n_active)]);
        }
      }
      v_dm2(p, a) = diagonal_dm2;
      if (p < n_occ) {
        double value = 0.0;
        for (int v = 0; v < n_active; ++v) {
          value +=
              (6.0 * x.papa[papa_index(p, a, p, v, n_mo, n_active)] -
               2.0 * x.ppaa[ppaa_index(p, p, a, v, n_mo, n_active)]) *
              dm1_active(a, v);
        }
        jkcaa(p, a) = value;
      }
    }
  }

  Eigen::MatrixXd diagonal = Eigen::MatrixXd::Zero(n_mo, n_mo);
  for (int p = 0; p < n_mo; ++p) {
    for (int q = 0; q < n_mo; ++q) {
      diagonal(p, q) =
          x.h_core(p, p) * dm1(q, q) - x.h_core(p, q) * dm1(p, q) +
          x.h_core(q, q) * dm1(p, p) - x.h_core(q, p) * dm1(q, p) -
          gpq(p, p) - gpq(q, q);
    }
    diagonal(p, p) += 2.0 * gpq(p, p);
  }

  const Eigen::VectorXd v_total_diagonal = v_total.diagonal();
  for (int p = 0; p < n_mo; ++p) {
    for (int c = 0; c < n_core; ++c) {
      diagonal(p, c) += 2.0 * v_total_diagonal(p);
      diagonal(c, p) += 2.0 * v_total_diagonal(p);
    }
  }
  for (int c = 0; c < n_core; ++c) {
    diagonal(c, c) -= 4.0 * v_total_diagonal(c);
  }

  for (int p = 0; p < n_mo; ++p) {
    for (int a = 0; a < n_active; ++a) {
      const int active = n_core + a;
      const double core_dm1 = x.v_core(p, p) * dm1_active(a, a);
      diagonal(p, active) += core_dm1;
      diagonal(active, p) += core_dm1;
    }
  }
  for (int a = 0; a < n_active; ++a) {
    for (int b = 0; b < n_active; ++b) {
      const double value = -x.v_core(n_core + a, n_core + b) *
                           dm1_active(a, b);
      diagonal(n_core + a, n_core + b) += value;
      diagonal(n_core + b, n_core + a) += value;
    }
  }

  for (int p = n_core; p < n_mo; ++p) {
    for (int c = 0; c < n_core; ++c) {
      const double value = 6.0 * x.k_pc(p, c) - 2.0 * x.j_pc(p, c);
      diagonal(p, c) += value;
      diagonal(c, p) += value;
    }
  }
  for (int p = 0; p < n_occ; ++p) {
    for (int a = 0; a < n_active; ++a) {
      diagonal(p, n_core + a) -= jkcaa(p, a);
      diagonal(n_core + a, p) -= jkcaa(p, a);
    }
  }
  for (int p = 0; p < n_mo; ++p) {
    for (int a = 0; a < n_active; ++a) {
      diagonal(n_core + a, p) += v_dm2(p, a);
      diagonal(p, n_core + a) += v_dm2(p, a);
    }
  }

  diagonal *= 2.0;
  diagonal = (0.5 * (diagonal + diagonal.transpose())).eval();
  if (!diagonal.allFinite()) {
    throw std::runtime_error("non-finite CASSCF orbital Hessian diagonal");
  }
  return diagonal;
}

OeoCasscfPreconditioner build_oeo_casscf_preconditioner(
    const VbScfInput& input,
    const AcceptedPointContext& accepted) {
  const auto& orbital_input = input.orbital_preparation_input;
  const auto& prepared = accepted.prepared_active_space;
  const auto& orbitals = prepared.orbital_result;
  const int n_bf = orbital_input.n_basis_functions;
  const int n_core = prepared.n_inactive_doubly_occupied_orbitals;
  const int n_active = accepted.n_active_orbitals;
  const int n_occ = n_core + n_active;
  if (!input.complete_active_space ||
      orbital_input.orbital_type != kOrbitalTypeOeo || n_bf <= n_occ ||
      orbitals.auxiliary_orbital_matrix.rows() != n_bf ||
      orbitals.auxiliary_orbital_matrix.cols() != n_bf ||
      accepted.active_one_electron_gradient.size() !=
          static_cast<std::size_t>(n_active) * n_active) {
    throw std::invalid_argument(
        "CASSCF rotation diagonal requires a complete full-AO OEO point");
  }
  const Eigen::Map<const Eigen::MatrixXd> overlap(
      orbital_input.ao_overlap_matrix.data(), n_bf, n_bf);
  const Eigen::Map<const Eigen::MatrixXd> active_overlap(
      orbitals.active_orbital_overlap_matrix.data(), n_active, n_active);
  Eigen::LLT<Eigen::MatrixXd> active_llt(active_overlap);
  if (active_llt.info() != Eigen::Success) {
    throw std::runtime_error("active overlap factorization failed");
  }
  const Eigen::MatrixXd active_transform = active_llt.matrixU();
  const Eigen::MatrixXd inverse_active_transform =
      active_transform.template triangularView<Eigen::Upper>().solve(
          Eigen::MatrixXd::Identity(n_active, n_active));

  Eigen::MatrixXd mo = orbitals.auxiliary_orbital_matrix;
  mo.leftCols(n_core) =
      orbitals.physical_orbital_frame.inactive_orthonormal_orbital_matrix;
  mo.middleCols(n_core, n_active) =
      orbitals.auxiliary_orbital_matrix.middleCols(n_core, n_active) *
      inverse_active_transform;
  if ((mo.transpose() * overlap * mo -
       Eigen::MatrixXd::Identity(n_bf, n_bf)).cwiseAbs().maxCoeff() > 2.0e-8) {
    throw std::runtime_error("CASSCF MO frame is not AO-metric orthonormal");
  }

  const Eigen::Map<const Eigen::MatrixXd> original_dm1(
      accepted.active_one_electron_gradient.data(), n_active, n_active);
  const Eigen::MatrixXd dm1 =
      active_transform * original_dm1 * active_transform.transpose();
  const std::vector<double> packed_dm2 = pack_symmetric_dm2(
      transform_dm2(
          unpack_symmetric_active_two_rdm(
              accepted.packed_active_two_electron_gradient, n_active),
          active_transform),
      n_active);

  CasscfDiagonalIntermediates x;
  x.n_core = n_core;
  x.n_active = n_active;
  x.h_core = mo.transpose() *
      input.ao_integral_input.ao_core_hamiltonian_matrix * mo;
  x.v_core = mo.transpose() *
      prepared.ao_effective_one_electron_result.ao_coulomb_exchange_matrix * mo;

  const auto& active_two_electron = prepared.active_space_two_electron_result;
  const Eigen::MatrixXd active_pair_transform =
      build_pair_transform(inverse_active_transform);
  ExactCtxPairMatrix active_pair_products;
  if (active_two_electron.representation ==
      ActiveSpaceTwoElectronRepresentation::ResolutionOfIdentity) {
    if (active_two_electron.ri_active_pair_factors.size() == 0 ||
        !input.ri_factorization) {
      throw std::invalid_argument(
          "RI CASSCF diagonal requires accepted active-pair factors");
    }
    const Eigen::MatrixXd orthogonal_active_factors =
        active_two_electron.ri_active_pair_factors * active_pair_transform;
    active_pair_products.noalias() =
        input.ri_factorization->metric_whitened_ao_pair_factors.transpose() *
        orthogonal_active_factors;
  } else if (active_two_electron.dense_ao_pair_products.size() != 0) {
    active_pair_products.noalias() =
        active_two_electron.dense_ao_pair_products * active_pair_transform;
  } else {
    // The memory-bounded exact forward path deliberately does not retain
    // G*Q_active. Rebuild only that absent accepted tensor from the AO graph.
    ExactCtxPairMatrix active_pair_map;
    build_packed_orbital_pair_map(
        mo.middleCols(n_core, n_active), &active_pair_map);
    active_pair_products = apply_ao_pair_operator(input, active_pair_map);
  }
  const int n_mo_pairs = n_bf * (n_bf + 1) / 2;
  const int n_active_pairs = n_active * (n_active + 1) / 2;
  Eigen::MatrixXd ppaa_pairs = Eigen::MatrixXd::Zero(
      n_mo_pairs, n_active_pairs);
#pragma omp parallel for schedule(static)
  for (int p = 0; p < n_bf; ++p) {
    for (int q = 0; q <= p; ++q) {
      const int pq = TwoElectronIndexer::packed_pair_index(p, q);
      Eigen::VectorXd pair_coefficients(active_pair_products.rows());
      Eigen::Index ao_pair = 0;
      for (int mu = 0; mu < n_bf; ++mu) {
        for (int nu = 0; nu <= mu; ++nu, ++ao_pair) {
          double coefficient = mo(mu, p) * mo(nu, q);
          if (mu != nu) coefficient += mo(nu, p) * mo(mu, q);
          pair_coefficients[ao_pair] = coefficient;
        }
      }
      ppaa_pairs.row(pq).noalias() =
          pair_coefficients.transpose() * active_pair_products;
    }
  }
  x.ppaa.resize(
      static_cast<std::size_t>(n_bf) * n_bf * n_active * n_active);
  for (int p = 0; p < n_bf; ++p)
    for (int q = 0; q < n_bf; ++q) {
      const int pq = TwoElectronIndexer::packed_pair_index(p, q);
      for (int a = 0; a < n_active; ++a)
        for (int b = 0; b < n_active; ++b) {
          const int ab = TwoElectronIndexer::packed_pair_index(a, b);
          x.ppaa[ppaa_index(p, q, a, b, n_bf, n_active)] =
              ppaa_pairs(pq, ab);
        }
    }
  x.papa.resize(
      static_cast<std::size_t>(n_bf) * n_active * n_bf * n_active);
  const int transform_block_width = std::max(1, 2 * n_active);
  for (int p_begin = 0; p_begin < n_bf; p_begin += transform_block_width) {
    const int p_count = std::min(transform_block_width, n_bf - p_begin);
    const ExactCtxPairMatrix block_map = build_mixed_pair_map(
        mo.middleCols(p_begin, p_count),
        mo.middleCols(n_core, n_active));
    const ExactCtxPairMatrix block_products =
        apply_ao_pair_operator(input, block_map);
    for (int q = 0; q < n_bf; ++q) {
      Eigen::MatrixXd contractions = Eigen::MatrixXd::Zero(
          n_active, p_count * n_active);
      Eigen::Index pair = 0;
      for (int mu = 0; mu < n_bf; ++mu) {
        for (int nu = 0; nu <= mu; ++nu, ++pair) {
          for (int b = 0; b < n_active; ++b) {
            double coefficient = mo(mu, q) * mo(nu, n_core + b);
            if (mu != nu) {
              coefficient += mo(nu, q) * mo(mu, n_core + b);
            }
            contractions.row(b).noalias() +=
                coefficient * block_products.row(pair);
          }
        }
      }
      for (int local_p = 0; local_p < p_count; ++local_p)
        for (int a = 0; a < n_active; ++a)
          for (int b = 0; b < n_active; ++b)
            x.papa[papa_index(
                p_begin + local_p, a, q, b, n_bf, n_active)] =
                contractions(b, local_p * n_active + a);
    }
  }
  ExactCtxPairMatrix core_diagonal_map(
      static_cast<Eigen::Index>(n_bf) * (n_bf + 1) / 2, n_core);
  Eigen::Index ao_pair = 0;
  for (int mu = 0; mu < n_bf; ++mu)
    for (int nu = 0; nu <= mu; ++nu, ++ao_pair)
      for (int c = 0; c < n_core; ++c)
        core_diagonal_map(ao_pair, c) =
            mo(mu, c) * mo(nu, c) * (mu == nu ? 1.0 : 2.0);
  const ExactCtxPairMatrix core_diagonal_products =
      apply_ao_pair_operator(input, core_diagonal_map);
  x.j_pc = Eigen::MatrixXd::Zero(n_bf, n_core);
  for (int p = 0; p < n_bf; ++p) {
    Eigen::VectorXd diagonal_pair(core_diagonal_map.rows());
    ao_pair = 0;
    for (int mu = 0; mu < n_bf; ++mu)
      for (int nu = 0; nu <= mu; ++nu, ++ao_pair)
        diagonal_pair[ao_pair] =
            mo(mu, p) * mo(nu, p) * (mu == nu ? 1.0 : 2.0);
    x.j_pc.row(p) = diagonal_pair.transpose() * core_diagonal_products;
  }

  x.k_pc = Eigen::MatrixXd::Zero(n_bf, n_core);
  for (int p_begin = 0; p_begin < n_bf; p_begin += transform_block_width) {
    const int p_count = std::min(transform_block_width, n_bf - p_begin);
    const ExactCtxPairMatrix block_map = build_mixed_pair_map(
        mo.middleCols(p_begin, p_count), mo.leftCols(n_core));
    const ExactCtxPairMatrix block_products =
        apply_ao_pair_operator(input, block_map);
    for (int local_p = 0; local_p < p_count; ++local_p)
      for (int c = 0; c < n_core; ++c) {
        const int column = local_p * n_core + c;
        x.k_pc(p_begin + local_p, c) =
            block_map.col(column).dot(block_products.col(column));
      }
  }
  OeoCasscfPreconditioner result;
  result.mo_coefficients = std::move(mo);
  result.rotation_diagonal =
      build_casscf_orbital_hessian_diagonal(x, dm1, packed_dm2);
  result.inactive_inverse_transform =
      orbitals.physical_orbital_frame
          .localized_representative_selector
          .inactive_inverse_transpose_right_transform.transpose();
  result.active_inverse_transform = inverse_active_transform;
  return result;
}

}  // namespace xmvb::vb
