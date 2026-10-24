#pragma once

// Tiny, deliberately dense test oracle. Occupations are sorted integer lists;
// signs come from list insertion/erasure positions, independently of the
// production bit-string links and parity implementation.

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

namespace spinor_reference {

using Occupation = std::vector<int>;
using Basis = std::vector<Occupation>;
using Complex = std::complex<double>;

inline void
append_occupations(int n_spinor, int n_electrons, int next, Occupation& current, Basis& result) {
    if (static_cast<int>(current.size()) == n_electrons) {
        result.push_back(current);
        return;
    }
    const int remaining = n_electrons - static_cast<int>(current.size());
    for (int p = next; p <= n_spinor - remaining; ++p) {
        current.push_back(p);
        append_occupations(n_spinor, n_electrons, p + 1, current, result);
        current.pop_back();
    }
}

inline Basis basis(int n_spinor, int n_electrons) {
    if (n_spinor < 0 || n_spinor > 64 || n_electrons < 0 || n_electrons > n_spinor) {
        throw std::invalid_argument("invalid reference basis dimensions");
    }
    Basis result;
    Occupation current;
    append_occupations(n_spinor, n_electrons, 0, current, result);
    return result;
}

inline std::uint64_t mask(const Occupation& occupation) {
    std::uint64_t result = 0;
    for (int p : occupation) {
        result |= std::uint64_t{1} << p;
    }
    return result;
}

inline Occupation occupation(std::uint64_t determinant) {
    Occupation result;
    for (int p = 0; p < 64; ++p) {
        if ((determinant & (std::uint64_t{1} << p)) != 0) {
            result.push_back(p);
        }
    }
    return result;
}

inline int find(const Basis& determinants, const Occupation& target) {
    const auto found = std::find(determinants.begin(), determinants.end(), target);
    return found == determinants.end() ? -1 : static_cast<int>(found - determinants.begin());
}

inline bool annihilate(Occupation& state, int orbital, int& sign) {
    const auto found = std::lower_bound(state.begin(), state.end(), orbital);
    if (found == state.end() || *found != orbital) {
        return false;
    }
    if ((found - state.begin()) % 2 != 0) {
        sign = -sign;
    }
    state.erase(found);
    return true;
}

inline bool create(Occupation& state, int orbital, int& sign) {
    const auto position = std::lower_bound(state.begin(), state.end(), orbital);
    if (position != state.end() && *position == orbital) {
        return false;
    }
    if ((position - state.begin()) % 2 != 0) {
        sign = -sign;
    }
    state.insert(position, orbital);
    return true;
}

inline std::size_t eri_index(int n, int p, int q, int r, int s) {
    return ((static_cast<std::size_t>(p) * n + q) * n + r) * n + s;
}

inline Eigen::MatrixXcd one_body_matrix(const Basis& determinants, const Eigen::MatrixXcd& h) {
    const int n = static_cast<int>(h.rows());
    if (h.cols() != n) {
        throw std::invalid_argument("one-body reference matrix must be square");
    }
    const int d = static_cast<int>(determinants.size());
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(d, d);
    for (int source = 0; source < d; ++source) {
        for (int q = 0; q < n; ++q) {
            for (int p = 0; p < n; ++p) {
                Occupation state = determinants[source];
                int sign = 1;
                if (!annihilate(state, q, sign) || !create(state, p, sign)) {
                    continue;
                }
                const int target = find(determinants, state);
                if (target < 0) {
                    throw std::logic_error("reference one-body target absent");
                }
                result(target, source) += static_cast<double>(sign) * h(p, q);
            }
        }
    }
    return result;
}

inline Eigen::MatrixXcd
two_body_matrix(const Basis& determinants, int n, const std::vector<Complex>& eri) {
    const std::size_t n4 = static_cast<std::size_t>(n) * n * n * n;
    if (eri.size() != n4) {
        throw std::invalid_argument("two-body reference tensor has wrong size");
    }
    const int d = static_cast<int>(determinants.size());
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(d, d);
    for (int source = 0; source < d; ++source) {
        for (int p = 0; p < n; ++p) {
            for (int q = 0; q < n; ++q) {
                for (int r = 0; r < n; ++r) {
                    for (int s = 0; s < n; ++s) {
                        Occupation state = determinants[source];
                        int sign = 1;
                        // Rightmost operator acts first: a_q, a_s, a_r^+, a_p^+.
                        if (!annihilate(state, q, sign) || !annihilate(state, s, sign) ||
                            !create(state, r, sign) || !create(state, p, sign)) {
                            continue;
                        }
                        const int target = find(determinants, state);
                        if (target < 0) {
                            throw std::logic_error("reference two-body target absent");
                        }
                        result(target, source) +=
                            0.5 * static_cast<double>(sign) * eri[eri_index(n, p, q, r, s)];
                    }
                }
            }
        }
    }
    return result;
}

inline Eigen::MatrixXcd dense_hamiltonian(const Basis& determinants,
                                          const Eigen::MatrixXcd& h,
                                          const std::vector<Complex>& eri) {
    return one_body_matrix(determinants, h) +
           two_body_matrix(determinants, static_cast<int>(h.rows()), eri);
}

inline Eigen::MatrixXcd deterministic_complex(int rows, int columns) {
    Eigen::MatrixXcd result(rows, columns);
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            result(row, column) = Complex{std::sin(0.23 * (row + 1) * (column + 2)),
                                          0.37 * std::cos(0.19 * (row + 2) * (column + 1))};
        }
    }
    return result;
}

// The masks select only the output ordering. All matrix elements are still
// generated with the independent occupied-list basis and operator algebra.
inline Eigen::MatrixXcd dense_hamiltonian(const std::vector<std::uint64_t>& masks,
                                          int n_spinor,
                                          const Eigen::MatrixXcd& h,
                                          const std::vector<Complex>& eri) {
    if (h.rows() != n_spinor || h.cols() != n_spinor || masks.empty()) {
        throw std::invalid_argument("dense reference dimensions");
    }
    const int n_electrons = static_cast<int>(occupation(masks.front()).size());
    const Basis independent = basis(n_spinor, n_electrons);
    const Eigen::MatrixXcd dense = dense_hamiltonian(independent, h, eri);
    Eigen::MatrixXcd reordered(masks.size(), masks.size());
    for (int row = 0; row < reordered.rows(); ++row) {
        const int independent_row = find(independent, occupation(masks[row]));
        if (independent_row < 0) {
            throw std::invalid_argument("dense reference mask outside basis");
        }
        for (int column = 0; column < reordered.cols(); ++column) {
            const int independent_column = find(independent, occupation(masks[column]));
            if (independent_column < 0) {
                throw std::invalid_argument("dense reference mask outside basis");
            }
            reordered(row, column) = dense(independent_row, independent_column);
        }
    }
    return reordered;
}

// Exterior matrix for a one-particle transform R. The caller supplies the
// determinant order, so production masks can be used without using production
// signs or links. Rows/columns of every minor use ascending occupied indices.
inline Eigen::MatrixXcd wedge_matrix(const Eigen::MatrixXcd& r,
                                     const std::vector<std::uint64_t>& masks) {
    if (r.rows() != r.cols() || r.rows() > 64) {
        throw std::invalid_argument("exterior reference transform dimensions");
    }
    const int d = static_cast<int>(masks.size());
    Eigen::MatrixXcd result(d, d);
    for (int target = 0; target < d; ++target) {
        const Occupation rows = occupation(masks[target]);
        for (int source = 0; source < d; ++source) {
            const Occupation columns = occupation(masks[source]);
            if (rows.size() != columns.size()) {
                throw std::invalid_argument("mixed particle numbers in exterior basis");
            }
            const int n_electrons = static_cast<int>(rows.size());
            if (n_electrons == 0) {
                result(target, source) = 1.0;
                continue;
            }
            Eigen::MatrixXcd minor(n_electrons, n_electrons);
            for (int i = 0; i < n_electrons; ++i) {
                for (int j = 0; j < n_electrons; ++j) {
                    if (rows[i] >= r.rows() || columns[j] >= r.cols()) {
                        throw std::invalid_argument("exterior mask outside spinor range");
                    }
                    minor(i, j) = r(rows[i], columns[j]);
                }
            }
            result(target, source) = minor.determinant();
        }
    }
    return result;
}

} // namespace spinor_reference
