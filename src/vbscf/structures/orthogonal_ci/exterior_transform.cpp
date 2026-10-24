#include "vbscf/structures/orthogonal_ci/exterior_transform.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <iterator>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "parallel/openmp.hpp"
#include "core/timing.hpp"
#include "core/workspace.hpp"
#include "vbscf/structures/orthogonal_ci/exterior_rows.hpp"

namespace xmvb::vb {
namespace {

double replacement_sign(std::uint64_t source, int removed, int inserted) {
    const int low = std::min(removed, inserted);
    const int high = std::max(removed, inserted);
    std::uint64_t between = 0;
    if (high - low > 1) {
        between =
            source & (((std::uint64_t{1} << high) - 1) ^ ((std::uint64_t{1} << (low + 1)) - 1));
    }
    return __builtin_popcountll(between) % 2 == 0 ? 1.0 : -1.0;
}

} // namespace

struct ExteriorOrbitalTransform::RowCache {
    struct TrackedRows {
        workspace::Reservation residency;
        workspace::Temporary temporary;
        std::shared_ptr<const Eigen::MatrixXd> rows;
        TrackedRows(std::shared_ptr<const Eigen::MatrixXd> value,
                    std::size_t bytes,
                    workspace::Reservation reservation)
            : residency(std::move(reservation)), temporary(bytes), rows(std::move(value)) {}
    };
    struct Entry {
        std::uint64_t key = 0;
        std::shared_ptr<const Eigen::MatrixXd> rows;
        std::size_t bytes = 0;
    };

    std::mutex mutex;
    std::list<Entry> entries;
    std::unordered_map<std::uint64_t, std::list<Entry>::iterator> index;
    std::atomic<std::size_t> cached_bytes{0};
    std::size_t budget = 0;

    /** Retain at most the largest requested panel, within shared headroom.
   * A factor cache must not silently retain the full compound matrix as a
   * sweep visits more rows. Borrowed evicted rows retain their own charge.
   */
    void request(std::size_t bytes) {
        const auto available = workspace::available();
        std::lock_guard lock(mutex);
        const auto resident = cached_bytes.load(std::memory_order_relaxed);
        budget = std::min(std::max(budget, bytes), available + resident);
        trim(0);
    }

    void trim(std::size_t incoming) {
        auto resident = cached_bytes.load(std::memory_order_relaxed);
        while (!entries.empty() && (incoming > budget || resident > budget - incoming)) {
            const auto last = std::prev(entries.end());
            resident -= last->bytes;
            index.erase(last->key);
            entries.erase(last);
        }
        cached_bytes.store(resident, std::memory_order_relaxed);
    }

    static std::uint64_t key(int begin, int count) noexcept {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(begin)) << 32) |
               static_cast<std::uint32_t>(count);
    }
    std::shared_ptr<const Eigen::MatrixXd> find(std::uint64_t key) {
        std::lock_guard lock(mutex);
        const auto found = index.find(key);
        if (found == index.end())
            return {};
        entries.splice(entries.begin(), entries, found->second);
        return found->second->rows;
    }
    template <class Build>
    std::shared_ptr<const Eigen::MatrixXd>
    retain(std::uint64_t key, std::size_t bytes, Build&& build) {
        {
            std::lock_guard lock(mutex);
            const auto found = index.find(key);
            if (found != index.end()) {
                entries.splice(entries.begin(), entries, found->second);
                return found->second->rows;
            }
            // Release cold owners before allocating a replacement numerical row.
            trim(bytes);
        }
        workspace::Reservation residency;
        const bool admitted = residency.try_resize(bytes);
        auto rows = build();
        const auto* data = rows.get();
        auto owner = std::make_shared<TrackedRows>(std::move(rows), bytes, std::move(residency));
        std::shared_ptr<const Eigen::MatrixXd> tracked(owner, data);
        std::lock_guard lock(mutex);
        if (!admitted || bytes > budget)
            return tracked;
        const auto found = index.find(key);
        if (found != index.end()) {
            entries.splice(entries.begin(), entries, found->second);
            return found->second->rows;
        }
        trim(bytes);
        // A borrowed row block may outlive eviction. Keep its numerical charge
        // with the shared owner, including when the cache no longer retains it.
        entries.push_front({key, tracked, bytes});
        try {
            index.emplace(key, entries.begin());
        } catch (...) {
            entries.pop_front();
            throw;
        }
        cached_bytes.fetch_add(bytes, std::memory_order_relaxed);
        return tracked;
    }
};

ExteriorOrbitalTransform::ExteriorOrbitalTransform(const mcscf::StringTable& occupations,
                                                   const Eigen::Ref<const Eigen::MatrixXd>& upper)
    : ExteriorOrbitalTransform(occupations.basis(upper.rows()), upper) {}

ExteriorOrbitalTransform::ExteriorOrbitalTransform(
    mcscf::SpinStrings strings, const Eigen::Ref<const Eigen::MatrixXd>& upper_orbital_transform)
    : n_orbitals_(upper_orbital_transform.rows()), n_electrons_(strings.electrons()),
      orbital_transform_(upper_orbital_transform), strings_(std::move(strings)),
      row_cache_(std::make_shared<RowCache>()) {
    profiling::ScopedPhase phase(profiling::Phase::CiTransformBuild);
    if (n_orbitals_ <= 0 || n_orbitals_ > 63 || upper_orbital_transform.cols() != n_orbitals_ ||
        !strings_.complete() || strings_.orbitals() != n_orbitals_ ||
        !upper_orbital_transform.allFinite()) {
        throw std::invalid_argument("invalid exterior orbital transform inputs");
    }
    const double lower_error =
        upper_orbital_transform.template triangularView<Eigen::StrictlyLower>()
            .toDenseMatrix()
            .cwiseAbs()
            .maxCoeff();
    const double scale = std::max(1.0, upper_orbital_transform.cwiseAbs().maxCoeff());
    if (lower_error > 64.0 * std::numeric_limits<double>::epsilon() * scale ||
        (upper_orbital_transform.diagonal().array() <= 0.0).any()) {
        throw std::invalid_argument(
            "exterior orbital transform must be nonsingular upper triangular");
    }

    row_plan_ = std::make_shared<ExteriorRowPlan>(n_orbitals_, n_electrons_, strings_.masks());
    determinant_scales_.resize(static_cast<int>(strings_.masks().size()));
    for (int determinant = 0; determinant < static_cast<int>(strings_.masks().size());
         ++determinant) {
        double scale_factor = 1.0;
        for (const int orbital : strings_.occupied(determinant))
            scale_factor *= upper_orbital_transform(orbital, orbital);
        determinant_scales_[determinant] = scale_factor;
    }

    // R = U D, where D contains the column diagonal and U is unit upper
    // triangular. Reduce U to identity by right-looking column eliminations.
    // Applying the inverses in elimination order evaluates U on a vector
    // because matrix factors act from right to left.
    Eigen::MatrixXd unit_upper = upper_orbital_transform;
    workspace::Temporary unit_upper_temporary(sizeof(double) * unit_upper.size());
    for (int column = 0; column < n_orbitals_; ++column) {
        unit_upper.col(column) /= upper_orbital_transform(column, column);
    }
    struct OrbitalShear {
        int inserted = 0;
        int removed = 0;
        double coefficient = 0.0;
    };
    std::vector<OrbitalShear> eliminations;
    for (int removed = 1; removed < n_orbitals_; ++removed) {
        for (int inserted = removed - 1; inserted >= 0; --inserted) {
            const double coefficient = unit_upper(inserted, removed);
            eliminations.push_back({inserted, removed, coefficient});
            unit_upper.col(removed).noalias() -= coefficient * unit_upper.col(inserted);
        }
    }

    shears_.reserve(eliminations.size());
    for (const OrbitalShear& operation : eliminations) {
        Shear shear;
        shear.coefficient = operation.coefficient;
        const std::uint64_t removed_bit = std::uint64_t{1} << operation.removed;
        const std::uint64_t inserted_bit = std::uint64_t{1} << operation.inserted;
        for (int source = 0; source < static_cast<int>(strings_.masks().size()); ++source) {
            const std::uint64_t source_mask = strings_.masks()[source];
            if ((source_mask & removed_bit) == 0 || (source_mask & inserted_bit) != 0) {
                continue;
            }
            const std::uint64_t target_mask = (source_mask ^ removed_bit) | inserted_bit;
            shear.pairs.push_back(
                ShearEdge{source,
                          strings_.index(target_mask),
                          replacement_sign(source_mask, operation.removed, operation.inserted)});
        }
        const std::size_t words = (strings_.masks().size() + 63) / 64;
        shear.source_words.assign(words, 0);
        for (const auto& pair : shear.pairs)
            shear.source_words[pair.source / 64] |= std::uint64_t{1} << (pair.source % 64);
        shear.word_offsets.resize(words + 1, 0);
        for (std::size_t w = 0; w < words; ++w)
            shear.word_offsets[w + 1] =
                shear.word_offsets[w] + std::popcount(shear.source_words[w]);
        shears_.push_back(std::move(shear));
    }
}

void ExteriorOrbitalTransform::validate_left(const Eigen::MatrixXd& coefficients) const {
    if (coefficients.rows() != dimension()) {
        throw std::invalid_argument("left exterior transform determinant dimension mismatch");
    }
}

void ExteriorOrbitalTransform::validate_right(const Eigen::MatrixXd& coefficients) const {
    if (coefficients.cols() != dimension()) {
        throw std::invalid_argument("right exterior transform determinant dimension mismatch");
    }
}

void ExteriorOrbitalTransform::apply_left(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr) {
        throw std::invalid_argument("left exterior transform output is null");
    }
    validate_left(*coefficients);
    transform_left(*coefficients, false, false);
}

void ExteriorOrbitalTransform::apply_left_block(Eigen::Ref<Eigen::MatrixXd> coefficients) const {
    if (coefficients.rows() != dimension())
        throw std::invalid_argument("left exterior block dimension mismatch");
    transform_left(coefficients, false, false);
}

void ExteriorOrbitalTransform::apply_right(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr) {
        throw std::invalid_argument("right exterior transform output is null");
    }
    apply_right_block(*coefficients);
}

void ExteriorOrbitalTransform::apply_right_block(Eigen::Ref<Eigen::MatrixXd> coefficients) const {
    if (coefficients.cols() != dimension()) {
        throw std::invalid_argument("right exterior transform determinant dimension mismatch");
    }
    transform_right(coefficients, false, false);
}

void ExteriorOrbitalTransform::apply_adjoint_left(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr) {
        throw std::invalid_argument("left exterior adjoint output is null");
    }
    validate_left(*coefficients);
    transform_left(*coefficients, true, false);
}

ExteriorProjection ExteriorOrbitalTransform::projection(const std::vector<int>& rows) const {
    std::vector<unsigned char> needed(dimension(), 0);
    for (int row : rows) {
        if (row < 0 || row >= dimension())
            throw std::invalid_argument("exterior projection row is outside the spin space");
        needed[row] = 1;
    }
    ExteriorProjection result;
    result.shear_pairs.resize(shears_.size());
    // The adjoint executes shears in reverse. Walk their dependency graph in
    // forward order, starting from the requested final rows. Pairs in one
    // elementary shear are disjoint, so this sweep needs no transitive iteration.
    for (std::size_t i = 0; i < shears_.size(); ++i) {
        const auto& pairs = shears_[i].pairs;
        for (int j = 0; j < static_cast<int>(pairs.size()); ++j)
            if (needed[pairs[j].source]) {
                result.shear_pairs[i].push_back(j);
                needed[pairs[j].target] = 1;
            }
    }
    return result;
}

void ExteriorOrbitalTransform::apply_adjoint_left(Eigen::MatrixXd* coefficients,
                                                  const ExteriorProjection& projection) const {
    if (!coefficients)
        throw std::invalid_argument("invalid exterior projection plan");
    apply_adjoint_left_block(*coefficients, projection);
}

void ExteriorOrbitalTransform::apply_adjoint_left_block(
    Eigen::Ref<Eigen::MatrixXd> coefficients, const ExteriorProjection& projection) const {
    if (coefficients.rows() != dimension() || projection.shear_pairs.size() != shears_.size())
        throw std::invalid_argument("invalid exterior projection block");
    profiling::ScopedPhase phase(profiling::Phase::CiTransformLeft);
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(cooperative_openmp_thread_count(), coefficients.size()),
        coefficients.cols());
    cooperative_openmp_for(coefficients.cols(), workers, [&](std::ptrdiff_t column) {
        auto values = coefficients.col(column);
        for (int i = static_cast<int>(shears_.size()) - 1; i >= 0; --i) {
            const auto& shear = shears_[i];
            if (shear.coefficient == 0.0)
                continue;
            for (int j : projection.shear_pairs[i]) {
                const auto& pair = shear.pairs[j];
                values[pair.source] += shear.coefficient * pair.sign * values[pair.target];
            }
        }
        values.array() *= determinant_scales_.array();
    });
}

void ExteriorOrbitalTransform::apply_supported_left(Eigen::MatrixXd* coefficients,
                                                    std::span<const int> offsets,
                                                    std::span<const int> rows,
                                                    bool adjoint) const {
    xmvb::profiling::ScopedPhase validation(xmvb::profiling::Phase::CiSupportedValidation);
    if (!coefficients || offsets.size() < 2 || offsets.front() != 0 ||
        offsets.back() != static_cast<int>(rows.size()))
        throw std::invalid_argument("invalid supported exterior column pattern");
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiTransformLeft);
    validate_left(*coefficients);
    const int beta_count = static_cast<int>(offsets.size() - 1);
    if (coefficients->cols() % beta_count != 0)
        throw std::invalid_argument("supported exterior beta columns do not match RHS blocks");
    for (int beta = 0; beta < beta_count; ++beta) {
        if (offsets[beta] < 0 || offsets[beta] > offsets[beta + 1] ||
            offsets[beta + 1] > static_cast<int>(rows.size()))
            throw std::invalid_argument("invalid supported exterior column offsets");
    }
    for (const int alpha : rows)
        if (alpha < 0 || alpha >= dimension())
            throw std::invalid_argument("supported exterior row is outside the spin space");
    validation.stop();
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(cooperative_openmp_thread_count(), coefficients->size()),
        beta_count);
    const int n_shears = static_cast<int>(shears_.size());
    cooperative_openmp_for(workers, workers, [&](std::ptrdiff_t part) {
        std::vector<std::uint8_t> allowed(dimension());
        std::vector<std::uint64_t> words((dimension() + 63) / 64);
        workspace::Temporary support_temporary(allowed.size() * sizeof(std::uint8_t) +
                                               words.size() * sizeof(std::uint64_t));
        const int first = beta_count * part / workers;
        const int last = beta_count * (part + 1) / workers;
        for (int beta = first; beta < last; ++beta) {
            std::fill(allowed.begin(), allowed.end(), 0);
            std::fill(words.begin(), words.end(), 0);
            for (int index = offsets[beta]; index < offsets[beta + 1]; ++index) {
                allowed[rows[index]] = 1;
                words[rows[index] / 64] |= std::uint64_t{1} << (rows[index] % 64);
            }
            for (Eigen::Index column = beta; column < coefficients->cols(); column += beta_count) {
                auto values = coefficients->col(column);
                for (int alpha = 0; alpha < dimension(); ++alpha) {
                    if (!allowed[alpha])
                        values[alpha] = 0.0;
                    else if (!adjoint)
                        values[alpha] *= determinant_scales_[alpha];
                }
            }
            // The graph belongs to beta's symbolic support, not to an RHS. Visit
            // its intersection once and update the independent RHS lanes together.
            const bool full = offsets[beta + 1] - offsets[beta] == dimension();
            for (int index = 0; index < n_shears; ++index) {
                const Shear& shear = shears_[adjoint ? n_shears - 1 - index : index];
                if (shear.coefficient == 0.0)
                    continue;
                const auto update = [&](const ShearEdge& pair) {
                    const int target = adjoint ? pair.source : pair.target;
                    const int source = adjoint ? pair.target : pair.source;
                    const double factor = shear.coefficient * pair.sign;
                    for (Eigen::Index column = beta; column < coefficients->cols();
                         column += beta_count)
                        (*coefficients)(target, column) += factor * (*coefficients)(source, column);
                };
                if (full) {
                    for (const auto& pair : shear.pairs)
                        update(pair);
                } else
                    for (std::size_t w = 0; w < words.size(); ++w) {
                        const auto sources = shear.source_words[w];
                        auto live = sources & words[w];
                        while (live) {
                            const int bit = std::countr_zero(live);
                            const int edge =
                                shear.word_offsets[w] +
                                std::popcount(sources & ((std::uint64_t{1} << bit) - 1));
                            const auto& pair = shear.pairs[edge];
                            if (allowed[pair.target])
                                update(pair);
                            live &= live - 1;
                        }
                    }
            }
            if (adjoint)
                for (Eigen::Index column = beta; column < coefficients->cols();
                     column += beta_count) {
                    auto values = coefficients->col(column);
                    for (int alpha = 0; alpha < dimension(); ++alpha)
                        if (allowed[alpha])
                            values[alpha] *= determinant_scales_[alpha];
                }
        }
    });
}

void ExteriorOrbitalTransform::apply_adjoint_right(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr) {
        throw std::invalid_argument("right exterior adjoint output is null");
    }
    apply_adjoint_right_block(*coefficients);
}

void ExteriorOrbitalTransform::apply_adjoint_right_block(
    Eigen::Ref<Eigen::MatrixXd> coefficients) const {
    if (coefficients.cols() != dimension()) {
        throw std::invalid_argument("right exterior transform determinant dimension mismatch");
    }
    transform_right(coefficients, true, false);
}

std::shared_ptr<const Eigen::MatrixXd> ExteriorOrbitalTransform::row_block(int begin,
                                                                           int count) const {
    if (begin < 0 || count <= 0 || begin > dimension() - count) {
        throw std::invalid_argument("exterior row block is out of range");
    }
    const std::size_t width = static_cast<std::size_t>(dimension());
    if (static_cast<std::size_t>(count) >
        std::numeric_limits<std::size_t>::max() / width / sizeof(double)) {
        throw std::length_error("exterior row block byte count overflows");
    }
    const auto bytes = sizeof(double) * width * count;
    row_cache_->request(bytes);
    return row_cache_->retain(RowCache::key(begin, count), bytes, [&] {
        // Generate outside the mutex; cache hits construct no row schedule.
        std::vector<int> selected(count);
        std::iota(selected.begin(), selected.end(), begin);
        return std::make_shared<Eigen::MatrixXd>(
            row_plan_->apply(orbital_transform_, strings_.masks(), selected));
    });
}

Eigen::SparseMatrix<double>
ExteriorOrbitalTransform::generate_rows(std::span<const int> rows) const {
    for (const int row : rows)
        if (row < 0 || row >= dimension())
            throw std::invalid_argument("exterior row index is out of range");
    const auto row_bytes = sizeof(double) * static_cast<std::size_t>(dimension());
    if (rows.size() > std::numeric_limits<std::size_t>::max() / row_bytes)
        throw std::length_error("exterior row panel byte count overflows");
    row_cache_->request(rows.size() * row_bytes);
    // A cached factor is contiguous by row. Assemble it in the same ordering;
    // converting the complete panel once avoids per-row scattered stores.
    ExteriorRowPlan::Rows result(rows.size(), dimension());
    workspace::Temporary result_temporary(sizeof(double) * result.size());
    std::vector<int> missing, positions;
    for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
        if (auto value = row_cache_->find(RowCache::key(rows[i], 1)))
            result.row(i) = value->row(0);
        else {
            missing.push_back(rows[i]);
            positions.push_back(i);
        }
    }
    if (!missing.empty()) {
        const auto values = row_plan_->apply(orbital_transform_, strings_.masks(), missing);
        workspace::Temporary values_temporary(sizeof(double) * values.size());
        for (int i = 0; i < static_cast<int>(missing.size()); ++i) {
            result.row(positions[i]) = values.row(i);
            // Cache immutable orbital factors, not state-dependent CI columns.
            // Input generation and output projection share the existing row budget.
            row_cache_->retain(RowCache::key(missing[i], 1), row_bytes, [&] {
                return std::make_shared<Eigen::MatrixXd>(values.row(i));
            });
        }
    }
    // Convert the assembled panel once, then count/pack independent CSC
    // columns in parallel. No per-spin-product allocation or serial sparse scan.
    const Eigen::MatrixXd dense = result;
    workspace::Temporary dense_temporary(sizeof(double) * dense.size());
    Eigen::SparseMatrix<double> factors(dense.rows(), dense.cols());
    int* offsets = factors.outerIndexPtr();
    offsets[0] = 0;
    const int workers = coarse_openmp_thread_count(cooperative_openmp_thread_count(), dense.size());
    cooperative_openmp_for(dense.cols(), workers, [&](std::ptrdiff_t column) {
        offsets[column + 1] = (dense.col(column).array() != 0.0).count();
    });
    for (int column = 0; column < dense.cols(); ++column)
        offsets[column + 1] += offsets[column];
    factors.resizeNonZeros(offsets[dense.cols()]);
    cooperative_openmp_for(dense.cols(), workers, [&](std::ptrdiff_t column) {
        int slot = offsets[column];
        for (int row = 0; row < dense.rows(); ++row) {
            const double value = dense(row, column);
            if (value == 0.0)
                continue;
            factors.innerIndexPtr()[slot] = row;
            factors.valuePtr()[slot++] = value;
        }
    });
    return factors;
}

void ExteriorOrbitalTransform::validate_inverse() const {
    if (!determinant_scales_.allFinite() || (determinant_scales_.array() == 0.0).any()) {
        throw std::runtime_error("exterior transform has noninvertible numerical scales");
    }
}

void ExteriorOrbitalTransform::apply_inverse_left(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr)
        throw std::invalid_argument("inverse exterior transform output is null");
    validate_left(*coefficients);
    validate_inverse();
    transform_left(*coefficients, false, true);
}

void ExteriorOrbitalTransform::apply_inverse_adjoint_left(Eigen::MatrixXd* coefficients) const {
    if (coefficients == nullptr)
        throw std::invalid_argument("inverse exterior adjoint output is null");
    validate_left(*coefficients);
    validate_inverse();
    transform_left(*coefficients, true, true);
}

void ExteriorOrbitalTransform::apply_inverse_right_block(
    Eigen::Ref<Eigen::MatrixXd> coefficients) const {
    if (coefficients.cols() != dimension())
        throw std::invalid_argument("inverse exterior column dimension mismatch");
    validate_inverse();
    transform_right(coefficients, false, true);
}

void ExteriorOrbitalTransform::apply_inverse_adjoint_right_block(
    Eigen::Ref<Eigen::MatrixXd> coefficients) const {
    if (coefficients.cols() != dimension())
        throw std::invalid_argument("inverse exterior adjoint column dimension mismatch");
    validate_inverse();
    transform_right(coefficients, true, true);
}

void ExteriorOrbitalTransform::transform_left(Eigen::Ref<Eigen::MatrixXd> coefficients,
                                              bool adjoint,
                                              bool inverse) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiTransformLeft);
    const bool reverse = adjoint != inverse;
    const int n_shears = static_cast<int>(shears_.size());
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(cooperative_openmp_thread_count(), coefficients.size()),
        coefficients.cols());
    // A column is an independent vector. Keep it resident while applying the
    // complete ordered product, rather than streaming strided rows per shear.
    cooperative_openmp_for(coefficients.cols(), workers, [&](std::ptrdiff_t column) {
        auto values = coefficients.col(column);
        if (!reverse) {
            if (inverse)
                values.array() /= determinant_scales_.array();
            else
                values.array() *= determinant_scales_.array();
        }
        for (int index = 0; index < n_shears; ++index) {
            const Shear& shear = shears_[reverse ? n_shears - 1 - index : index];
            if (shear.coefficient == 0.0)
                continue;
            const double factor = inverse ? -shear.coefficient : shear.coefficient;
            for (const ShearEdge& pair : shear.pairs) {
                const int target = adjoint ? pair.source : pair.target;
                const int source = adjoint ? pair.target : pair.source;
                values[target] += (factor * pair.sign) * values[source];
            }
        }
        if (reverse) {
            if (inverse)
                values.array() /= determinant_scales_.array();
            else
                values.array() *= determinant_scales_.array();
        }
    });
}

void ExteriorOrbitalTransform::transform_right(Eigen::Ref<Eigen::MatrixXd> coefficients,
                                               bool adjoint,
                                               bool inverse) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiTransformRight);
    const bool reverse = adjoint != inverse;
    const int n_shears = static_cast<int>(shears_.size());
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(cooperative_openmp_thread_count(), coefficients.size()),
        coefficients.rows());
    // Workers own disjoint row ranges. Each range executes all shears without
    // intermediate barriers. Column segments stay contiguous for SIMD updates.
    cooperative_openmp_for(workers, workers, [&](std::ptrdiff_t part) {
        const Eigen::Index first = coefficients.rows() * part / workers;
        const Eigen::Index end = coefficients.rows() * (part + 1) / workers;
        auto values = coefficients.middleRows(first, end - first);
        if (!reverse) {
            if (inverse)
                values.array().rowwise() /= determinant_scales_.transpose().array();
            else
                values.array().rowwise() *= determinant_scales_.transpose().array();
        }
        for (int index = 0; index < n_shears; ++index) {
            const Shear& shear = shears_[reverse ? n_shears - 1 - index : index];
            if (shear.coefficient == 0.0)
                continue;
            const double factor = inverse ? -shear.coefficient : shear.coefficient;
            for (const ShearEdge& pair : shear.pairs) {
                const int target = adjoint ? pair.source : pair.target;
                const int source = adjoint ? pair.target : pair.source;
                values.col(target) += (factor * pair.sign) * values.col(source);
            }
        }
        if (reverse) {
            if (inverse)
                values.array().rowwise() /= determinant_scales_.transpose().array();
            else
                values.array().rowwise() *= determinant_scales_.transpose().array();
        }
    });
}

ExteriorTransformDirection ExteriorOrbitalTransform::direction(
    const Eigen::Ref<const Eigen::MatrixXd>& delta_orbital_transform) const {
    if (delta_orbital_transform.rows() != n_orbitals_ ||
        delta_orbital_transform.cols() != n_orbitals_ || !delta_orbital_transform.allFinite()) {
        throw std::invalid_argument("exterior transform direction has incompatible dimensions");
    }
    const double lower_error =
        delta_orbital_transform.template triangularView<Eigen::StrictlyLower>()
            .toDenseMatrix()
            .cwiseAbs()
            .maxCoeff();
    const double scale = std::max(1.0, delta_orbital_transform.cwiseAbs().maxCoeff());
    if (lower_error > 64.0 * std::numeric_limits<double>::epsilon() * scale) {
        throw std::invalid_argument("exterior transform direction must be upper triangular");
    }

    ExteriorTransformDirection result;
    result.determinant_scales.resize(dimension());
    for (int determinant = 0; determinant < dimension(); ++determinant) {
        double logarithmic_direction = 0.0;
        for (const int orbital : strings_.occupied(determinant))
            logarithmic_direction +=
                delta_orbital_transform(orbital, orbital) / orbital_transform_(orbital, orbital);
        result.determinant_scales[determinant] =
            determinant_scales_[determinant] * logarithmic_direction;
    }

    Eigen::MatrixXd unit_upper = orbital_transform_;
    Eigen::MatrixXd delta_unit_upper = delta_orbital_transform;
    workspace::Temporary unit_upper_temporary(sizeof(double) * unit_upper.size());
    workspace::Temporary delta_unit_upper_temporary(sizeof(double) * delta_unit_upper.size());
    for (int column = 0; column < n_orbitals_; ++column) {
        const double diagonal = orbital_transform_(column, column);
        const double delta_diagonal = delta_orbital_transform(column, column);
        delta_unit_upper.col(column) =
            delta_unit_upper.col(column) / diagonal -
            unit_upper.col(column) * (delta_diagonal / (diagonal * diagonal));
        unit_upper.col(column) /= diagonal;
    }

    result.shear_coefficients.reserve(shears_.size());
    for (int removed = 1; removed < n_orbitals_; ++removed) {
        for (int inserted = removed - 1; inserted >= 0; --inserted) {
            const double coefficient = unit_upper(inserted, removed);
            const double delta_coefficient = delta_unit_upper(inserted, removed);
            result.shear_coefficients.push_back(delta_coefficient);
            delta_unit_upper.col(removed).noalias() -=
                delta_coefficient * unit_upper.col(inserted) +
                coefficient * delta_unit_upper.col(inserted);
            unit_upper.col(removed).noalias() -= coefficient * unit_upper.col(inserted);
        }
    }
    if (result.shear_coefficients.size() != shears_.size()) {
        throw std::logic_error("exterior transform direction does not match its shear schedule");
    }
    return result;
}

void ExteriorOrbitalTransform::apply_directional_left(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
    transform_directional_left(direction, coefficients, directional_coefficients, false);
}

void ExteriorOrbitalTransform::transform_directional_left(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* tangent,
    bool adjoint) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiDirectionalTransform);
    if (coefficients == nullptr || tangent == nullptr || coefficients == tangent ||
        coefficients->rows() != tangent->rows() || coefficients->cols() != tangent->cols() ||
        direction.determinant_scales.size() != dimension() ||
        direction.shear_coefficients.size() != shears_.size()) {
        throw std::invalid_argument("invalid directional left exterior action");
    }
    validate_left(*coefficients);
    const int n_shears = static_cast<int>(shears_.size());
    const int workers = bounded_openmp_thread_count(
        coarse_openmp_thread_count(cooperative_openmp_thread_count(), coefficients->size()),
        coefficients->cols());
    // Different columns are independent. Primal and tangent follow the same
    // ordered shears in one pass, with no thread-local copies or shear barriers.
    cooperative_openmp_for(coefficients->cols(), workers, [&](std::ptrdiff_t column) {
        auto value = coefficients->col(column);
        auto delta = tangent->col(column);
        const auto scale = [&]() {
            delta.array() = determinant_scales_.array() * delta.array() +
                            direction.determinant_scales.array() * value.array();
            value.array() *= determinant_scales_.array();
        };
        if (!adjoint)
            scale();
        for (int index = 0; index < n_shears; ++index) {
            const int ordered = adjoint ? n_shears - 1 - index : index;
            const Shear& shear = shears_[ordered];
            const double dc = direction.shear_coefficients[ordered];
            for (const ShearEdge& pair : shear.pairs) {
                const int source = adjoint ? pair.target : pair.source;
                const int target = adjoint ? pair.source : pair.target;
                delta[target] +=
                    pair.sign * (dc * value[source] + shear.coefficient * delta[source]);
                value[target] += (shear.coefficient * pair.sign) * value[source];
            }
        }
        if (adjoint)
            scale();
    });
}

void ExteriorOrbitalTransform::apply_directional_right(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiDirectionalTransform);
    if (coefficients == nullptr || directional_coefficients == nullptr ||
        coefficients->rows() != directional_coefficients->rows() ||
        coefficients->cols() != directional_coefficients->cols() ||
        direction.determinant_scales.size() != dimension() ||
        direction.shear_coefficients.size() != shears_.size()) {
        throw std::invalid_argument("invalid directional right exterior action");
    }
    validate_right(*coefficients);
    for (int column = 0; column < dimension(); ++column) {
        directional_coefficients->col(column) =
            determinant_scales_[column] * directional_coefficients->col(column) +
            direction.determinant_scales[column] * coefficients->col(column);
        coefficients->col(column) *= determinant_scales_[column];
    }
    for (std::size_t shear_index = 0; shear_index < shears_.size(); ++shear_index) {
        const Shear& shear = shears_[shear_index];
        const double delta_coefficient = direction.shear_coefficients[shear_index];
        for (const ShearEdge& pair : shear.pairs) {
            directional_coefficients->col(pair.target) +=
                pair.sign * (delta_coefficient * coefficients->col(pair.source) +
                             shear.coefficient * directional_coefficients->col(pair.source));
            coefficients->col(pair.target) +=
                (shear.coefficient * pair.sign) * coefficients->col(pair.source);
        }
    }
}

void ExteriorOrbitalTransform::apply_directional_adjoint_left(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
    transform_directional_left(direction, coefficients, directional_coefficients, true);
}

void ExteriorOrbitalTransform::apply_directional_adjoint_right(
    const ExteriorTransformDirection& direction,
    Eigen::MatrixXd* coefficients,
    Eigen::MatrixXd* directional_coefficients) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiDirectionalTransform);
    if (coefficients == nullptr || directional_coefficients == nullptr ||
        coefficients->rows() != directional_coefficients->rows() ||
        coefficients->cols() != directional_coefficients->cols() ||
        direction.determinant_scales.size() != dimension() ||
        direction.shear_coefficients.size() != shears_.size()) {
        throw std::invalid_argument("invalid directional adjoint-right exterior action");
    }
    validate_right(*coefficients);
    for (std::size_t reverse = shears_.size(); reverse-- > 0;) {
        const Shear& shear = shears_[reverse];
        const double delta_coefficient = direction.shear_coefficients[reverse];
        for (const ShearEdge& pair : shear.pairs) {
            directional_coefficients->col(pair.source) +=
                pair.sign * (delta_coefficient * coefficients->col(pair.target) +
                             shear.coefficient * directional_coefficients->col(pair.target));
            coefficients->col(pair.source) +=
                (shear.coefficient * pair.sign) * coefficients->col(pair.target);
        }
    }
    for (int column = 0; column < dimension(); ++column) {
        directional_coefficients->col(column) =
            determinant_scales_[column] * directional_coefficients->col(column) +
            direction.determinant_scales[column] * coefficients->col(column);
        coefficients->col(column) *= determinant_scales_[column];
    }
}

std::size_t ExteriorOrbitalTransform::shear_pair_count() const noexcept {
    std::size_t count = 0;
    for (const Shear& shear : shears_) {
        count += shear.pairs.size();
    }
    return count;
}

Eigen::MatrixXd
ExteriorOrbitalTransform::generator_pullback(const Eigen::MatrixXd& seed,
                                             const Eigen::MatrixXd& image,
                                             const Eigen::MatrixXd* delta_seed,
                                             const Eigen::MatrixXd* delta_image,
                                             const Eigen::MatrixXd* delta_upper) const {
    const ExteriorGeneratorTerm terms[]{{seed, image, 1.0, delta_seed, delta_image}};
    return generator_pullback(terms, delta_upper);
}

Eigen::MatrixXd
ExteriorOrbitalTransform::generator_pullback(std::span<const ExteriorGeneratorTerm> terms,
                                             const Eigen::MatrixXd* delta_upper) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiExteriorGeneratorPullback);
    const bool directional = delta_upper != nullptr;
    if (terms.empty())
        throw std::invalid_argument("empty generator pullback terms");
    std::size_t columns = 0;
    for (const auto& term : terms) {
        if (term.seed.rows() != dimension() || term.image.rows() != dimension() ||
            term.seed.cols() != term.image.cols() || !std::isfinite(term.scale) ||
            directional != (term.delta_seed != nullptr) ||
            directional != (term.delta_image != nullptr))
            throw std::invalid_argument("invalid generator pullback panels");
        if (directional &&
            (term.delta_seed->rows() != dimension() || term.delta_image->rows() != dimension() ||
             term.delta_seed->cols() != term.seed.cols() ||
             term.delta_image->cols() != term.image.cols()))
            throw std::invalid_argument("invalid directional generator panels");
        columns += term.seed.cols();
    }
    Eigen::MatrixXd g = Eigen::MatrixXd::Zero(n_orbitals_, n_orbitals_);
    Eigen::MatrixXd dg = g;
    workspace::Temporary g_temporary(sizeof(double) * g.size());
    workspace::Temporary dg_temporary(sizeof(double) * dg.size());
    const auto contract = [&](int target, int source, bool tangent) -> double {
        double value = 0;
        for (const auto& term : terms)
            value +=
                term.scale * (tangent ? term.delta_seed->row(target).dot(term.image.row(source)) +
                                            term.seed.row(target).dot(term.delta_image->row(source))
                                      : term.seed.row(target).dot(term.image.row(source)));
        return value;
    };
    const int workers = coarse_openmp_thread_count(
        cooperative_openmp_thread_count(),
        columns * (static_cast<std::size_t>(dimension()) + shear_pair_count()));
    std::vector<Eigen::MatrixXd> diagonals(workers,
                                           Eigen::MatrixXd::Zero(n_orbitals_, directional ? 2 : 1));
    workspace::Temporary diagonals_temporary(sizeof(double) * static_cast<std::size_t>(workers) *
                                             n_orbitals_ * (directional ? 2 : 1));
    cooperative_openmp_for(workers, workers, [&](std::ptrdiff_t part) {
        auto& diagonal = diagonals[part];
        for (int string = dimension() * part / workers; string < dimension() * (part + 1) / workers;
             ++string) {
            const double value = contract(string, string, false);
            const double derivative = directional ? contract(string, string, true) : 0.0;
            for (const int orbital : strings_.occupied(string)) {
                diagonal(orbital, 0) += value;
                if (directional)
                    diagonal(orbital, 1) += derivative;
            }
        }
        // Every upper generator owns one shear list and one output entry. Unlike
        // carrier rows, these outputs need neither atomics nor n_orbital^2 copies.
        int shear = 0;
        for (int removed = 1; removed < n_orbitals_; ++removed)
            for (int inserted = removed - 1; inserted >= 0; --inserted, ++shear) {
                if (shear % workers != part)
                    continue;
                double value = 0, derivative = 0;
                for (const auto& pair : shears_[shear].pairs) {
                    value += pair.sign * contract(pair.target, pair.source, false);
                    if (directional)
                        derivative += pair.sign * contract(pair.target, pair.source, true);
                }
                g(inserted, removed) = value;
                if (directional)
                    dg(inserted, removed) = derivative;
            }
    });
    for (const auto& diagonal : diagonals) {
        g.diagonal() += diagonal.col(0);
        if (directional)
            dg.diagonal() += diagonal.col(1);
    }
    return moment_pullback(g, directional ? &dg : nullptr, delta_upper);
}

Eigen::MatrixXd
ExteriorOrbitalTransform::moment_pullback(const Eigen::Ref<const Eigen::MatrixXd>& moment,
                                          const Eigen::MatrixXd* delta_moment,
                                          const Eigen::MatrixXd* delta_upper) const {
    xmvb::profiling::ScopedPhase phase(xmvb::profiling::Phase::CiExteriorMomentPullback);
    const Eigen::MatrixXd inverse = orbital_transform_.triangularView<Eigen::Upper>()
                                        .solve(Eigen::MatrixXd::Identity(n_orbitals_, n_orbitals_))
                                        .transpose();
    workspace::Temporary inverse_temporary(sizeof(double) * inverse.size());
    Eigen::MatrixXd result;
    if (delta_upper) {
        const Eigen::MatrixXd di = -inverse * delta_upper->transpose() * inverse;
        workspace::Temporary di_temporary(sizeof(double) * di.size());
        result = *delta_moment * inverse + moment * di;
    } else {
        result = moment * inverse;
    }
    result.triangularView<Eigen::StrictlyLower>().setZero();
    return result;
}

std::size_t ExteriorOrbitalTransform::dynamic_bytes() const noexcept {
    std::size_t bytes = static_cast<std::size_t>(orbital_transform_.size()) * sizeof(double) +
                        strings_.dynamic_bytes() +
                        static_cast<std::size_t>(determinant_scales_.size()) * sizeof(double) +
                        shears_.capacity() * sizeof(Shear);
    for (const Shear& shear : shears_) {
        bytes += shear.pairs.capacity() * sizeof(ShearEdge);
        bytes += shear.source_words.capacity() * sizeof(std::uint64_t) +
                 shear.word_offsets.capacity() * sizeof(int);
    }
    if (row_cache_)
        bytes += row_cache_->cached_bytes.load(std::memory_order_relaxed);
    bytes += row_plan_->dynamic_bytes();
    return bytes;
}

} // namespace xmvb::vb
