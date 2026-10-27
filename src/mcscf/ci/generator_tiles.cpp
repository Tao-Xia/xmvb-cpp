#include <cstdint>
#include "mcscf/ci/sigma.hpp"

#include <stdexcept>

#include "core/timing.hpp"
#include "parallel/openmp.hpp"

namespace xmvb::mcscf {

void DirectCiSigmaAction::apply_one_body_generator_columns(
    StringIndex first,
    const DirectCiColumnReader& columns,
    const Eigen::Ref<const Eigen::MatrixXd>& generator,
    Eigen::Ref<Eigen::MatrixXd> result,
    bool alpha_only) const {
    profiling::ScopedPhase phase(profiling::Phase::CiGenerator);
    const std::int64_t count = result.cols();
    if (!columns || first < 0 || count <= 0 || count > n_beta_ - first ||
        result.rows() != n_alpha_ || generator.rows() != n_orbitals_ ||
        generator.cols() != n_orbitals_ || !generator.allFinite())
        throw std::invalid_argument("invalid generated one-body panel");
    const auto borrow = [&](int b) {
        const auto view = columns(b);
        if (view.count < 0 || view.count > n_alpha_ ||
            (view.count > 0 && (!view.values || (!view.rows && view.count != n_alpha_))))
            throw std::invalid_argument("invalid immutable one-body column");
        return view;
    };
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(n_alpha_, count);
    for (std::int64_t b = 0; b < count; ++b) {
        const auto view = borrow(first + b);
        for (std::int64_t a = 0; a < view.count; ++a)
            input(view.rows ? view.rows[a] : a, b) = view.values[a];
    }
    const auto& beta = beta_connections();
    std::vector<DirectCiColumnView> spectators;
    if (!alpha_only) {
        spectators.resize(n_beta_);
        for (std::int64_t b = 0; b < n_beta_; ++b)
            spectators[b] = borrow(b);
    }
    Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(n_alpha_);
    for (std::int64_t a = 0; a < n_alpha_; ++a)
        for (const int p : alpha_->occupied(a))
            diagonal[a] += generator(p, p);
    const int workers = coarse_openmp_thread_count(cooperative_openmp_thread_count(), input.size());
    cooperative_openmp_for(count, workers, [&](std::ptrdiff_t b) {
        double beta_diagonal = 0;
        if (!alpha_only)
            for (const int p : beta.occupied(first + b))
                beta_diagonal += generator(p, p);
        for (std::int64_t a = 0; a < n_alpha_; ++a) {
            double value = (diagonal[a] + beta_diagonal) * input(a, b);
            for (const auto& link : alpha_->singles(a)) {
                const double factor = generator(link.created, link.annihilated);
                if (factor != 0)
                    value += link.sign * factor * input(link.source, b);
            }
            result(a, b) = value;
        }
        if (!alpha_only)
            for (const auto& link : beta.singles(first + b)) {
                const double factor = link.sign * generator(link.created, link.annihilated);
                if (factor == 0)
                    continue;
                const auto& source = spectators[link.source];
                for (std::int64_t a = 0; a < source.count; ++a)
                    result(source.rows ? source.rows[a] : a, b) += factor * source.values[a];
            }
    });
}

} // namespace xmvb::mcscf
