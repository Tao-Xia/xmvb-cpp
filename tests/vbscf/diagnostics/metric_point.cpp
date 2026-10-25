#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/QR>

#include "input/loading/loader.hpp"
#include "output/trace/binary.hpp"
#include "vbscf/orbitals/charts/chart.hpp"
#include "vbscf/orbitals/charts/physical_metric.hpp"
#include "vbscf/orbitals/preparation/preparer.hpp"

namespace {
using namespace xmvb::vb;

/** Independent differentiated ray/span map; no energy, gradient or CI solve. */
class MetricFeatures {
  public:
    explicit MetricFeatures(const OrbitalPreparationInput& input, bool span)
        : n_(input.n_basis_functions),
          ni_((input.n_total_electrons - input.n_active_electrons) / 2),
          na_(input.n_active_orbitals), span_(span), s_(input.ao_overlap_matrix),
          c_(Eigen::MatrixXd::Zero(n_, ni_ + na_)) {
        for (int i = 0; i < ni_ + na_; ++i)
            for (int j = 0; j < stored_sparse_orbital_coefficient_count(input, i); ++j)
                c_(input.orbital_basis_index_table[i * n_ + j] - 1, i) =
                    input.orbital_value_table[i * n_ + j];
        root_ = s_.llt().matrixU();
        const Eigen::MatrixXd h = c_.leftCols(ni_).transpose() * s_ * c_.leftCols(ni_);
        const Eigen::MatrixXd upper = h.llt().matrixU();
        invroot_ = upper.triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(ni_, ni_));
        inverse_ = h.ldlt().solve(Eigen::MatrixXd::Identity(ni_, ni_));
        p_ = Eigen::MatrixXd::Identity(n_, n_) -
             c_.leftCols(ni_) * inverse_ * c_.leftCols(ni_).transpose() * s_;
        d_ = p_ * c_.rightCols(na_);
        coupling_ = inverse_ * c_.leftCols(ni_).transpose() * s_ * c_.rightCols(na_);
        if (span_) {
            const Eigen::MatrixXd h_active = d_.transpose() * s_ * d_;
            const Eigen::MatrixXd upper_active = h_active.llt().matrixU();
            active_invroot_ = upper_active.triangularView<Eigen::Upper>().solve(
                Eigen::MatrixXd::Identity(na_, na_));
            active_p_ =
                Eigen::MatrixXd::Identity(n_, n_) - d_ * h_active.ldlt().solve(d_.transpose() * s_);
        }
        std::cout << "inactive_inverse_skew=" << (inverse_ - inverse_.transpose()).norm()
                  << "\ninactive_inverse_factor_gap="
                  << (inverse_ - invroot_ * invroot_.transpose()).norm()
                  << "\ninactive_inverse_norm=" << inverse_.norm() << '\n';
    }

    Eigen::VectorXd apply(const OrbitalPreparationInput& input,
                          const SparseParameterLayout& layout,
                          const Eigen::VectorXd& packed) const {
        Eigen::MatrixXd dc = Eigen::MatrixXd::Zero(n_, ni_ + na_);
        for (int i = 0; i < ni_ + na_; ++i)
            for (int j = 0; j < layout.orbital_coefficient_count(i); ++j)
                dc(input.orbital_basis_index_table[i * n_ + j] - 1, i) =
                    packed[layout.packed_index(i, j)];
        const Eigen::MatrixXd hi = p_ * dc.leftCols(ni_);
        Eigen::MatrixXd da =
            p_ * dc.rightCols(na_) - hi * coupling_ -
            c_.leftCols(ni_) * inverse_ * (hi.transpose() * s_ * c_.rightCols(na_));
        Eigen::MatrixXd f(n_, ni_ + na_);
        f.leftCols(ni_) = root_ * hi * invroot_;
        if (span_)
            f.rightCols(na_) = root_ * active_p_ * da * active_invroot_;
        else
            for (int i = 0; i < na_; ++i) {
                const double norm = d_.col(i).dot(s_ * d_.col(i));
                const Eigen::VectorXd tangent =
                    da.col(i) - d_.col(i) * (d_.col(i).dot(s_ * da.col(i)) / norm);
                f.col(ni_ + i) = root_ * tangent / std::sqrt(norm);
            }
        return Eigen::Map<const Eigen::VectorXd>(f.data(), f.size());
    }

  private:
    int n_, ni_, na_;
    bool span_;
    Eigen::MatrixXd s_, c_, root_, invroot_, inverse_, p_, d_, coupling_, active_invroot_,
        active_p_;
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::invalid_argument(
                "usage: test_metric_point INPUT.xmi ORBITALS.bin OUTPUT_DIR");
        auto loaded = xmvb::vb::load_vbscf_input_with_timings(argv[1]);
        auto& input = loaded.input.orbital_preparation_input;
        std::ifstream stream(argv[2], std::ios::binary);
        stream.read(reinterpret_cast<char*>(input.orbital_value_table.data()),
                    input.orbital_value_table.size() * sizeof(double));
        if (!stream || stream.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("raw orbital checkpoint dimensions differ");
        const SparseParameterLayout layout(input);
        const auto prepared = ActiveSpaceOrbitalPreparer().prepare(input);
        const int occupied =
            (input.n_total_electrons - input.n_active_electrons) / 2 + input.n_active_orbitals;
        const OrbitalChart chart(input,
                                 layout,
                                 prepared.auxiliary_orbital_matrix.leftCols(occupied),
                                 prepared.physical_orbital_frame.normalized_orbital_matrix,
                                 nullptr,
                                 false,
                                 loaded.input.complete_active_space,
                                 nullptr,
                                 OrbitalChart::Coordinates::Unwhitened);
        const OrbitalPhysicalMetric metric(input, chart.uses_active_subspace_gauge());
        std::cout << std::setprecision(17) << "packed=" << layout.size()
                  << "\nreduced=" << chart.reduced_size() << '\n';
        const MetricFeatures features(input, chart.uses_active_subspace_gauge());
        const int n = chart.reduced_size();
        Eigen::MatrixXd action(n, n), lift(layout.size(), n),
            feature(input.n_basis_functions * occupied, n);
        for (int j = 0; j < n; ++j) {
            lift.col(j) = chart.expand_step(Eigen::VectorXd::Unit(n, j));
            action.col(j) = chart.project_reduced_gradient(metric.apply(layout, lift.col(j)));
            feature.col(j) = features.apply(input, layout, lift.col(j));
        }
        const Eigen::MatrixXd gram = feature.transpose() * feature;
        std::cout << "action_norm=" << action.norm()
                  << "\naction_skew=" << (action - action.transpose()).norm()
                  << "\nfeature_gram_norm=" << gram.norm()
                  << "\naction_feature_gap=" << (action - gram).norm() << '\n';
        const int width = std::min(16, n);
        Eigen::MatrixXd seed(n, width);
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < n; ++i)
                seed(i, j) =
                    std::sin(0.17 * (i + 1) * (j + 2)) + std::cos(0.31 * (i + 2) * (j + 1));
        const Eigen::MatrixXd basis =
            seed.householderQr().householderQ() * Eigen::MatrixXd::Identity(n, width);
        const Eigen::MatrixXd images = action * basis, projected = basis.transpose() * images;
        const Eigen::MatrixXd magnitude = basis.cwiseAbs().transpose() * images.cwiseAbs();
        const double u = 0.5 * std::numeric_limits<double>::epsilon(), gamma = n * u / (1 - n * u);
        const double bound =
            256 * std::numeric_limits<double>::epsilon() * width * std::max(1.0, projected.norm()) +
            ((gamma / (1 - gamma)) * (magnitude + magnitude.transpose()) +
             u * (projected.cwiseAbs() + projected.transpose().cwiseAbs()))
                .stableNorm();
        std::cout << "projected_skew=" << (projected - projected.transpose()).norm()
                  << "\nprojected_bound=" << bound << '\n';
        const std::filesystem::path out(argv[3]);
        std::filesystem::create_directories(out);
        xmvb::output::write_binary_buffer(out / "metric.bin", action.data(), action.size());
        xmvb::output::write_binary_buffer(out / "feature_gram.bin", gram.data(), gram.size());
        xmvb::output::write_binary_buffer(out / "lift.bin", lift.data(), lift.size());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
