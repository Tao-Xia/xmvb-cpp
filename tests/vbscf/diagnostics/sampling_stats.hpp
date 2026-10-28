#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>

#include <Eigen/QR>

#include "sampling_point.hpp"

namespace sampling_audit {
struct Method {
    std::string name;
    Eigen::VectorXd probability, control_sum;
    Eigen::MatrixXd control;
    double setup_seconds = 0;
};
inline Eigen::VectorXd draw(const Eigen::VectorXd& p, std::size_t samples, std::uint64_t seed) {
    std::mt19937_64 random(seed);
    std::discrete_distribution<int> distribution(p.data(), p.data() + p.size());
    Eigen::VectorXd weights = Eigen::VectorXd::Zero(p.size());
    for (std::size_t i = 0; i < samples; ++i)
        ++weights[distribution(random)];
    weights.array() /= static_cast<double>(samples) * p.array();
    return weights;
}
inline std::vector<int> touched(const Eigen::VectorXd& weights) {
    std::vector<int> result;
    for (int j = 0; j < weights.size(); ++j)
        if (weights[j] != 0)
            result.push_back(j);
    return result;
}
inline Eigen::VectorXd
correction(const Method& method, const Eigen::MatrixXd& features, const Eigen::VectorXd& weights) {
    if (method.control.size() == 0)
        return Eigen::VectorXd::Zero(method.control_sum.size());
    return method.control_sum - method.control * (features * weights);
}
struct Error {
    double h, m, energy;
    bool positive;
};
inline Error
error(const Eigen::VectorXd& value, const Eigen::VectorXd& reference, const Eigen::VectorXd& c) {
    const int n = c.size();
    const double denom = c.dot(value.tail(n)), refdenom = c.dot(reference.tail(n));
    return {(value.head(n) - reference.head(n)).norm() / reference.head(n).norm(),
            (value.tail(n) - reference.tail(n)).norm() / reference.tail(n).norm(),
            denom > 0 ? c.dot(value.head(n)) / denom - c.dot(reference.head(n)) / refdenom
                      : std::numeric_limits<double>::quiet_NaN(),
            denom > 0};
}

/** Exact population variance of a single importance draw, conditional on pilot.
 * Targets are linear H/M images and the first-order Rayleigh energy fluctuation.
 * The latter is not the variance of a noisy denominator ratio.
 */
inline Eigen::Vector3d variance(const Eigen::MatrixXd& residual,
                                const Eigen::VectorXd& probability,
                                const Eigen::VectorXd& reference,
                                const Eigen::VectorXd& c) {
    const int n = c.size();
    const double norm = c.dot(reference.tail(n)), e = c.dot(reference.head(n)) / norm;
    const Eigen::VectorXd sum = residual.rowwise().sum();
    const double sum_e = c.dot(sum.head(n) - e * sum.tail(n)) / norm;
    long double vh = 0, vm = 0, ve = 0;
    for (int j = 0; j < residual.cols(); ++j) {
        vh += static_cast<long double>(residual.col(j).head(n).squaredNorm()) / probability[j];
        vm += static_cast<long double>(residual.col(j).tail(n).squaredNorm()) / probability[j];
        const double ej = c.dot(residual.col(j).head(n) - e * residual.col(j).tail(n)) / norm;
        ve += static_cast<long double>(ej) * ej / probability[j];
    }
    return {static_cast<double>(std::max(0.0L, vh - sum.head(n).squaredNorm())),
            static_cast<double>(std::max(0.0L, vm - sum.tail(n).squaredNorm())),
            static_cast<double>(std::max(0.0L, ve - static_cast<long double>(sum_e) * sum_e))};
}

inline void statistical_survey(const std::string& prefix,
                               const Eigen::MatrixXd& contributions,
                               const Eigen::MatrixXd& features,
                               const std::vector<Method>& methods,
                               const Eigen::VectorXd& reference,
                               const Eigen::VectorXd& c,
                               const std::vector<std::size_t>& sample_counts) {
    std::ofstream analytic(prefix + "-variance.csv"), empirical(prefix + "-trials.csv"),
        columns(prefix + "-columns.csv");
    columns << std::setprecision(17)
            << "method,column,probability,H_norm,M_norm,energy_linear_contribution_Eh\n";
    analytic << std::setprecision(17)
             << "method,var_H,var_M,var_energy_linear,relative_H_single_sd,relative_M_single_sd,"
                "samples_H_rms_1e-3,samples_H_rms_1e-5,samples_M_rms_1e-5,"
                "samples_energy_linear_rms_1e-7,samples_energy_linear_rms_1e-5,"
                "oracle_min_var_H,oracle_min_var_energy_linear,oracle_min_samples_energy_1e-7\n";
    empirical << std::setprecision(17)
              << "method,samples,replicate,unique_columns,H_relative_error,M_relative_error,"
                 "energy_error_Eh,positive_denominator\n";
    const int n = c.size();
    for (const auto& method : methods) {
        Eigen::MatrixXd residual = contributions;
        if (method.control.size())
            residual.noalias() -= method.control * features;
        const auto v = variance(residual, method.probability, reference, c);
        const double denom = c.dot(reference.tail(n)), energy = c.dot(reference.head(n)) / denom;
        long double sum_h_norm = 0, sum_abs_energy = 0, sum_energy = 0;
        for (int j = 0; j < residual.cols(); ++j) {
            const double hn = residual.col(j).head(n).norm(), mn = residual.col(j).tail(n).norm(),
                         ej = c.dot(residual.col(j).head(n) - energy * residual.col(j).tail(n)) /
                              denom;
            sum_h_norm += hn;
            sum_abs_energy += std::abs(ej);
            sum_energy += ej;
            columns << method.name << ',' << j << ',' << method.probability[j] << ',' << hn << ','
                    << mn << ',' << ej << '\n';
        }
        const Eigen::VectorXd residual_sum = residual.rowwise().sum();
        const long double lower_h = std::max(
                              0.0L, sum_h_norm * sum_h_norm - residual_sum.head(n).squaredNorm()),
                          lower_e = std::max(
                              0.0L, sum_abs_energy * sum_abs_energy - sum_energy * sum_energy);
        const double hnorm2 = reference.head(n).squaredNorm(),
                     mnorm2 = reference.tail(n).squaredNorm();
        analytic << method.name << ',' << v[0] << ',' << v[1] << ',' << v[2] << ','
                 << std::sqrt(v[0] / hnorm2) << ',' << std::sqrt(v[1] / mnorm2) << ','
                 << std::ceil(v[0] / hnorm2 / 1e-6) << ',' << std::ceil(v[0] / hnorm2 / 1e-10)
                 << ',' << std::ceil(v[1] / mnorm2 / 1e-10) << ',' << std::ceil(v[2] / 1e-14) << ','
                 << std::ceil(v[2] / 1e-10) << ',' << lower_h << ',' << lower_e << ','
                 << std::ceil(lower_e / 1e-14L) << '\n';
        for (auto samples : sample_counts)
            for (int replicate = 0; replicate < 64; ++replicate) {
                const auto weights = draw(method.probability, samples, 0x3759afULL + replicate);
                const Eigen::VectorXd value =
                    contributions * weights + correction(method, features, weights);
                const auto e = error(value, reference, c);
                empirical << method.name << ',' << samples << ',' << replicate << ','
                          << touched(weights).size() << ',' << e.h << ',' << e.m << ',' << e.energy
                          << ',' << e.positive << '\n';
            }
    }
}
} // namespace sampling_audit
