#include "mcscf/states/weights.hpp"

#include <stdexcept>

namespace xmvb::mcscf {

void validate_weights(const std::vector<double>& weights) {
    if (weights.empty()) {
        throw std::invalid_argument("state_average_weights must not be empty");
    }
    for (const double weight : weights) {
        if (weight < 0.0) {
            throw std::invalid_argument("state_average_weights must be non-negative");
        }
    }
}

std::vector<double> normalize_weights(const std::vector<double>& weights) {
    validate_weights(weights);
    double sum = 0.0;
    for (const double weight : weights) {
        sum += weight;
    }
    if (sum <= 0.0) {
        throw std::invalid_argument("state_average_weights must sum to a positive value");
    }
    auto normalized = weights;
    for (double& weight : normalized) {
        weight /= sum;
    }
    return normalized;
}

} // namespace xmvb::mcscf
