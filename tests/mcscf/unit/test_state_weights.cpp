#include <cmath>
#include <stdexcept>
#include <vector>
#include "mcscf/states/weights.hpp"

int main() {
    const auto weights = xmvb::mcscf::normalize_weights({2.0, 0.0, 6.0});
    if (weights != std::vector<double>{0.25, 0.0, 0.75}) {
        throw std::runtime_error("state normalization changed weights or ordering");
    }
    for (const auto& invalid : {std::vector<double>{}, {0.0, 0.0}, {1.0, -1.0}}) {
        bool rejected = false;
        try {
            (void)xmvb::mcscf::normalize_weights(invalid);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid state weights were accepted");
    }
    const auto equal = xmvb::mcscf::normalize_weights({1.0, 1.0, 1.0});
    double sum = 0.0;
    for (const double weight : equal)
        sum += weight;
    if (std::abs(sum - 1.0) > 1e-15) {
        throw std::runtime_error("equal state weights are not normalized");
    }
}
