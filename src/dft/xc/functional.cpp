#include "dft/xc/functional.hpp"

#include <xc.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xmvb::dft {
namespace {

std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\n\r");
    if (first == std::string::npos)
        return {};
    const auto last = text.find_last_not_of(" \t\n\r");
    return text.substr(first, last - first + 1);
}

struct InputNoise {
    double density = 0.0;
    double sigma = 0.0;
};

void validate_inputs(const Eigen::Ref<const Eigen::MatrixXd>& rho,
                     const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                     int n_spin,
                     bool gga) {
    const int n_sigma = n_spin == 1 ? 1 : 3;
    const Eigen::Index n_points = rho.rows();
    if (rho.cols() != n_spin || sigma.rows() != n_points ||
        (gga ? sigma.cols() != n_sigma : (sigma.cols() != 0 && sigma.cols() != n_sigma))) {
        throw std::invalid_argument("XC rho/sigma matrix shape mismatch");
    }

    double density_scale = 0.0;
    for (Eigen::Index i = 0; i < n_points; ++i) {
        for (int spin = 0; spin < n_spin; ++spin) {
            const double value = rho(i, spin);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("XC density must be finite");
            }
            density_scale = std::max(density_scale, value);
        }
    }
    InputNoise noise;
    noise.density = 32.0 * std::numeric_limits<double>::epsilon() * density_scale;
    for (Eigen::Index i = 0; i < n_points; ++i) {
        for (int spin = 0; spin < n_spin; ++spin) {
            if (rho(i, spin) < -noise.density) {
                throw std::invalid_argument("XC density is materially negative");
            }
        }
    }

    if (gga) {
        double sigma_scale = 0.0;
        for (Eigen::Index i = 0; i < n_points; ++i) {
            for (int channel = 0; channel < n_sigma; ++channel) {
                const double value = sigma(i, channel);
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("XC sigma must be finite");
                }
                if (n_spin == 1 || channel != 1) {
                    sigma_scale = std::max(sigma_scale, value);
                }
            }
        }
        noise.sigma = 32.0 * std::numeric_limits<double>::epsilon() * sigma_scale;
        for (Eigen::Index i = 0; i < n_points; ++i) {
            for (int channel = 0; channel < n_sigma; ++channel) {
                if ((n_spin == 1 || channel != 1) && sigma(i, channel) < -noise.sigma) {
                    throw std::invalid_argument("XC diagonal sigma is materially negative");
                }
            }
        }
    }
}

void pack_inputs(const Eigen::Ref<const Eigen::MatrixXd>& rho,
                 const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                 int n_spin,
                 bool gga,
                 Eigen::Index first,
                 Eigen::Index count,
                 double* packed_rho,
                 double* packed_sigma) {
    const int n_sigma = n_spin == 1 ? 1 : 3;
    for (Eigen::Index local = 0; local < count; ++local) {
        const Eigen::Index point = first + local;
        for (int spin = 0; spin < n_spin; ++spin) {
            packed_rho[static_cast<std::size_t>(local) * n_spin + spin] =
                std::max(0.0, rho(point, spin));
        }
        if (gga) {
            for (int channel = 0; channel < n_sigma; ++channel) {
                const double value = sigma(point, channel);
                packed_sigma[static_cast<std::size_t>(local) * n_sigma + channel] =
                    (n_spin == 2 && channel == 1) ? value : std::max(0.0, value);
            }
        }
    }
}

struct Component {
    xc_func_type function{};
    int family = 0;
    double exact_exchange = 0.0;
    bool initialized = false;

    Component(const std::string& name, int identifier, int polarization) {
        if (xc_func_init(&function, identifier, polarization) != 0) {
            throw std::invalid_argument("Libxc could not initialize " + name);
        }
        initialized = true;
        const auto* info = xc_func_get_info(&function);
        const int flags = xc_func_info_get_flags(info);
        family = xc_func_info_get_family(info);
        const bool hybrid = family == XC_FAMILY_HYB_LDA || family == XC_FAMILY_HYB_GGA;
        const int required =
            XC_FLAGS_HAVE_EXC | XC_FLAGS_HAVE_VXC | XC_FLAGS_HAVE_FXC | XC_FLAGS_3D;
        const int unsupported = XC_FLAGS_HYB_CAM | XC_FLAGS_HYB_CAMY | XC_FLAGS_HYB_LC |
                                XC_FLAGS_HYB_LCY | XC_FLAGS_VV10;
        if ((!hybrid && family != XC_FAMILY_LDA && family != XC_FAMILY_GGA) ||
            xc_func_info_get_kind(info) == XC_KINETIC || (flags & required) != required ||
            (flags & unsupported) != 0) {
            xc_func_end(&function);
            initialized = false;
            throw std::invalid_argument(
                "XC requires 3D LDA/GGA or a global hybrid with EXC/VXC/FXC; "
                "meta-GGA, range separation and nonlocal correlation are unsupported: " +
                name);
        }
        if (hybrid)
            exact_exchange = xc_hyb_exx_coef(&function);
        // Global hybrids use the same semilocal Libxc evaluation routines.
        if (family == XC_FAMILY_HYB_LDA)
            family = XC_FAMILY_LDA;
        if (family == XC_FAMILY_HYB_GGA)
            family = XC_FAMILY_GGA;
    }

    ~Component() {
        if (initialized)
            xc_func_end(&function);
    }
    Component(const Component&) = delete;
    Component& operator=(const Component&) = delete;
};

} // namespace

struct Functional::Impl {
    int n_spin = 1;
    bool gga = false;
    double exact_exchange = 0.0;
    std::vector<std::unique_ptr<Component>> components;
};

Functional::Functional(const std::string& name, int n_spin) : impl_(std::make_unique<Impl>()) {
    if (n_spin != 1 && n_spin != 2) {
        throw std::invalid_argument("Libxc n_spin must be 1 or 2");
    }
    impl_->n_spin = n_spin;
    const int polarization = n_spin == 1 ? XC_UNPOLARIZED : XC_POLARIZED;
    const std::string resolved = canonical_name(name);
    std::size_t first = 0;
    for (;;) {
        const auto comma = resolved.find(',', first);
        const std::string token = trim(resolved.substr(first, comma - first));
        if (token.empty()) {
            throw std::invalid_argument("empty Libxc XC component");
        }
        const int identifier = xc_functional_get_number(token.c_str());
        if (identifier <= 0) {
            throw std::invalid_argument("unknown Libxc functional: " + token);
        }
        auto component = std::make_unique<Component>(token, identifier, polarization);
        impl_->gga = impl_->gga || component->family == XC_FAMILY_GGA;
        impl_->exact_exchange += component->exact_exchange;
        impl_->components.push_back(std::move(component));
        if (comma == std::string::npos)
            break;
        first = comma + 1;
    }
}

Functional::~Functional() = default;
Functional::Functional(Functional&&) noexcept = default;
Functional& Functional::operator=(Functional&&) noexcept = default;

std::string Functional::canonical_name(const std::string& name) {
    std::string resolved = trim(name);
    std::transform(resolved.begin(), resolved.end(), resolved.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    using Alias = std::pair<std::string_view, std::string_view>;
    static constexpr Alias aliases[] = {{"LDA", "LDA_X,LDA_C_VWN"},
                                        {"SVWN", "LDA_X,LDA_C_VWN"},
                                        {"SVWN5", "LDA_X,LDA_C_VWN"},
                                        {"PBE", "GGA_X_PBE,GGA_C_PBE"},
                                        {"BLYP", "GGA_X_B88,GGA_C_LYP"},
                                        {"BP86", "GGA_X_B88,GGA_C_P86"},
                                        {"PW91", "GGA_X_PW91,GGA_C_PW91"},
                                        {"REVPBE", "GGA_X_PBE_R,GGA_C_PBE"},
                                        {"PBESOL", "GGA_X_PBE_SOL,GGA_C_PBE_SOL"},
                                        {"PBE0", "HYB_GGA_XC_PBEH"},
                                        {"PBEH", "HYB_GGA_XC_PBEH"},
                                        {"B3LYP", "HYB_GGA_XC_B3LYP"},
                                        {"B3LYP5", "HYB_GGA_XC_B3LYP5"}};
    for (const auto& [alias, component] : aliases)
        if (resolved == alias)
            return std::string(component);
    return resolved;
}

std::string Functional::library_version() {
    int major = 0, minor = 0, micro = 0;
    xc_version(&major, &minor, &micro);
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(micro);
}

int Functional::n_spin() const {
    if (!impl_)
        throw std::invalid_argument("moved-from XC functional");
    return impl_->n_spin;
}

bool Functional::is_gga() const {
    if (!impl_)
        throw std::invalid_argument("moved-from XC functional");
    return impl_->gga;
}

double Functional::exact_exchange() const {
    if (!impl_)
        throw std::invalid_argument("moved-from XC functional");
    return impl_->exact_exchange;
}

void Functional::evaluate(const Eigen::Ref<const Eigen::MatrixXd>& rho,
                          const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                          XcValues* out) const {
    if (!impl_ || !out) {
        throw std::invalid_argument("invalid XC functional or output");
    }
    const int n_spin = impl_->n_spin;
    const int n_sigma = n_spin == 1 ? 1 : 3;
    const Eigen::Index n_points = rho.rows();
    validate_inputs(rho, sigma, n_spin, impl_->gga);
    std::vector<double> packed_rho(static_cast<std::size_t>(n_points) * n_spin);
    std::vector<double> packed_sigma;
    if (impl_->gga) {
        packed_sigma.resize(static_cast<std::size_t>(n_points) * n_sigma);
    }
    pack_inputs(
        rho, sigma, n_spin, impl_->gga, 0, n_points, packed_rho.data(), packed_sigma.data());

    out->exc.setZero(n_points);
    out->vrho.setZero(n_points, n_spin);
    out->vsigma.setZero(n_points, impl_->gga ? n_sigma : 0);
    if (n_points == 0)
        return;

    std::vector<double> component_exc(static_cast<std::size_t>(n_points));
    std::vector<double> component_vrho(static_cast<std::size_t>(n_points) * n_spin);
    std::vector<double> component_vsigma;
    if (impl_->gga) {
        component_vsigma.resize(static_cast<std::size_t>(n_points) * n_sigma);
    }
    for (const auto& component : impl_->components) {
        if (component->family == XC_FAMILY_GGA) {
            xc_gga_exc_vxc(&component->function,
                           static_cast<std::size_t>(n_points),
                           packed_rho.data(),
                           packed_sigma.data(),
                           component_exc.data(),
                           component_vrho.data(),
                           component_vsigma.data());
        } else {
            xc_lda_exc_vxc(&component->function,
                           static_cast<std::size_t>(n_points),
                           packed_rho.data(),
                           component_exc.data(),
                           component_vrho.data());
        }
        for (Eigen::Index i = 0; i < n_points; ++i) {
            const std::size_t point = static_cast<std::size_t>(i);
            if (!std::isfinite(component_exc[point])) {
                throw std::runtime_error("Libxc returned nonfinite XC energy");
            }
            out->exc[i] += component_exc[point];
            for (int spin = 0; spin < n_spin; ++spin) {
                const double value = component_vrho[point * n_spin + spin];
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Libxc returned nonfinite XC vrho");
                }
                out->vrho(i, spin) += value;
            }
            if (component->family == XC_FAMILY_GGA) {
                for (int channel = 0; channel < n_sigma; ++channel) {
                    const double value = component_vsigma[point * n_sigma + channel];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("Libxc returned nonfinite XC vsigma");
                    }
                    out->vsigma(i, channel) += value;
                }
            }
        }
    }
}

void Functional::kernel(const Eigen::Ref<const Eigen::MatrixXd>& rho,
                        const Eigen::Ref<const Eigen::MatrixXd>& sigma,
                        XcKernel* out) const {
    if (!impl_ || !out) {
        throw std::invalid_argument("invalid XC functional or response output");
    }
    const int n_spin = impl_->n_spin;
    const int n_sigma = n_spin == 1 ? 1 : 3;
    const Eigen::Index n_points = rho.rows();
    validate_inputs(rho, sigma, n_spin, impl_->gga);
    for (const auto& component : impl_->components) {
        const auto* info = xc_func_get_info(&component->function);
        if (!(xc_func_info_get_flags(info) & XC_FLAGS_HAVE_FXC)) {
            throw std::invalid_argument("XC component has no Libxc FXC derivative");
        }
    }

    out->n_spin = n_spin;
    out->rho2.setZero(n_points, n_spin == 1 ? 1 : 3);
    out->rho_sigma.setZero(n_points, impl_->gga ? n_spin * n_sigma : 0);
    out->sigma2.setZero(n_points, impl_->gga ? (n_spin == 1 ? 1 : 6) : 0);
    if (n_points == 0)
        return;

    // Libxc stores spin-polarized derivatives as
    // rho2=(aa,ab,bb), rho-sigma=(a:aa,ab,bb; b:aa,ab,bb),
    // sigma2=(aa-aa,aa-ab,aa-bb,ab-ab,ab-bb,bb-bb).
    constexpr Eigen::Index tile_size = 128;
    const Eigen::Index capacity = std::min(tile_size, n_points);
    const int n_rho2 = n_spin == 1 ? 1 : 3;
    const int n_rhosigma = n_spin * n_sigma;
    const int n_sigma2 = n_spin == 1 ? 1 : 6;
    std::vector<double> packed_rho(static_cast<std::size_t>(capacity) * n_spin);
    std::vector<double> packed_sigma;
    std::vector<double> v2rho2(static_cast<std::size_t>(capacity) * n_rho2);
    std::vector<double> v2rhosigma;
    std::vector<double> v2sigma2;
    if (impl_->gga) {
        packed_sigma.resize(static_cast<std::size_t>(capacity) * n_sigma);
        v2rhosigma.resize(static_cast<std::size_t>(capacity) * n_rhosigma);
        v2sigma2.resize(static_cast<std::size_t>(capacity) * n_sigma2);
    }

    for (Eigen::Index first = 0; first < n_points; first += tile_size) {
        const Eigen::Index count = std::min(tile_size, n_points - first);
        pack_inputs(
            rho, sigma, n_spin, impl_->gga, first, count, packed_rho.data(), packed_sigma.data());
        for (const auto& component : impl_->components) {
            if (component->family == XC_FAMILY_GGA) {
                xc_gga_fxc(&component->function,
                           static_cast<std::size_t>(count),
                           packed_rho.data(),
                           packed_sigma.data(),
                           v2rho2.data(),
                           v2rhosigma.data(),
                           v2sigma2.data());
            } else {
                xc_lda_fxc(&component->function,
                           static_cast<std::size_t>(count),
                           packed_rho.data(),
                           v2rho2.data());
            }
            for (Eigen::Index local = 0; local < count; ++local) {
                const Eigen::Index point = first + local;
                const std::size_t offset = static_cast<std::size_t>(local);
                const double* rr = v2rho2.data() + offset * n_rho2;
                for (int index = 0; index < n_rho2; ++index) {
                    if (!std::isfinite(rr[index])) {
                        throw std::runtime_error("Libxc returned nonfinite XC v2rho2");
                    }
                    out->rho2(point, index) += rr[index];
                }
                if (component->family != XC_FAMILY_GGA)
                    continue;
                const double* rs = v2rhosigma.data() + offset * n_rhosigma;
                const double* ss = v2sigma2.data() + offset * n_sigma2;
                for (int index = 0; index < n_rhosigma; ++index) {
                    if (!std::isfinite(rs[index])) {
                        throw std::runtime_error("Libxc returned nonfinite XC v2rhosigma");
                    }
                    out->rho_sigma(point, index) += rs[index];
                }
                for (int index = 0; index < n_sigma2; ++index) {
                    if (!std::isfinite(ss[index])) {
                        throw std::runtime_error("Libxc returned nonfinite XC v2sigma2");
                    }
                    out->sigma2(point, index) += ss[index];
                }
            }
        }
    }
}

void XcKernel::apply(const Eigen::Ref<const Eigen::MatrixXd>& drho,
                     const Eigen::Ref<const Eigen::MatrixXd>& dsigma,
                     Eigen::MatrixXd* dvrho,
                     Eigen::MatrixXd* dvsigma) const {
    const Eigen::Index rows = rho2.rows();
    const bool gga = sigma2.cols() != 0;
    const int ns = n_spin == 1 ? 1 : 3;
    if (!dvrho || !dvsigma || dvrho == dvsigma || drho.rows() != rows || drho.cols() != n_spin ||
        !drho.allFinite() || dsigma.rows() != rows ||
        (gga && (dsigma.cols() != ns || !dsigma.allFinite())))
        throw std::invalid_argument("XC response direction shape or value mismatch");
    dvrho->setZero(rows, n_spin);
    dvsigma->setZero(rows, gga ? ns : 0);
    constexpr int rr_index[2][2] = {{0, 1}, {1, 2}};
    constexpr int ss_index[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    for (Eigen::Index p = 0; p < rows; ++p) {
        for (int s = 0; s < n_spin; ++s) {
            for (int t = 0; t < n_spin; ++t)
                (*dvrho)(p, s) += rho2(p, n_spin == 1 ? 0 : rr_index[s][t]) * drho(p, t);
            if (gga)
                for (int a = 0; a < ns; ++a)
                    (*dvrho)(p, s) += rho_sigma(p, s * ns + a) * dsigma(p, a);
        }
        if (!gga)
            continue;
        for (int a = 0; a < ns; ++a) {
            for (int s = 0; s < n_spin; ++s)
                (*dvsigma)(p, a) += rho_sigma(p, s * ns + a) * drho(p, s);
            for (int b = 0; b < ns; ++b)
                (*dvsigma)(p, a) += sigma2(p, n_spin == 1 ? 0 : ss_index[a][b]) * dsigma(p, b);
        }
    }
}

} // namespace xmvb::dft
