#include "mcscf/ci/workspace.hpp"

#include <optional>

#include "mcscf/ci/scratch.hpp"

namespace xmvb::mcscf {
using detail::ChannelWorkspace;

struct CiWorkspace::Impl {
    std::optional<ChannelWorkspace> sigma;
    std::optional<workspace::Scope> scope;
    ChannelWorkspace* previous = nullptr;
    std::size_t bytes = 0;
    explicit Impl(std::size_t requested, bool use_sigma) {
        if (workspace::has_scope()) {
            bytes = std::min(requested, workspace::available());
            return;
        }
        workspace::Reservation reusable;
        if (use_sigma)
            sigma.emplace(requested, true);
        scope.emplace(requested, sigma ? sigma->residency() : reusable);
        bytes = scope->bytes();
        if (sigma)
            previous = ChannelWorkspace::bind(&*sigma);
    }
    ~Impl() {
        if (sigma) {
            sigma->retain_idle();
            ChannelWorkspace::bind(previous);
        }
    }
};
CiWorkspace::CiWorkspace(std::size_t bytes, bool use_sigma)
    : impl_(std::make_unique<Impl>(bytes, use_sigma)) {}
CiWorkspace::~CiWorkspace() = default;
std::size_t CiWorkspace::bytes() const noexcept {
    return impl_->bytes;
}
void CiWorkspace::sigma_budget(std::size_t bytes) const {
    ChannelWorkspace::limit_action(bytes);
}

} // namespace xmvb::mcscf
