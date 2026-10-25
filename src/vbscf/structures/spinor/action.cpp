#include "vbscf/structures/spinor/action.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace xmvb::vb::spinor {

Action::Action(Basis basis, Transform transform, Sigma sigma, std::size_t workspace_bytes)
    : basis_(std::move(basis)), transform_(std::move(transform)), sigma_(std::move(sigma)),
      workspace_bytes_(workspace_bytes) {
    if (basis_.strings() != transform_.strings() || basis_.strings() != sigma_.strings() ||
        workspace_bytes_ == 0) {
        throw std::invalid_argument("spinor action carrier or workspace mismatch");
    }
}

Images Action::apply(const Eigen::Ref<const Eigen::MatrixXcd>& block) const {
    if (block.rows() != dimension() || block.cols() <= 0 || !block.allFinite()) {
        throw std::invalid_argument("invalid selected spinor action block");
    }
    // Conservative simultaneously-live amplitude and result allowance. Static
    // input storage (B, factors, exact ERI, strings) is not a workspace allocation.
    const std::size_t rows = static_cast<std::size_t>(basis_.strings()->dimension());
    const std::size_t structures = static_cast<std::size_t>(dimension());
    const std::size_t columns = static_cast<std::size_t>(block.cols());
    const std::size_t n = static_cast<std::size_t>(basis_.strings()->n_spinor());
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (rows > maximum / 8 || structures > (maximum - 8 * rows) / 2) {
        throw std::length_error("spinor workspace estimate overflows");
    }
    const std::size_t elements = 8 * rows + 2 * structures;
    // RI counterterm and matrix-product scratch; n is bounded to 63 labels.
    const std::size_t scratch = 2 * n * n * sizeof(std::complex<double>);
    if (columns > maximum / elements / sizeof(std::complex<double>) || scratch > workspace_bytes_ ||
        columns * elements * sizeof(std::complex<double>) > workspace_bytes_ - scratch) {
        throw std::length_error("spinor carrier block exceeds workspace budget");
    }

    Eigen::MatrixXcd image = basis_.expand(block);
    transform_.apply(image);
    Eigen::MatrixXcd h_image = sigma_.apply(image);
    transform_.apply_adjoint(h_image);
    transform_.apply_adjoint(image);
    return {basis_.contract(h_image), basis_.contract(image)};
}

} // namespace xmvb::vb::spinor
