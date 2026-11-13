#pragma once

#include <Eigen/Core>

#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vbscf/structures/expansion/types.hpp"

namespace xmvb::output {

/** Display order and unique-spin text shared by all reported CI roots.
 * Coefficients are consumed one row at a time, without a product-sized
 * numerical or text array. Repeated spin strings are decoded only once.
 */
class DeterminantReport {
  public:
    DeterminantReport(const vb::StructureExpansion& expansion, int n_inactive);
    /** Coefficients refer to the original expansion, before report normalization. */
    void write(std::ostream& output, const Eigen::Ref<const Eigen::VectorXd>& coefficients);

  private:
    std::string_view spin_text(const mcscf::StringTable& strings,
                               std::size_t id,
                               std::vector<std::optional<std::string>>& cache);

    const vb::StructureExpansion& expansion_;
    int n_inactive_;
    bool shared_spin_strings_;
    std::vector<std::size_t> display_order_;
    std::vector<std::optional<std::string>> alpha_text_, beta_text_;
    std::string scratch_;
};

} // namespace xmvb::output
