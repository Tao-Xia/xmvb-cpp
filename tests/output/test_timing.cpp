#include "output/text/report.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {
class FlushBuffer : public std::stringbuf {
  public:
    int flushes = 0;
    int sync() override {
        ++flushes;
        return std::stringbuf::sync();
    }
};
} // namespace

int main() {
    FlushBuffer buffer;
    std::ostream output(&buffer);
    output << std::scientific << std::setprecision(3);
    const auto flags = output.flags();
    xmvb::output::print_scf_convergence_marker(output, 12.345678901);
    if (buffer.str() != "SCF_CONVERGED program_wall_seconds=12.345678901\n" || buffer.flushes != 1)
        throw std::runtime_error("convergence marker must be complete and immediately flushed");
    if (output.flags() != flags || output.precision() != 3)
        throw std::runtime_error("convergence marker changed report formatting");
}
