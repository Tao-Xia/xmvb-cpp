#pragma once

namespace xmvb::vb {
struct InputDeck;
}
namespace xmvb::cli {
struct Options;
int run_dft(const Options& command_line, const vb::InputDeck& deck);
} // namespace xmvb::cli
