#pragma once

namespace xmvb::vb {
struct InputDeck;
}

namespace xmvb::cli {
struct Options;

/** Run the exact, single-state CASSCF path selected by `$CTRL CASSCF`. */
int run_casscf(const Options& command_line, const vb::InputDeck& deck);

} // namespace xmvb::cli
