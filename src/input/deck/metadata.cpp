#include "input/deck/metadata.hpp"

#include "input/deck/model.hpp"

namespace xmvb::vb {

InputDeckMetadata parse_input_deck_metadata(
    const std::string& input_file_path) {
  return parse_input_deck_model(input_file_path).metadata;
}

}  // namespace xmvb::vb
