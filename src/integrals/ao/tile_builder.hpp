#pragma once

#include "integrals/ao/tiles.hpp"

#include <unordered_map>

namespace xmvb::integrals {

/** Staging is confined to the block rows of one shell and released at finish. */
class AoPairTiles::Builder::Shell {
  public:
    Shell(Builder& builder, std::size_t shell);
    void add(std::size_t left, std::size_t right, double value);
    void finish();

  private:
    struct Pending {
        Tile tile;
        std::vector<std::uint16_t> coordinates;
        void add(int row, int col, int height, int width, bool diagonal, double value);
        Tile finish(int height, int width, bool diagonal);
    };
    AoPairTiles& tiles_;
    std::size_t begin_;
    std::vector<std::unordered_map<std::size_t, Pending>> rows_;
};

} // namespace xmvb::integrals
