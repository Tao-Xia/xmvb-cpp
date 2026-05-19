#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace xmvb::vb {

inline bool parse_env_flag_with_default(const char* name, bool default_value) {
  if (name == nullptr || name[0] == '\0') return default_value;
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  // Standard convention: "0"/"false"/"FALSE" → false; anything else → true
  return std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "false") != 0 &&
      std::strcmp(value, "FALSE") != 0;
}

inline int parse_env_int_with_default(const char* name, int default_value) {
  if (name == nullptr || name[0] == '\0') return default_value;
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || (end != nullptr && *end != '\0') || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

inline double parse_env_double_with_default(const char* name, double default_value) {
  if (name == nullptr || name[0] == '\0') return default_value;
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || (end != nullptr && end[0] != '\0') || !std::isfinite(parsed))
    return default_value;
  return parsed;
}

}  // namespace xmvb::vb
