#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <string_view>

#include <mp-units/systems/si/si.h>

#include <scope_guard.hpp>

#define defer DEFER

namespace trace_utils {
namespace utils {
std::string random_string(std::size_t length);

mp_units::quantity<mp_units::si::second> parse_duration(const std::string& duration_str);
} // namespace utils
} // namespace trace_utils

#endif
