#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <chrono>

#include <mp-units/systems/si/si.h>

#include <scope_guard.hpp>

#define defer DEFER

namespace trace_utils {
namespace utils {
std::string random_string(std::size_t length);

mp_units::quantity<mp_units::si::second> parse_duration(const std::string& duration_str);

inline std::chrono::time_point<std::chrono::steady_clock> get_time() {
    return std::chrono::steady_clock::now();
}

template<typename Func>
auto get_time(Func&& func) {
    auto start = get_time();
    func();
    auto end = get_time();
    return end - start;
}

using f_sec = std::chrono::duration<float>;
} // namespace utils
} // namespace trace_utils

#endif
