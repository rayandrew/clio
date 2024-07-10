#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <chrono>

#include <mp-units/systems/si/si.h>
#include <mp-units/systems/isq/isq.h>

#include <regex>

#include <scope_guard.hpp>

#include <trace-utils/exception.hpp>

#define defer DEFER

namespace trace_utils {
namespace utils {
std::string random_string(std::size_t length);

// mp_units::QuantityOf<mp_units::isq::time> decltype(auto)
template<mp_units::QuantityOf<mp_units::isq::time> D>
void parse_duration(const std::string& duration_str, D& dest) {
    using namespace mp_units;
    using namespace mp_units::si;
    using namespace mp_units::si::unit_symbols;

    constexpr auto q = R"((\d+\.?\d*)\s*([a-zA-Z]+))";
    static std::regex regex(q);
    std::smatch match;

    if (!std::regex_match(duration_str, match, regex)) {
        throw std::invalid_argument("Invalid duration format: " + duration_str);
    }

    // Extract the numerical value and unit from the regex match
    double value = std::stod(match[1].str());
    std::string unit = match[2].str();
    
    if (unit.find("ms") != std::string::npos) {
        dest = value * ms;
        return;
    }

    if (unit.find("us") != std::string::npos) {
        dest = value * us;
        return;
    }

    if (unit.find("ns") != std::string::npos) {
        dest = value * ns;
        return;
    }

    if (unit.find("s") != std::string::npos) {
        dest = value * second;
        return;
    }

    if (unit.find("m") != std::string::npos) {
        dest = value * minute;
        return;
    }

    if (unit.find("h") != std::string::npos) {
        dest = value * hour;
        return;
    }

    if (unit.find("d") != std::string::npos) {
        dest = value * day;
        return;
    }
    
    throw Exception(fmt::format("Unit {} is not defined!", unit));
}
    
// mp_units::quantity<mp_units::si::second> parse_duration(const std::string& duration_str);

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
