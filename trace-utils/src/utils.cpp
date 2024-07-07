#include "utils.hpp"

#include <algorithm>
#include <string>
#include <random>

#include <fmt/format.h>
#include <fmt/std.h>

#include <trace-utils/exception.hpp>
#include <trace-utils/logger.hpp>

namespace trace_utils {
namespace internal {
static fs::path path_dir_exe = "invalid";
static fs::path path_exe = "invalid";
    
fs::path get_exe_path() {
    if (path_exe == "invalid") {
        path_exe = fs::canonical("/proc/self/exe");
    }
    return path_exe;
}

fs::path get_dir_exe_path() {
    if (path_dir_exe == "invalid") {
        path_dir_exe = get_exe_path().parent_path();
    }
    return path_dir_exe;
}
    
std::string clean_control_characters(std::string_view sv) {
    std::string cleaned;
    cleaned.reserve(sv.size());
    std::copy_if(sv.begin(), sv.end(), std::back_inserter(cleaned), [](unsigned char c) {
        return !std::iscntrl(c) || c == '\n';
        // return std::isprint(c) || c == '\n' || c == ',';
    });
    return cleaned;
}     
} // namespace internal

namespace utils {
// CC-BY-SA 4.0
// Taken from https://stackoverflow.com/a/50556436/2418586
std::string random_string(std::size_t length) {
    std::mt19937 generator{std::random_device{}()};
   //modify range according to your need "A-Z","a-z" or "0-9" or whatever you need.
    std::uniform_int_distribution<int> distribution{'a', 'z'};

    std::string rand_str(length, '\0');
    for(auto& dis: rand_str)
        dis = distribution(generator);

    return rand_str;
}
} // namespace utils
} // namespace trace_utils
