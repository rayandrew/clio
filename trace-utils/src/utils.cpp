#include "utils.hpp"

#include <algorithm>

#include <fmt/format.h>
#include <fmt/std.h>

#include <trace-utils/exception.hpp>
#include <trace-utils/logger.hpp>

#include "csv.hpp"

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
} // namespace trace_utils
