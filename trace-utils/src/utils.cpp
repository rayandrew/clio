#include "utils.hpp"

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
}
} // namespace trace_utils
