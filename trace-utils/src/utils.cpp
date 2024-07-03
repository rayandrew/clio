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
static fs::path path_7zip_shared_library = "invalid";
    
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

    
// fs::path get_7zip_lib_path() {
//     if (path_7zip_shared_library == "invalid") {
//         auto p = fs::weakly_canonical(get_dir_exe_path() / "../lib/lib7zip.so");
//         if (fs::exists(p)) {
//             path_7zip_shared_library = p;
//         } else {
//             auto p = fs::weakly_canonical(get_dir_exe_path() / "../build/_deps/7zip-src/lib7zip.so");
//             if (!fs::exists(p)) {
//                 throw Exception("Error, cannot find lib7zip.so");
//             }

//             path_7zip_shared_library = p;
            
//         }
//         log()->info("Find 7zip shared library = {}", path_7zip_shared_library);
//     }
//     return path_7zip_shared_library;
// }

// bit7z::Bit7zLibrary get_bit7z_lib() {
//     return bit7z::Bit7zLibrary{internal::get_7zip_lib_path()};
// }
}
    
void read_csv(const fs::path& path) {

}
} // namespace trace_utils
