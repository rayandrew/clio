#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <algorithm>

#include <bit7z/bitarchivereader.hpp>

#include <trace-utils/internal/filesystem.hpp>


namespace trace_utils {
namespace internal {
fs::path get_exe_path();
fs::path get_7zip_lib_path();
bit7z::Bit7zLibrary get_bit7z_lib();
}
    
template<typename Func>
void extract_tar_gz_to_memory(const char* filename, Func&& func) {
    auto lib = internal::get_bit7z_lib();
    bit7z::BitArchiveReader archive{lib, filename, bit7z::BitFormat::GZip};
    std::for_each(archive.begin(), archive.end(), func);
}
    

void read_csv(const fs::path& path);
} // namespace trace_utils

#endif
