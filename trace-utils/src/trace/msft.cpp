#include <trace-utils/trace/msft.hpp>
#include <trace-utils/logger.hpp>

#include <cstdint>

#include <fmt/base.h>
#include <fmt/format.h>
#include <bit7z/bitfileextractor.hpp>
#include <scope_guard.hpp>


#include "../utils.hpp"

namespace trace_utils::trace {
void MsftTrace::read(const char* filename) {
    auto lib = internal::get_bit7z_lib();
    bit7z::BitFileExtractor extractor{lib, bit7z::BitFormat::GZip};
    extract_tar_gz_to_memory(filename, [&](auto item) {
        // fmt::print("\n");
        // fmt::print("Item index: {}\n", item.index());
        // fmt::print("    Name: {}\n", item.name());
        // fmt::print("    Extension: {}\n", item.extension());
        // fmt::print("    Path: {}\n", item.path());
        // fmt::print("    IsDir: {}\n", item.isDir());
        // fmt::print("    Size: {}\n", item.size());
        // fmt::print("    Packed size: {}\n", item.packSize());
        // fmt::print("    CRC: {:x}\n", item.crc());
        /* log()->info("\n"); */
        log()->info("Item index: {}", item.index());
        log()->info("    Name: {}", item.name());
        log()->info("    Extension: {}", item.extension());
        log()->info("    Path: {}", item.path());
        log()->info("    IsDir: {}", item.isDir());
        log()->info("    Size: {}", item.size());
        log()->info("    Packed size: {}", item.packSize());
        log()->info("    CRC: {:x}", item.crc());
        std::vector<std::uint8_t> buffer;
        extractor.extract(filename, buffer, item.index());
        
    });
}
} // namespace trace_utils::trace
