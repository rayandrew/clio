#ifndef __TRACE_UTILS_PRIV_UTILS_HPP__
#define __TRACE_UTILS_PRIV_UTILS_HPP__

#include <algorithm>
#include <type_traits>
#include <utility>
#include <string_view>

#include <archive.h>
#include <archive_entry.h>

#include <mp-units/format.h>
#include <mp-units/systems/iec80000/iec80000.h>

#include <trace-utils/utils.hpp>
#include <trace-utils/internal/filesystem.hpp>
#include <trace-utils/logger.hpp>

namespace trace_utils {

using namespace mp_units;
using namespace mp_units::iec80000::unit_symbols; 

namespace internal {
fs::path get_exe_path();

std::string clean_control_characters(std::string_view sv);

template<typename Func>
inline int archive_read_data_callback(struct archive *ar, struct archive_entry *entry, Func&& func) {
    int r;
    size_t size;
    const void *buff;
    int64_t offset;

    for (;;) {
        r = archive_read_data_block(ar, &buff, &size, &offset);
        if (size > 0) {
            func(entry, buff, size);
        }
        if (r == ARCHIVE_EOF) {
            return ARCHIVE_OK;
        } else if (r < ARCHIVE_OK) {
            return r;
        }
    }
}
}

template<typename Func>
void process_block(const char* block,
                   size_t block_size,
                   std::string& buffer,
                   Func&& callback) {
    buffer.append(block, block_size);
    std::string_view full_buffer(buffer);
    std::string clean_full_buffer_str = internal::clean_control_characters(full_buffer);
    std::string_view clean_full_buffer(clean_full_buffer_str);

    std::size_t last_newline = clean_full_buffer.rfind('\n');

    if (last_newline != std::string_view::npos) {
        std::string_view part = clean_full_buffer.substr(0, last_newline + 1);
        if (!part.empty()) {
            callback(part);
        }
        buffer = clean_full_buffer.substr(last_newline + 1);
    } else {
        buffer = clean_full_buffer_str;
    }
}

template<typename Func, QuantityOf<mp_units::iec80000::storage_size> Size>
void read_tar_gz(const fs::path& path,
                 Func&& func,
                 Size block_size) {
    auto block_size_bytes = block_size.numerical_value_in(mp_units::iec80000::byte);
    auto block_size_bytes_size_t = static_cast<std::size_t>(block_size_bytes);
    // log()->info("Reading path = {} with block size = {}", path, block_size);
    struct archive *a;
    struct archive_entry *entry;
    int flags, r;

    flags = ARCHIVE_EXTRACT_TIME;
    flags |= ARCHIVE_EXTRACT_UNLINK;
    flags |= ARCHIVE_EXTRACT_SECURE_NODOTDOT;

    a = archive_read_new();
    if (!a) { throw Exception("Cannot archive_read_new"); }
    defer { archive_read_free(a); };
    archive_read_support_format_tar(a);
    archive_read_support_filter_gzip(a);

    if ((r = archive_read_open_filename(a, path.string().c_str(), block_size_bytes_size_t))) {
        throw Exception("Cannot archive_read_open_filename");
    }
    defer { archive_read_close(a); };

    for (;;) {
        r = archive_read_next_header(a, &entry);
        if (r == ARCHIVE_EOF) {
            break;
        }
        else if (archive_entry_size(entry) > 0) {
            auto size_gbytes = (static_cast<float>(archive_entry_size(entry)) * B).in(GB);
            log()->debug("Reading {} with size {}", archive_entry_pathname(entry), size_gbytes);
            r = internal::archive_read_data_callback(a, entry, std::forward<Func>(func));
            if (r < ARCHIVE_OK) {
                log()->error("Encountered error while reading data");
            }
        }
    }
}

template<typename Func>
void read_tar_gz(const fs::path& path,
                 Func&& func) {
    read_tar_gz(path, func, 1000 * MB);
}

template<typename Func, QuantityOf<mp_units::iec80000::storage_size> Size>
void read_tar_gz_csv(const fs::path& path,
                     Func&& func,
                     Size block_size) {
    auto block_size_bytes = static_cast<std::size_t>(block_size.numerical_value_in(mp_units::iec80000::byte));
    std::size_t count = 0;
    std::string buffer;
    buffer.reserve(block_size_bytes + 1);
    read_tar_gz(path, [&](auto* entry, const auto* block, auto size) {
        process_block(
            reinterpret_cast<const char*>(block),
            size,
            buffer,
            [&](auto blk) {
                count += 1;
                func(blk, count, entry);
            });
    }, block_size);
}

template<typename Func>
void read_tar_gz_csv(const fs::path& path,
                     Func&& func) {
    read_tar_gz_csv(path, func, 1000 * MB);
}
} // namespace trace_utils

#endif
