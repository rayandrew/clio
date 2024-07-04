#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <algorithm>
#include <type_traits>
#include <utility>
#include <string_view>

#include <archive.h>
#include <archive_entry.h>

#include <scope_guard.hpp>

#include <mp-units/format.h>
#include <mp-units/systems/iec80000/iec80000.h>

#include <trace-utils/internal/filesystem.hpp>
#include <trace-utils/logger.hpp>

#define defer DEFER

namespace trace_utils {
namespace internal {
fs::path get_exe_path();
    
template<typename Func>
inline int archive_read_data_callback(struct archive *ar, struct archive_entry *entry, Func&& func) {
    int r;
    size_t size;
    const void *buff;
    int64_t offset;

    for (;;) {
        r = archive_read_data_block(ar, &buff, &size, &offset);
        if (size > 0) {
            func(entry, buff, size, offset);
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
void read_tar_gz(const fs::path& path, Func&& func) {
    using namespace mp_units;
    using namespace mp_units::iec80000::unit_symbols;
    
    struct archive *a;
    struct archive_entry *entry;
    int flags, r;

    /* Select which attributes we want to restore. */
    flags = ARCHIVE_EXTRACT_TIME;
    flags |= ARCHIVE_EXTRACT_UNLINK;
    flags |= ARCHIVE_EXTRACT_SECURE_NODOTDOT;

    a = archive_read_new();
    if (!a) { throw Exception("Cannot archive_read_new"); }
    defer { archive_read_free(a); };
    archive_read_support_format_tar(a);
    archive_read_support_filter_gzip(a);

    if ((r = archive_read_open_filename(a, path.string().c_str(), 10240))) {
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
            log()->info("Reading {} with size {}", archive_entry_pathname(entry), size_gbytes);
            r = internal::archive_read_data_callback(a, entry, std::forward<Func>(func));
            if (r < ARCHIVE_OK) {
                log()->error("Encountered error while reading data");
            }
        }
    }
}

template<typename Func>
void read_tar_gz_csv(const fs::path& path, Func&& func) {
    char leftovers[512] = {0};
    std::size_t length_leftovers = 0;
    std::size_t count = 0;
    read_tar_gz(path, [&](auto* entry, const auto* buffer, [[maybe_unused]] auto size, [[maybe_unused]] auto offset) {
        auto leftover_string = std::string{leftovers, leftovers + length_leftovers};
        auto buffer_string = leftover_string + std::string{reinterpret_cast<const char*>(buffer)};
        auto sv = std::string_view{buffer_string};

        std::size_t start = 0;
        std::size_t end = sv.find('\n');

        while (end != std::string_view::npos) {
            auto line = sv.substr(start, end - start);
            count += 1;
            func(line, count, entry);
            start = end + 1;
            end = sv.find('\n', start);
        }
        
         // reset leftovers buffer
        memset(leftovers, 0, sizeof(leftovers));
        length_leftovers = 0;

        // copy new leftovers
        auto leftover_data = sv.substr(start);
        memcpy(leftovers, leftover_data.data(), leftover_data.size());
        length_leftovers = leftover_data.size();
    });
}
} // namespace trace_utils

#endif
