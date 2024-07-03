#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <algorithm>
#include <type_traits>
#include <utility>
#include <ostream>
#include <istream>
#include <streambuf>

// #include <bit7z/bitarchivereader.hpp>
// #include <bit7z/bitmemextractor.hpp>
// #include <bit7z/bitfileextractor.hpp>

#include <archive.h>
#include <archive_entry.h>

#include <scope_guard.hpp>

#include <trace-utils/internal/filesystem.hpp>
#include <trace-utils/logger.hpp>


namespace trace_utils {
namespace internal {
fs::path get_exe_path();
// fs::path get_7zip_lib_path();
// bit7z::Bit7zLibrary get_bit7z_lib();

// https://stackoverflow.com/a/13059195
struct membuf: std::streambuf {
    membuf(char const* base, size_t size) {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
    }
};

struct imemstream: virtual membuf, std::istream {
    imemstream(char const* base, size_t size)
        : membuf(base, size)
        , std::istream(static_cast<std::streambuf*>(this)) {
    }
};

struct omemstream: virtual membuf, std::ostream {
    omemstream(char const* base, size_t size)
        : membuf(base, size)
        , std::ostream(static_cast<std::streambuf*>(this)) {
    }
};
}

#define defer DEFER

template<typename Func>
inline int read_data(struct archive *ar, Func&& func) {
    int r;
    size_t size;
    const void *buff;
    int64_t offset;

    for (;;) {
        r = archive_read_data_block(ar, &buff, &size, &offset);
        func(buff, size, offset);
        if (r == ARCHIVE_EOF) {
            return ARCHIVE_OK;
        } else if (r < ARCHIVE_OK) {
            return r;
        }
    }
}

    
template<typename Func>
void extract_tar_gz_to_memory(const char* filename, Func&& func) {
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

    if ((r = archive_read_open_filename(a, filename, 10240))) {
        throw Exception("Cannot archive_read_open_filename");
    }
    defer { archive_read_close(a); };

    for (;;) {
        r = archive_read_next_header(a, &entry);
        if (r == ARCHIVE_EOF) {
            break;
        }
        else if (archive_entry_size(entry) > 0) {
            log()->info("Reading {} with size {}", archive_entry_pathname(entry), archive_entry_size(entry));
            r = read_data(a, func);
            if (r < ARCHIVE_OK) {
                log()->info("Encountered error while reading data");
            }
        }
    }
}
    
void read_csv(const fs::path& path);
} // namespace trace_utils

#endif
