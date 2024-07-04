#include <trace-utils/trace/tencent.hpp>

#include <cstdint>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>

#include <trace-utils/logger.hpp>

#include "../csv.hpp"
#include "../utils.hpp"

namespace trace_utils::trace {
namespace tencent {
trace::Entry Entry::convert() const {
    trace::Entry entry;
    entry.timestamp = timestamp * 1e3;
    entry.disk_id = volume_id;
    entry.offset = offset * 512;
    entry.size = size * 512;
    entry.read = read == 0;
    return entry;
}
} // namespace tencent
    
void TencentTrace::raw_stream(const fs::path& path, RawReadFn&& read_fn) const {
    read_tar_gz_csv(path, [&](auto line, auto line_count, auto* entry) {
        try {
            io::CSVReader<5> csv{archive_entry_pathname(entry), line.begin(), line.end()};
            TencentTrace::Entry entry;
            csv.read_row(entry.timestamp, entry.offset, entry.size, entry.read, entry.volume_id);
            read_fn(entry);
        } catch (const std::exception& ex) {
            log()->error("Skipping line due to cannot parse at line {} in file {} with archive path {}", line_count, path, archive_entry_pathname(entry));
            log()->error("   Message: {}", ex.what());
        }
    });
}
} // namespace trace_utils::trace
