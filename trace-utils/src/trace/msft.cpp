#include <trace-utils/trace/msft.hpp>

#include <fmt/std.h>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"
#include "../csv.hpp"

namespace trace_utils::trace {
namespace msft {
trace::Entry Entry::convert() const {
    trace::Entry entry;
    entry.timestamp = timestamp * 1e3;
    entry.disk_id = disk_id;
    entry.offset = offset;
    entry.size = size;
    entry.read = type == "Read";
    return entry;
}
} // namespace msft
    
void MsftTrace::raw_stream(const fs::path& path, RawReadFn&& read_fn) const {
    // read_tar_gz_csv(path, [&](auto line, auto line_count, auto* entry) {
    //     try {
    //         io::CSVReader<7> csv{"", line.cbegin(), line.cend()};
    //         MsftTrace::Entry entry;
    //         csv.read_row(entry.timestamp, entry.hostname, entry.disk_id, entry.type, entry.offset, entry.size, entry.response_time);
    //         read_fn(entry);
    //     } catch (const std::exception& ex) {
    //         log()->error("Skipping line due to cannot parse at line {} in file {} with archive path {}", line_count, path, archive_entry_pathname(entry));
    //         log()->error("   Message: {}", ex.what());
    //     }
    // });
}
} // namespace trace_utils::trace
