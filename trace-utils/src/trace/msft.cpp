#include <trace-utils/trace/msft.hpp>

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
    read_tar_gz_csv(path, [&](auto line) {
        io::CSVReader<7> csv{"", line.begin(), line.end()};
        MsftTrace::Entry entry;
        csv.read_row(entry.timestamp, entry.hostname, entry.disk_id, entry.type, entry.offset, entry.size, entry.response_time);
        read_fn(entry);
    });
}
} // namespace trace_utils::trace
