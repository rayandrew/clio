#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/logger.hpp>

#include <cstdint>

#include <fmt/base.h>
#include <fmt/format.h>
#include <scope_guard.hpp>

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
    read_tar_gz_csv(path, [&](auto line) {
        io::CSVReader<5> csv{"", line.begin(), line.end()};
        TencentTrace::Entry entry;
        csv.read_row(entry.timestamp, entry.offset, entry.size, entry.read, entry.volume_id);
        read_fn(entry);
    });
}
} // namespace trace_utils::trace
