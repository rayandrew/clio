#include <trace-utils/trace/msft.hpp>

#include <fmt/std.h>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"

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

std::vector<std::string> Entry::to_vec() const {
    return {
        std::to_string(timestamp),
        hostname,
        std::to_string(disk_id),
        type,
        std::to_string(offset),
        std::to_string(size),
        std::to_string(response_time),
    };
}
} // namespace msft
    
void MsftTrace::raw_stream(const fs::path& path, RawReadFn&& read_fn) const {

}

void MsftTrace::raw_stream_column(const fs::path& path,
                                  unsigned int column,
                                  RawReadColumnFn&& read_fn) const {

}
} // namespace trace_utils::trace
