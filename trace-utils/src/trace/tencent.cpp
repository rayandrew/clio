#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/logger.hpp>

#include <cstdint>

#include <fmt/base.h>
#include <fmt/format.h>
#include <scope_guard.hpp>

#include "../csv.hpp"
#include "../utils.hpp"

namespace trace_utils::trace {
void TencentTrace::stream(const fs::path& path, ReadFn&& read_fn) const {
    read_tar_gz_csv(path, [&](auto line) {
        io::CSVReader<5> csv{"", line.begin(), line.end()};
        TencentTrace::Entry entry;
        int read;
        csv.read_row(entry.timestamp, entry.offset, entry.size, read, entry.volume_id);
        entry.read = read == 0; // flipped in Tencent
        read_fn(entry);
    });
}
} // namespace trace_utils::trace
