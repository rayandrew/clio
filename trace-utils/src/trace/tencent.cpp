#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/logger.hpp>

#include <cstdint>
#include <string_view>
#include <array>

#include <fmt/base.h>
#include <fmt/format.h>
#include <scope_guard.hpp>

#include "../csv.hpp"
#include "../utils.hpp"


namespace trace_utils::trace {
void TencentTrace::read(const char* filename) {
    char leftovers[128] = {0};
    std::size_t length_leftovers = 0;
    extract_tar_gz_to_memory(filename, [&](const auto* buffer, auto size, auto offset) {
        // log()->info("Reading buffer with size={} and offset={}", size, offset);

        auto leftover_string = std::string{leftovers, leftovers + length_leftovers};
        auto buffer_string = leftover_string + std::string{reinterpret_cast<const char*>(buffer)};
        auto sv = std::string_view{buffer_string};

        std::size_t start = 0;
        std::size_t end = sv.find('\n');

        while (end != std::string_view::npos) {
            auto line = sv.substr(start, end - start);
            io::CSVReader<5> csv{"", line.begin(), line.end()};
            auto entry = TencentTrace::Entry{};
            int read;
            csv.read_row(entry.timestamp, entry.offset, entry.size, read, entry.volume_id);
            entry.read = read == 0;
            data.push_back(entry);
            start = end + 1;
            end = sv.find('\n', start);
        }
        memset(leftovers, 0, sizeof(leftovers));
        length_leftovers = 0; // reset
        auto leftover_data = sv.substr(start);
        memcpy(leftovers, leftover_data.data(), leftover_data.size());
        length_leftovers = leftover_data.size();
    });
}
} // namespace trace_utils::trace
