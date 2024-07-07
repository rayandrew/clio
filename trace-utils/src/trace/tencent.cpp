#include <trace-utils/trace/tencent.hpp>

#include <cstdint>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>
#include <csv2/reader.hpp>

#include <trace-utils/logger.hpp>

// #include "../csv.hpp"
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
    read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {
        using namespace csv2;
        std::string cell_value{""};

        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.parse_view(block)) {
            for (const auto row : csv) {
                TencentTrace::Entry entry;
                std::size_t col{0};
                for (const auto cell : row) {
                    col += 1;
                    cell.read_raw_value(cell_value);
                    switch (col) {
                    case 1:
                        entry.timestamp = std::stof(cell_value);
                        break;
                    case 2:
                        entry.offset = std::stoul(cell_value);
                        break;
                    case 3:
                        entry.size = std::stoul(cell_value);
                        break;
                    case 4:
                        entry.read = std::stoi(cell_value);
                        break;
                    case 5:
                        entry.volume_id = std::stoi(cell_value);
                        break;
                    default:
                        // extra columns, ignore
                        break;
                    }
                    cell_value.clear();
                    if (col >= 5) {
                        break;
                    }
                }
                if (col == 0) {
                    continue;
                }
                read_fn(entry);
            }
        } else {
            throw Exception(fmt::format("Cannot parse CSV on file {}", path));
        }
    });
}

void TencentTrace::raw_stream_column(const fs::path& path,
                                     unsigned int column,
                                     RawReadColumnFn&& read_fn) const {
    read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {
        using namespace csv2;

        std::string cell_value{""};

        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.parse_view(block)) {
            for (const auto row : csv) {
                std::size_t col{0};
                for (const auto cell : row) {
                    col += 1;
                    if (col == column) {
                        cell.read_raw_value(cell_value);
                        read_fn(cell_value);
                        cell_value.clear();
                    }
                }
                if (col == 0) {
                    continue;
                }
                if (column > col) {
                    std::string row_value;
                    row.read_raw_value(row_value);
                    throw Exception(fmt::format("Expected to get column {} but number of columns in the file is {}. Line: \"{}\"", column, col, row_value));
                }
            }
        } else {
            throw Exception(fmt::format("Cannot parse CSV on file {}", path));
        }
    });
}
} // namespace trace_utils::trace
