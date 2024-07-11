#include <trace-utils/trace/tencent.hpp>

#include <cstdint>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>
#include <csv2/reader.hpp>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"

namespace trace_utils::trace {
namespace tencent {
trace::Entry Entry::convert() const {
    trace::Entry entry;
    entry.timestamp = static_cast<double>(timestamp * 1000.0); // in ms
    entry.disk_id = volume;
    entry.offset = offset * 512;
    entry.size = size * 512;
    entry.read = read == 0;
    return entry;
}

std::vector<std::string> Entry::to_vec() const {
    return {
        std::to_string(timestamp),
        std::to_string(offset),
        std::to_string(size),
        std::to_string(read),
        std::to_string(volume),
    };
}

template<typename Csv, typename Fn>
void read_csv(Csv&& csv, Fn&& fn) {
    std::string cell_value{""};

    for (const auto row : csv) {
        TencentTrace::Entry entry;
        std::size_t col{0};
        for (const auto cell : row) {
            col += 1;
            cell.read_raw_value(cell_value);
            switch (col) {
            case 1:
                entry.timestamp = std::stod(cell_value);
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
                entry.volume = std::stoi(cell_value);
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
        if (col < 5) {
            continue;
        }
        fn(entry);
    }
}

template<typename Csv, typename Fn>
void read_csv_column(Csv&& csv, unsigned int column, Fn&& fn) {
    std::string cell_value{""};
        
    for (const auto row : csv) {
        std::size_t col{0};
        for (const auto cell : row) {
            col += 1;
            if (col == column) {
                cell.read_raw_value(cell_value);
                fn(cell_value);
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
}
} // namespace tencent


    
void TencentTrace::raw_stream(const fs::path& path, RawReadFn&& read_fn) const {
    using namespace csv2;
    if (internal::is_tar_file(path) || internal::is_gz_file(path)) {
        read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {
            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                tencent::read_csv(csv,
                                  std::forward<RawReadFn>(read_fn));
            }
        });
    } else if (internal::is_delimited_file(path, ',')) {
        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.mmap(path.string())) {
            tencent::read_csv(csv, std::forward<RawReadFn>(read_fn));
        }
    } else {
        throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
    }
}

void TencentTrace::raw_stream_column(const fs::path& path,
                                     unsigned int column,
                                     RawReadColumnFn&& read_fn) const {
    using namespace csv2;
    if (internal::is_tar_file(path) || internal::is_gz_file(path)) {
        read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {

            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                read_csv_column(csv, column, std::forward<RawReadColumnFn>(read_fn));
            } else {
                throw Exception(fmt::format("Cannot parse CSV on file {}", path));
            }
        });
    } else if (internal::is_delimited_file(path, ',')) {
        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.mmap(path.string())) {
            read_csv_column(csv, column, std::forward<RawReadColumnFn>(read_fn));
        }
    } else {
        throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
    } 
}
} // namespace trace_utils::trace
