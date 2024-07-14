#include <trace-utils/trace/tencent.hpp>

#include <cstdint>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>
#include <csv2/reader.hpp>
#include <magic_enum.hpp>

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
            cell.read_raw_value(cell_value);
            col += 1;
            auto column = magic_enum::enum_cast<TencentTrace::Column>(col);
            if (!column.has_value()) {
                break;
            }
            switch (col) {
            case magic_enum::enum_underlying(TencentTrace::Column::TIMESTAMP):
                entry.timestamp = std::stod(cell_value);
                break;
            case magic_enum::enum_underlying(TencentTrace::Column::OFFSET):
                entry.offset = std::stoul(cell_value);
                break;
            case magic_enum::enum_underlying(TencentTrace::Column::SIZE):
                entry.size = std::stoul(cell_value);
                break;
            case magic_enum::enum_underlying(TencentTrace::Column::READ):
                entry.read = std::stoi(cell_value);
                break;
            case magic_enum::enum_underlying(TencentTrace::Column::VOLUME):
                entry.volume = std::stoi(cell_value);
                break;
            default:
                // extra columns, ignore
                break;
            }
            cell_value.clear();
            if (col >= magic_enum::enum_count<TencentTrace::Column>()) {
                break;
            }
        }
        if (col < magic_enum::enum_count<TencentTrace::Column>()) {
            continue;
        }
        fn(entry);
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
    if (!magic_enum::enum_contains<Column>(column)) {
        throw Exception(fmt::format("Column {} is not defined inside Tencent trace", column));
    }
    
    using namespace csv2;
    if (internal::is_tar_file(path) || internal::is_gz_file(path)) {
        read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {

            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                read_csv_column<TencentTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
            } else {
                throw Exception(fmt::format("Cannot parse CSV on file {}", path));
            }
        });
    } else if (internal::is_delimited_file(path, ',')) {
        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.mmap(path.string())) {
            read_csv_column<TencentTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
        }
    } else {
        throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
    } 
}
} // namespace trace_utils::trace
