#include <trace-utils/trace/replayer.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>
#include <csv2/reader.hpp>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"

namespace trace_utils::trace {
namespace replayer {
template<typename Csv, typename Fn>
void read_csv(Csv&& csv, Fn&& fn) {
    std::string cell_value{""};

    for (const auto row : csv) {
        ReplayerTrace::Entry entry;
        std::size_t col{0};
        for (const auto cell : row) {
            col += 1;
            cell.read_raw_value(cell_value);
            auto column = magic_enum::enum_cast<ReplayerTrace::Column>(col);
            if (!column.has_value()) {
                break;
            }
            switch (col) {
            case magic_enum::enum_underlying(ReplayerTrace::Column::TIMESTAMP):
                entry.timestamp = std::stod(cell_value);
                break;
            case magic_enum::enum_underlying(ReplayerTrace::Column::DISK_ID):
                entry.disk_id = std::stoul(cell_value);
                break;
            case magic_enum::enum_underlying(ReplayerTrace::Column::OFFSET):
                entry.offset = std::stoul(cell_value);
                break;
            case magic_enum::enum_underlying(ReplayerTrace::Column::SIZE):
                entry.size = std::stoi(cell_value);
                break;
            case magic_enum::enum_underlying(ReplayerTrace::Column::READ):
                entry.read = std::stoi(cell_value) == 1;
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
}
    
void ReplayerTrace::raw_stream(const fs::path& path, RawReadFn&& read_fn) const {
    using namespace csv2;
    if (internal::is_tar_file(path) || internal::is_gz_file(path)) {
        read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {
            Reader<delimiter<' '>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                replayer::read_csv(csv,
                                  std::forward<RawReadFn>(read_fn));
            }
        });
    } else if (internal::is_delimited_file(path, ' ')) {
        Reader<delimiter<' '>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.mmap(path.string())) {
            replayer::read_csv(csv, std::forward<RawReadFn>(read_fn));
        }
    } else {
        fs::path error_log_dir = "error_log";
        fs::create_directory(error_log_dir);
        fs::path dest = error_log_dir / path.filename();
        fs::copy_file(path, dest, fs::copy_options::overwrite_existing);
        
        throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
    }

}

void ReplayerTrace::raw_stream_column(const fs::path& path,
                                     unsigned int column,
                                     RawReadColumnFn&& read_fn) const {
    using namespace csv2;
    if (internal::is_tar_file(path) || internal::is_gz_file(path)) {
        read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto* entry) {

            Reader<delimiter<' '>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                read_csv_column<ReplayerTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
            } else {
                throw Exception(fmt::format("Cannot parse CSV on file {}", path));
            }
        });
    } else if (internal::is_delimited_file(path, ' ')) {
        Reader<delimiter<' '>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.mmap(path.string())) {
            read_csv_column<ReplayerTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
        }
    } else {
        fs::path error_log_dir = "error_log";
        fs::create_directory(error_log_dir); 
        fs::path dest = error_log_dir / path.filename();
        fs::copy_file(path, dest, fs::copy_options::overwrite_existing);

        throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
    } 
}
} // namespace trace_utils::trace
