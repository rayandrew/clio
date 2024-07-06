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
    read_tar_gz_csv(path, [&](auto buffer, auto line, auto line_count, auto* entry) {
        using namespace csv2;
        try {
            std::string cell_value{""};

            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(buffer)) {
                for (const auto row : csv) {
                    TencentTrace::Entry entry;
                    size_t col{0};
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
                    read_fn(entry);
                }
            }
            
            // io::CSVReader<5> csv{archive_entry_pathname(entry), line.cbegin(), line.cend()};
            // while (csv.read_row(entry.timestamp, entry.offset, entry.size, entry.read, entry.volume_id)) {
            //        read_fn(entry);
            // }
        } catch (const std::exception& ex) {
            auto what = std::string{ex.what()};
            if (what.find("The integer") != std::string::npos) {
                auto l = std::string{line};
                std::string delimiter = "\n";
                auto x = l.find(delimiter);
                std::string token;
                if (x != std::string::npos) {
                    token = l.substr(0, x);
                }

                
                log()->error("Skipping line due to cannot parse at line {} in file {} with archive path {}", line_count, path, archive_entry_pathname(entry));
                if (!token.empty()) {
                    log()->error("       Pathname {}", path);
                    log()->error("       Buffer {}", buffer);
                    log()->error("       Line {}", token);
                    log()->error("       Full Line\n{}", line);
                    exit(1);
                }
            }
            // log()->error("Skipping line due to cannot parse at line {} in file {} with archive path {}", line_count, path, archive_entry_pathname(entry));
            // log()->error("   Message: {}", ex.what());
        }
    });
}

void TencentTrace::raw_stream_column(const fs::path& path,
                                     unsigned int column,
                                     RawReadColumnFn&& read_fn) const {
    read_tar_gz_csv(path, [&](auto buffer, auto line, auto line_count, auto* entry) {
        using namespace csv2;

        std::string cell_value{""};

        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
        if (csv.parse_view(buffer)) {
            for (const auto row : csv) {
                TencentTrace::Entry entry;
                size_t col{0};
                for (const auto cell : row) {
                    col += 1;
                    if (col == column) {
                        cell.read_raw_value(cell_value);
                        read_fn(cell_value);
                        cell_value.clear();
                    }
                }
                if (column > col) {
                    throw new Exception(fmt::format("Expected to get column {} but number of columns in the file is {}", column, col));
                }
            }
        }
    });
}
} // namespace trace_utils::trace
