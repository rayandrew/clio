#include <trace-utils/trace/alibaba.hpp>

#include <cstdint>
#include <iostream>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>
#include <csv2/reader.hpp>
#include <magic_enum.hpp>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"

namespace trace_utils::trace
{
    namespace alibaba
    {
        trace::Entry Entry::convert() const
        {
            trace::Entry entry;
            entry.timestamp = static_cast<double>(timestamp / 1000); // in ms
            entry.disk_id = device_id;
            entry.offset = offset;
            entry.size = length;
            entry.read = opcode == 'R';

            return entry;
        }

        std::vector<std::string> Entry::to_vec() const
        {
            return {
                std::to_string(timestamp),
                std::to_string(offset),
                std::to_string(length),
                std::to_string(opcode),
                std::to_string(device_id),
            };
        }

        template <typename Csv, typename Fn>
        void read_csv(Csv &&csv, Fn &&fn)
        {
            std::string cell_value{""};
            bool first_row = true;

            for (const auto row : csv)
            {
                if (first_row)
                {
                    first_row = false;
                    continue;
                }

                AlibabaTrace::Entry entry;
                std::size_t col{0};

                for (const auto cell : row)
                {
                    cell.read_raw_value(cell_value);
                    col += 1;
                    auto column = magic_enum::enum_cast<AlibabaTrace::Column>(col);
                    if (!column.has_value())
                    {
                        break;
                    }
                    switch (col)
                    {
                    case magic_enum::enum_underlying(AlibabaTrace::Column::TIMESTAMP):
                        entry.timestamp = std::stod(cell_value);
                        break;
                    case magic_enum::enum_underlying(AlibabaTrace::Column::OFFSET):
                        entry.offset = std::stoul(cell_value);
                        break;
                    case magic_enum::enum_underlying(AlibabaTrace::Column::LENGTH):
                        entry.length = std::stoul(cell_value);
                        break;
                    case magic_enum::enum_underlying(AlibabaTrace::Column::OPCODE):
                        entry.opcode = cell_value[0];
                        break;
                    case magic_enum::enum_underlying(AlibabaTrace::Column::DEVICE_ID):
                        entry.device_id = std::stoi(cell_value);
                        break;
                    default:
                        // extra columns, ignore
                        break;
                    }
                    cell_value.clear();
                    if (col >= magic_enum::enum_count<AlibabaTrace::Column>())
                    {
                        break;
                    }
                }
                if (col < magic_enum::enum_count<AlibabaTrace::Column>())
                {
                    continue;
                }
                fn(entry);
            }
        }
    } // namespace alibaba

    void AlibabaTrace::raw_stream(const fs::path &path, RawReadFn &&read_fn) const
    {
        using namespace csv2;
        if (internal::is_delimited_file(path, ','))
        {
            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.mmap(path.string()))
            {
                alibaba::read_csv(csv, std::forward<RawReadFn>(read_fn));
            }
        }
        else
        {
            throw Exception(fmt::format("File {} is not supported, expected csv", path));
        }
    }

    void AlibabaTrace::raw_stream_column(const fs::path &path,
                                         unsigned int column,
                                         RawReadColumnFn &&read_fn) const
    {
        if (!magic_enum::enum_contains<Column>(column))
        {
            throw Exception(fmt::format("Column {} is not defined inside Alibaba trace", column));
        }

        using namespace csv2;
        if (internal::is_tar_file(path) || internal::is_gz_file(path))
        {
            read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto *entry)
                            {

            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                read_csv_column<AlibabaTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
            } else {
                throw Exception(fmt::format("Cannot parse CSV on file {}", path));
            } });
        }
        else if (internal::is_delimited_file(path, ','))
        {
            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.mmap(path.string()))
            {
                read_csv_column<AlibabaTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
            }
        }
        else
        {
            throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
        }
    }
} // namespace trace_utils::trace
