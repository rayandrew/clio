#include <trace-utils/trace/replayed.hpp>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <scope_guard.hpp>
#include <csv2/reader.hpp>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"

namespace trace_utils::trace
{
    namespace replayed
    {
        std::vector<std::string> Entry::to_vec() const
        {
            return {
                std::to_string(timestamp),
                std::to_string(latency),
                read ? "1" : "0",
                std::to_string(size),
                std::to_string(offset),
                std::to_string(timestamp_submit),
                std::to_string(size_after_replay),
            };
        }

        template <typename Csv, typename Fn>
        void read_csv(Csv &&csv, Fn &&fn)
        {
            std::string cell_value{""};
            std::vector<ReplayedTrace::Entry> entries;

            for (const auto row : csv)
            {
                ReplayedTrace::Entry entry;
                std::size_t col{0};
                for (const auto cell : row)
                {
                    col += 1;
                    cell.read_raw_value(cell_value);
                    auto column = magic_enum::enum_cast<ReplayedTrace::Column>(col);
                    if (!column.has_value())
                    {
                        break;
                    }
                    switch (col)
                    {
                    case magic_enum::enum_underlying(ReplayedTrace::Column::TIMESTAMP):
                        entry.timestamp = std::stod(cell_value);
                        break;
                    case magic_enum::enum_underlying(ReplayedTrace::Column::LATENCY):
                        entry.latency = std::stod(cell_value);
                        break;
                    case magic_enum::enum_underlying(ReplayedTrace::Column::READ):
                        entry.read = std::stoi(cell_value) == 1;
                        break;
                    case magic_enum::enum_underlying(ReplayedTrace::Column::SIZE):
                        entry.size = std::stoi(cell_value);
                        break;
                    case magic_enum::enum_underlying(ReplayedTrace::Column::OFFSET):
                        entry.offset = std::stoul(cell_value);
                        break;
                    case magic_enum::enum_underlying(ReplayedTrace::Column::TIMESTAMP_SUBMIT):
                        entry.timestamp_submit = std::stod(cell_value);
                        break;
                    case magic_enum::enum_underlying(ReplayedTrace::Column::SIZE_AFTER_REPLAY):
                        entry.size_after_replay = std::stod(cell_value);
                        break;
                    default:
                        // extra columns, ignore
                        break;
                    }
                    cell_value.clear();
                    if (col >= magic_enum::enum_count<ReplayedTrace::Column>())
                    {
                        break;
                    }
                }
                if (col < magic_enum::enum_count<ReplayedTrace::Column>())
                {
                    continue;
                }
                entries.push_back(entry);
            }

            // Replayed is not guaranteed to be sorted
            oneapi::tbb::parallel_sort(entries.begin(), entries.end(),
                                       [=](const auto &a, const auto &b)
                                       {
                                           return a.timestamp < b.timestamp;
                                       });

            for (const auto &entry : entries)
            {
                fn(entry);
            }
        }
    }

    void ReplayedTrace::raw_stream(const fs::path &path, RawReadFn &&read_fn) const
    {
        using namespace csv2;
        if (internal::is_tar_file(path) || internal::is_gz_file(path))
        {
            read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto *entry)
                            {
            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                replayed::read_csv(csv,
                                  std::forward<RawReadFn>(read_fn));
            } });
        }
        else if (internal::is_delimited_file(path, ','))
        {
            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.mmap(path.string()))
            {
                replayed::read_csv(csv, std::forward<RawReadFn>(read_fn));
            }
        }
        else
        {
            throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
        }
    }

    void ReplayedTrace::raw_stream_column(const fs::path &path,
                                          unsigned int column,
                                          RawReadColumnFn &&read_fn) const
    {
        using namespace csv2;
        if (internal::is_tar_file(path) || internal::is_gz_file(path))
        {
            read_tar_gz_csv(path, [&](auto block, [[maybe_unused]] auto block_count, [[maybe_unused]] auto *entry)
                            {

            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.parse_view(block)) {
                read_csv_column<ReplayedTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
            } else {
                throw Exception(fmt::format("Cannot parse CSV on file {}", path));
            } });
        }
        else if (internal::is_delimited_file(path, ','))
        {
            Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<false>> csv;
            if (csv.mmap(path.string()))
            {
                read_csv_column<ReplayedTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
            }
        }
        else
        {
            throw Exception(fmt::format("File {} is not supported, expected csv or tar.gz", path));
        }
    }
} // namespace trace_utils::trace
