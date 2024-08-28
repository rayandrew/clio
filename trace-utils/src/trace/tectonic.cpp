#include <trace-utils/trace/tectonic.hpp>

#include <cstdint>
#include <iostream>

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/std.h>
#include <csv2/reader.hpp>
#include <magic_enum.hpp>
#include <scope_guard.hpp>

#include <trace-utils/logger.hpp>

#include "../utils.hpp"

namespace trace_utils::trace {
namespace tectonic {
trace::Entry Entry::convert() const {
  trace::Entry entry;
  entry.timestamp = static_cast<double>(timestamp * 1000);  // in ms
  entry.disk_id = device_id;
  entry.offset = offset;
  entry.size = length;
  entry.read = (opcode == '2') || (opcode == '1') || (opcode == '5');

  return entry;
}

std::vector<std::string> Entry::to_vec() const {
  return {
      std::to_string(timestamp), std::to_string(offset),    std::to_string(length),
      std::to_string(opcode),    std::to_string(device_id),
  };
}

template <typename Csv, typename Fn>
void read_csv(Csv&& csv, Fn&& fn) {
  std::string cell_value{""};

  for (const auto row : csv) {
    TectonicTrace::Entry entry;
    std::size_t          col{0};

    for (const auto cell : row) {
      cell.read_raw_value(cell_value);
      col += 1;
      auto column = magic_enum::enum_cast<TectonicTrace::Column>(col);
      if (!column.has_value()) {
        break;
      }
      switch (col) {
        case magic_enum::enum_underlying(TectonicTrace::Column::DEVICE_ID):
          entry.device_id = std::stoi(cell_value);
          break;
        case magic_enum::enum_underlying(TectonicTrace::Column::OPCODE):
          entry.opcode = cell_value[0];
          break;
        case magic_enum::enum_underlying(TectonicTrace::Column::OFFSET):
          entry.offset = std::stoul(cell_value);
          break;
        case magic_enum::enum_underlying(TectonicTrace::Column::LENGTH):
          entry.length = std::stoul(cell_value);
          break;
        case magic_enum::enum_underlying(TectonicTrace::Column::TIMESTAMP):
          entry.timestamp = std::stod(cell_value);
          break;

        default:
          // extra columns, ignore
          break;
      }
      cell_value.clear();
      if (col >= magic_enum::enum_count<TectonicTrace::Column>()) {
        break;
      }
    }
    if (col < magic_enum::enum_count<TectonicTrace::Column>()) {
      continue;
    }
    fn(entry);
  }
}
}  // namespace tectonic

void TectonicTrace::raw_stream(const fs::path& path, RawReadFn&& read_fn) const {
  using namespace csv2;
  if (internal::is_delimited_file(path, ' ')) {
    Reader<delimiter<' '>, quote_character<'"'>, first_row_is_header<true>> csv;
    if (csv.mmap(path.string())) {
      tectonic::read_csv(csv, std::forward<RawReadFn>(read_fn));
    }
  } else {
    throw Exception(fmt::format("File {} is not supported, expected csv", path));
  }
}

void TectonicTrace::raw_stream_column(const fs::path& path, unsigned int column, RawReadColumnFn&& read_fn) const {
  if (!magic_enum::enum_contains<Column>(column)) {
    throw Exception(fmt::format("Column {} is not defined inside Tectonic trace", column));
  }

  using namespace csv2;
  if (internal::is_delimited_file(path, ' ')) {
    Reader<delimiter<' '>, quote_character<'"'>, first_row_is_header<false>> csv;
    if (csv.mmap(path.string())) {
      read_csv_column<TectonicTrace>(csv, column, std::forward<RawReadColumnFn>(read_fn));
      // std::cout << "reading col 3" << std::endl;
    }
  } else {
    throw Exception(fmt::format("File {} is not supported, expected csv", path));
  }
}
}  // namespace trace_utils::trace
