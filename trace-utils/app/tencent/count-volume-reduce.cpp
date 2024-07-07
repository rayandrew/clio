#include "count-volume-reduce.hpp"

#include <algorithm>
#include <map>
#include <vector>
#include <utility>
#include <fstream>

#include <fmt/std.h>
#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>
#include <csv2/reader.hpp>
#include <csv2/writer.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>

namespace trace_utils::app::tencent {
namespace count_volume_reduce {
const char* name = "count-volume-reduce";
const char* description = "Tencent Count Volume Parallel Reduce";
}

typedef oneapi::tbb::concurrent_hash_map<std::string, unsigned long> ConcurrentTable;
    
CountVolumeReduceApp::CountVolumeReduceApp(): App(count_volume_reduce::name, count_volume_reduce::description) {

}
    
void CountVolumeReduceApp::setup(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required(); 
}

void CountVolumeReduceApp::run([[maybe_unused]] CLI::App* app) {
    using namespace csv2;
    
    auto output_path = fs::weakly_canonical(output);
    auto parent_output_path = output_path.parent_path();
    fs::create_directories(parent_output_path);
    
    auto input_path = fs::canonical(input) / "*.csv";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);

    ConcurrentTable map;
    oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
        log()->info("Path: {}", path);
        Reader<delimiter<','>, quote_character<'"'>, first_row_is_header<true>> csv;
        std::string cell_value{""};
        std::string device{""};
        unsigned long count = 0;
        if (csv.mmap(path.string())) {
            for (const auto row : csv) {
                std::size_t col{0};
                for (const auto cell : row) {
                    col += 1;
                    cell.read_raw_value(cell_value);
                    switch (col) {
                    case 1:
                        device = cell_value;
                        cell_value.clear();
                        break;
                    case 2:
                        count = std::stoul(cell_value);
                        cell_value.clear();
                        break;
                    default:
                        // extra columns, ignore
                        break;
                    }
                }
                if (col < 2) {
                    continue;
                }
                ConcurrentTable::accessor a;
                map.insert(a, device);
                a->second += count;
            }
        } else {
            throw Exception(fmt::format("Cannot open csv at path {}", path));
        }
    });

    std::vector<std::pair<std::string, unsigned long>> v(map.cbegin(), map.cend());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    std::ofstream stream(output_path);
    Writer<delimiter<','>> writer(stream);
    std::array<std::string, 2> buf{"device", "count"};
    writer.write_row(buf);
    for (const auto& [key, val] : v) {
        buf[0] = key;
        buf[1] = std::to_string(val);
        writer.write_row(buf);
    }
}
} // namespace trace_utils::app::tencent
