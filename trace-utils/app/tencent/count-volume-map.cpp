#include "count-volume-map.hpp"

#include <algorithm>
#include <map>
#include <array>
#include <fstream>

#include <fmt/std.h>
#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>
#include <csv2/writer.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>

namespace trace_utils::app::tencent {
namespace count_volume_map {
const char* name = "count-volume-map";
const char* description = "Tencent Count Volume Parallel Map";
}
    
CountVolumeMapApp::CountVolumeMapApp(): App(count_volume_map::name, count_volume_map::description) {

}
    
void CountVolumeMapApp::setup(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required(); 
}

void CountVolumeMapApp::run([[maybe_unused]] CLI::App* app) {
    auto output_path = fs::weakly_canonical(output);
    fs::create_directories(output_path);
    
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);

    oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
        using namespace csv2;
        log()->info("Path: {}", path);
        try {
            auto map = std::map<std::string, unsigned long>();
            trace_utils::trace::TencentTrace trace(path);
            trace.raw_stream_column(5, [&](const auto& item) {
                auto result = map.insert({ item, 1 });
                if (!result.second) {
                    ++(result.first->second);
                }
            });
            
            std::ofstream stream(output_path / fmt::format("{}.csv", path.stem()));
            Writer<delimiter<','>> writer(stream);
            std::array<std::string, 2> buf{"device", "count"};
            writer.write_row(buf);
            for (const auto& [key, val] : map) {
                buf[0] = key;
                buf[1] = std::to_string(val);
                writer.write_row(buf);
            }
        } catch (const std::exception& ex) {
            log()->info("Error: {}", ex.what());
        }
    });
}
} // namespace trace_utils::app::tencent