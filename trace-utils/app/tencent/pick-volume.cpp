#include "pick-volume.hpp"

#include <algorithm>

#include <fmt/std.h>
#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>

namespace trace_utils::app::tencent {
namespace pick_volume {
const char* name = "pick-volume";
const char* description = "Tencent Pick Volume";
}
    
PickVolume::PickVolume(): App(pick_volume::name, pick_volume::description) {

}
    
void PickVolume::setup(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required(); 
    parser->add_option("-v,--volume", volume, "Choose volume")->required();
}

void PickVolume::run([[maybe_unused]] CLI::App* app) {
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);
    oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
        log()->info("Path: {}", path);
        trace_utils::trace::TencentTrace trace(path);
        trace.stream([&](const auto& item) {

        });
    });
    

}
} // namespace trace_utils::app::tencent
