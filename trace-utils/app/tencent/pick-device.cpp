#include "pick-device.hpp"

#include <algorithm>

#include <fmt/std.h>
#include <glob/glob.h>
#include <natural_sort.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>

namespace trace_utils::app::tencent {
namespace pick_device {
const char* name = "pick-device";
const char* description = "Tencent Pick Device";
}
    
PickDevice::PickDevice(): App(pick_device::name, pick_device::description) {

}
    
void PickDevice::setup(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required(); 
    parser->add_option("-v,--volume", volume, "Choose volume")->required();
}

void PickDevice::run(CLI::App* app) {
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);
    std::for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
        log()->info("Path: {}", path);
        trace_utils::trace::TencentTrace trace(path);
        trace.stream([&](const auto& item) {
            // trace_utils::log()->info("Item {}", item);
        });
    });
    

}
} // namespace trace_utils::app::tencent
