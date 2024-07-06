#include "count-volume.hpp"

#include <algorithm>

#include <fmt/std.h>
#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>

namespace trace_utils::app::tencent {
namespace count_volume {
const char* name = "count-volume";
const char* description = "Tencent Count Volume";
}

typedef oneapi::tbb::concurrent_hash_map<unsigned long, int> UnsignedLongTable;

// struct Tally {
//     UnsignedLongTable& table;
//     Tally(UnsignedLongTable& table_) : table(table_) {}
//     void operator()(const oneapi::tbb::blocked_range<unsigned long*> range) const {
//         for(auto* p=range.begin(); p!=range.end(); ++p) {
//             UnsignedLongTable::accessor a;
//             table.insert(a, *p);
//             a->second += 1;
//         }
//     }
// };

    
CountVolume::CountVolume(): App(count_volume::name, count_volume::description) {

}
    
void CountVolume::setup(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required(); 
}

void CountVolume::run([[maybe_unused]] CLI::App* app) {
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);

    UnsignedLongTable table;
    oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
        log()->info("Path: {}", path);
        trace_utils::trace::TencentTrace trace(path);
        trace.raw_stream([&](const auto& item) {
            UnsignedLongTable::accessor a;
            table.insert(a, item.volume_id);
            a->second += 1;
        });
    });

    for (UnsignedLongTable::iterator i = table.begin(); i != table.end(); ++i) {
        log()->info("volume={}, count={}",i->first, i->second);
    }
}
} // namespace trace_utils::app::tencent
