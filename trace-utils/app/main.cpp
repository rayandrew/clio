#include <trace-utils/trace/tencent.hpp>

#include <fmt/core.h>
// #include <trace-utils/tar-gz.hpp>
#include <trace-utils/logger.hpp>


int main() {
    trace_utils::logger::create();
    // trace_utils::Logger::get();
    // trace_utils::MsftTrace trace;
    // trace_utils::trace::MsftTrace::Entry trace_entry;
    // fmt::print("Trace entry = {}\n", trace_entry);
    // extract_tar_gz_to_memory("/mnt/data/mnt/dev0/Data/tencent/parts/part-001.tgz");
    trace_utils::trace::TencentTrace trace;
    trace.read("/mnt/data/mnt/dev0/Data/tencent/parts/part-001.tgz");
    // std::cout << trace_entry.timestamp << std::endl;
    return 0;
}
