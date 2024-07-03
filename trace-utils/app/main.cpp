#include <trace-utils/trace/tencent.hpp>

#include <fmt/core.h>
#include <trace-utils/logger.hpp>


int main() {
    trace_utils::logger::create();
    trace_utils::trace::TencentTrace trace;
    trace.read("/mnt/data/mnt/dev0/Data/tencent/parts/part-001.tgz");
    return 0;
}
