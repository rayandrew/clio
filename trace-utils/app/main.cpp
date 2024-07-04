#include <trace-utils/trace/tencent.hpp>

#include <fmt/core.h>
#include <trace-utils/logger.hpp>


int main() {
    trace_utils::logger::create();
    trace_utils::trace::TencentTrace trace("/mnt/data/mnt/dev0/Data/tencent/parts/part-001.tgz");
    trace.stream([&](auto item) {
    });
    return 0;
}
