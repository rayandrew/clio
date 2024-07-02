#include <trace-utils/trace/msft.hpp>
#include <fmt/core.h>

int main() {
    /* trace_utils::MsftTrace trace; */
    trace_utils::trace::MsftTrace::Entry trace_entry;
    fmt::print("Trace entry = {}\n", trace_entry);
    /* std::cout << trace_entry.timestamp << std::endl; */
    return 0;
}
