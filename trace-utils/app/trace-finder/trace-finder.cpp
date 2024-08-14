#include "trace-finder.hpp"

#include "trace-finder-v1.hpp"

namespace trace_utils::app {
namespace trace_finder {
const char* name = "trace-finder";
const char* description = "Trace Finder";
} // namespace trace_finder
    
TraceFinderApp::TraceFinderApp(): NamespaceApp(trace_finder::name, trace_finder::description) {
    
}

void TraceFinderApp::setup_args(CLI::App* app) {
    NamespaceApp::setup_args(app);
    add<trace_finder::TraceFinderV1App>();

    for (auto& cmd: apps) cmd->setup_args(parser);
}
} // namespace trace_utils::app
