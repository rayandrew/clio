#include "stats.hpp"

namespace trace_utils::app {
namespace stats {
const char* name = "stats";
const char* description = "Trace Statistic";
} // namespace stats
    
StatsApp::StatsApp(): NamespaceApp(stats::name, stats::description) {
    
}

void StatsApp::setup_args(CLI::App* app) {
    NamespaceApp::setup_args(app);

    for (auto& cmd: apps) cmd->setup_args(parser);
}
} // namespace trace_utils::app
