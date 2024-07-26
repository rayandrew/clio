#include "calculate.hpp"

#include <trace-utils/logger.hpp>
#include <trace-utils/utils.hpp>
#include <trace-utils/exception.hpp>

#include "calculate-raw-trace.hpp"
#include "calculate-replayed-trace.hpp"
#include "plot-char-app.hpp"

namespace trace_utils::app::stats
{
    namespace calculate
    {
        const char *name = "calculate";
        const char *description = "Calculate Trace Stats";
    } // namespace stats

    CalculateApp::CalculateApp() : NamespaceApp(calculate::name, calculate::description)
    {
    }

    void CalculateApp::setup_args(CLI::App *app)
    {
        NamespaceApp::setup_args(app);
        add<stats::calculate::CalculateRawTraceApp>();
        add<stats::calculate::CalculateReplayedTraceApp>();
        add<stats::calculate::PlotCharApp>();

        for (auto &cmd : apps)
            cmd->setup_args(parser);
    }
} // namespace trace_utils::app::stats
