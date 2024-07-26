#ifndef __TRACE_UTILS_APP_STATS_PLOT_TRACE_HPP__
#define __TRACE_UTILS_APP_STATS_PLOT_TRACE_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>
#include <mp-units/format.h>
#include <mp-units/systems/si/si.h>

namespace trace_utils::app::stats::calculate
{
    class PlotCharApp : public App
    {
    public:
        PlotCharApp();
        ~PlotCharApp();
        virtual void setup_args(CLI::App *app) override;
        virtual void setup() override;
        virtual void run(CLI::App *app) override;

    private:
        fs::path input;
        fs::path output;
    };
} // namespace trace_utils::app::stats::calculate

#endif
