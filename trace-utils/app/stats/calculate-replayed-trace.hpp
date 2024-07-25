#ifndef __TRACE_UTILS_APP_STATS_CALCULATE_CALCULATE_REPLAYED_TRACE_HPP__
#define __TRACE_UTILS_APP_STATS_CALCULATE_CALCULATE_REPLAYED_TRACE_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>
#include <mp-units/format.h>
#include <mp-units/systems/si/si.h>

namespace trace_utils::app::stats::calculate {
class CalculateReplayedTraceApp : public App {
public:
    CalculateReplayedTraceApp();
    ~CalculateReplayedTraceApp();
    virtual void setup_args(CLI::App* app) override;
    virtual void setup() override;
    virtual void run(CLI::App* app) override;

private:
    fs::path input;
    fs::path output;
    mp_units::quantity<mp_units::si::second> window;
    std::string window_str;
};
} // namespace trace_utils::app::stats::calculate

#endif
