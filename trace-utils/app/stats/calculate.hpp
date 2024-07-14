#ifndef __TRACE_UTILS_APP_STATS_CALCULATE_HPP__
#define __TRACE_UTILS_APP_STATS_CALCULATE_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>
#include <mp-units/format.h>
#include <mp-units/systems/si/si.h>

namespace trace_utils::app::stats {
class CalculateApp : public NamespaceApp {
public:    
    CalculateApp();

    virtual void setup_args(CLI::App* app) override;
};
} // namespace trace_utils::app::stats

#endif
