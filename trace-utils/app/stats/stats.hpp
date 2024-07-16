#ifndef __TRACE_UTILS_APP_STATS_HPP__
#define __TRACE_UTILS_APP_STATS_HPP__

#include "../app.hpp"

namespace trace_utils::app {
class StatsApp : public NamespaceApp {
public:    
    StatsApp();

    virtual void setup_args(CLI::App* app) override;
};
} // namespace trace_utils::app

#endif
