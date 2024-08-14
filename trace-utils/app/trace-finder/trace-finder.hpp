#ifndef __TRACE_UTILS_APP_TRACE_FINDER_HPP__
#define __TRACE_UTILS_APP_TRACE_FINDER_HPP__

#include "../app.hpp"

namespace trace_utils::app {
class TraceFinderApp : public NamespaceApp {
public:    
    TraceFinderApp();

    virtual void setup_args(CLI::App* app) override;
};
} // namespace trace_utils::app

#endif
