#ifndef __TRACE_UTILS_APP_TRACE_FINDER_V1_HPP__
#define __TRACE_UTILS_APP_TRACE_FINDER_V1_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::app::trace_finder {
class TraceFinderV1App : public App {
public:
    TraceFinderV1App();
    ~TraceFinderV1App();
    virtual void setup_args(CLI::App* app) override;
    virtual void setup() override;
    virtual void run(CLI::App* app) override;

private:
    fs::path input;
    fs::path output;
    std::string metric;
    float stability_threshold;
    float drift_threshold;
    float group_threshold;
    float group_offset;
    int rolling_window;
};
} // namespace trace_utils::app::tencent

#endif
