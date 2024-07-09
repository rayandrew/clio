#ifndef __TRACE_UTILS_APP_TENCENT_COUNT_VOLUME_REDUCE_HPP__
#define __TRACE_UTILS_APP_TENCENT_COUNT_VOLUME_REDUCE_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::app::tencent { 
class CountVolumeReduceApp : public App {
public:
    CountVolumeReduceApp();
    virtual void setup_args(CLI::App* app) override;
    virtual void run(CLI::App* app) override;

private:
    fs::path input;
    fs::path output;
};
} // namespace trace_utils::app::tencent

#endif
