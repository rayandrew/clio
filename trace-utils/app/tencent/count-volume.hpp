#ifndef __TRACE_UTILS_APP_TENCENT_COUNT_VOLUME_HPP__
#define __TRACE_UTILS_APP_TENCENT_COUNT_VOLUME_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::app::tencent { 
class CountVolume : public App {
public:
    CountVolume();
    virtual void setup(CLI::App* app) override;
    virtual void run(CLI::App* app) override;

private:
    fs::path input;
    fs::path output;
};
} // namespace trace_utils::app::tencent

#endif
