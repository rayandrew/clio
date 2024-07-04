#ifndef __TRACE_UTILS_APP_TENCENT_PICK_DEVICE_HPP__
#define __TRACE_UTILS_APP_TENCENT_PICK_DEVICE_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::app::tencent { 
class PickDevice : public App {
public:
    PickDevice();
    virtual void setup(CLI::App* app) override;
    virtual void run(CLI::App* app) override;

private:
    fs::path input;
    fs::path output;
    unsigned long volume;
    
};
} // namespace trace_utils::app::tencent

#endif
