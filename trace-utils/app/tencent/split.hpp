#ifndef __TRACE_UTILS_APP_TENCENT_SPLIT_HPP__
#define __TRACE_UTILS_APP_TENCENT_SPLIT_HPP__

#include "../app.hpp"
#include <trace-utils/internal/filesystem.hpp>
#include <mp-units/format.h>
#include <mp-units/systems/si/si.h>
#include <mp-units/systems/isq/isq.h>

namespace trace_utils::app::tencent {
// using Duration = mp_units::QuantityOf<mp_units::isq::time>;
    
class SplitApp : public App {
public:
    SplitApp();
    virtual void setup_args(CLI::App* app) override;
    virtual void setup() override;
    virtual void run(CLI::App* app) override;

private:
    fs::path input;
    fs::path output;
    mp_units::quantity<mp_units::si::second> window;
    std::string window_str;
    fs::path tmp_dir_path;
};
} // namespace trace_utils::app::tencent

#endif
