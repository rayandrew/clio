#ifndef __TRACE_UTILS_APP_TENCENT_HPP__
#define __TRACE_UTILS_APP_TENCENT_HPP__

#include "../app.hpp"

namespace trace_utils::app {
class TencentApp : public NamespaceApp {
public:    
    TencentApp();

    virtual void setup(CLI::App* app) override;
};
} // namespace trace_utils::app

#endif
