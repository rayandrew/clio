#include "tencent.hpp"

#include "count-volume.hpp"
#include "pick-volume.hpp"

namespace trace_utils::app {
namespace tencent {
const char* name = "tencent";
const char* description = "Tencent Trace Utils";
} // namespace tencent
    
TencentApp::TencentApp(): NamespaceApp(tencent::name, tencent::description) {
    
}

void TencentApp::setup(CLI::App* app) {
    NamespaceApp::setup(app);
    add<tencent::CountVolume>();
    add<tencent::PickVolume>();

    for (auto& cmd: apps) cmd->setup(parser);
}
} // namespace trace_utils::app
