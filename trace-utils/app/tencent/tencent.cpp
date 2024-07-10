#include "tencent.hpp"

#include "count-volume-map.hpp"
#include "count-volume-reduce.hpp"
#include "pick-volume.hpp"
#include "split.hpp"

namespace trace_utils::app {
namespace tencent {
const char* name = "tencent";
const char* description = "Tencent Trace Utils";
} // namespace tencent
    
TencentApp::TencentApp(): NamespaceApp(tencent::name, tencent::description) {
    
}

void TencentApp::setup_args(CLI::App* app) {
    NamespaceApp::setup_args(app);
    add<tencent::CountVolumeMapApp>();
    add<tencent::CountVolumeReduceApp>();
    add<tencent::PickVolumeApp>();
    add<tencent::SplitApp>(); // ongoing ...
    // add<tencent::ConvertApp>();
    // add<tencent::CalculateCharacteristicApp>(); // generate data for CDF, etc ...

    for (auto& cmd: apps) cmd->setup_args(parser);
}
} // namespace trace_utils::app
