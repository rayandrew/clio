#include "alibaba.hpp"

// #include "count-volume-map.hpp"
// #include "count-volume-reduce.hpp"
// #include "pick-volume.hpp"
#include "split.hpp"

namespace trace_utils::app
{
    namespace alibaba
    {
        const char *name = "alibaba";
        const char *description = "alibaba Trace Utils";
    } // namespace alibaba

    AlibabaApp::AlibabaApp() : NamespaceApp(alibaba::name, alibaba::description)
    {
    }

    void AlibabaApp::setup_args(CLI::App *app)
    {
        NamespaceApp::setup_args(app);
        // add<alibaba::CountVolumeMapApp>();
        // add<alibaba::CountVolumeReduceApp>();
        // add<alibaba::PickVolumeApp>();
        add<alibaba::SplitApp>();

        for (auto &cmd : apps)
            cmd->setup_args(parser);
    }
} // namespace trace_utils::app
