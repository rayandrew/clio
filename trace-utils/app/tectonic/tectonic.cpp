#include "tectonic.hpp"

// #include "count-volume-map.hpp"
// #include "count-volume-reduce.hpp"
// #include "pick-volume.hpp"
#include "split.hpp"

namespace trace_utils::app
{
    namespace tectonic
    {
        const char *name = "tectonic";
        const char *description = "tectonic Trace Utils";
    } // namespace tectonic

    TectonicApp::TectonicApp() : NamespaceApp(tectonic::name, tectonic::description)
    {
    }

    void TectonicApp::setup_args(CLI::App *app)
    {
        NamespaceApp::setup_args(app);
        add<tectonic::SplitApp>();

        for (auto &cmd : apps)
            cmd->setup_args(parser);
    }
} // namespace trace_utils::app
