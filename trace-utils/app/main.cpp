#include <trace-utils/trace/tencent.hpp>

#include <fmt/core.h>

#include <CLI/CLI.hpp>

#include <trace-utils/logger.hpp>

#include "./tencent/tencent.hpp"


int main(int argc, char* argv[]) {
    trace_utils::logger::create();

    try {
        CLI::App app;
        app.set_help_all_flag("--help-all", "Expand all help");
        trace_utils::app::TencentApp tencent_app;
        tencent_app.setup(&app);
        app.parse(argc, argv);
        tencent_app(&app);
    } catch (const CLI::Error& err) {
        trace_utils::log()->error("CLI::Error: {}", err.what());
        trace_utils::log()->error("Run --help/--help-all to see options");
        return 1;
    } catch (const std::exception& err) {
        trace_utils::log()->error("Error: {}", err.what());
        return 1;
    }

    return 0;
}
