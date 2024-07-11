#include <trace-utils/trace/tencent.hpp>

#include <fmt/core.h>

#include <CLI/CLI.hpp>

#include <trace-utils/logger.hpp>

#include "./tencent/tencent.hpp"

#ifdef __unix__
#include <csignal>
#include <indicators/cursor_control.hpp>

namespace
{
    volatile std::sig_atomic_t shutdown;
}

std::sig_atomic_t signaled = 0;

void cleanup([[maybe_unused]] int sig) {
    shutdown = sig;
    indicators::show_console_cursor(true);
    std::exit(1);
    // throw trace_utils::Exception("Program interrupted");
}
#endif


int main(int argc, char* argv[]) {
#ifdef __unix__
    std::signal(SIGINT, cleanup);
#endif
    
    trace_utils::logger::create();

    CLI::App app;
    app.set_help_flag("-h,--help", "Expand help");
    app.set_help_all_flag("--help-all", "Expand all help");
    
    try {
        trace_utils::app::TencentApp tencent_app;
        tencent_app.setup_args(&app);
        app.parse(argc, argv);
        tencent_app(&app);
    } catch (const CLI::CallForHelp& err) {
        trace_utils::log()->info("Help\n\n{}", app.help(""));
        return 1;
    } catch (const CLI::CallForAllHelp& err) {
        trace_utils::log()->info("Help All\n\n{}", app.help("", CLI::AppFormatMode::All));
        return 1;
    } catch (const CLI::Error& err) {
        trace_utils::log()->error("CLI::Error: {}", err.what());
        trace_utils::log()->error("Run --help/--help-all to see options");
        return 1;
    } catch (const trace_utils::Exception& err) {
        trace_utils::log()->error("Trace Utils Error:\n\n  {}", err.what());
        return 1;
    } catch (const std::exception& err) {
        trace_utils::log()->error("Error: {}", err.what());
        return 1;
    }

    return 0;
}
