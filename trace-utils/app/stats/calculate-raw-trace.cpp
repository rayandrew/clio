#include "calculate-raw-trace.hpp"

#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>

#include <fmt/std.h>
#include <fmt/chrono.h>

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/characteristic.hpp>
#include <trace-utils/trace.hpp>
#include <trace-utils/trace/replayer.hpp>
#include <trace-utils/utils.hpp>

namespace trace_utils::app::stats::calculate {
namespace raw_trace {
const char* name = "raw-trace";
const char* description = "Calculate Raw Trace Stats";
} // namespace calculate_raw_trace


CalculateRawTraceApp::CalculateRawTraceApp(): App(raw_trace::name, raw_trace::description) {

}

CalculateRawTraceApp::~CalculateRawTraceApp() {
    indicators::show_console_cursor(true);
}

void CalculateRawTraceApp::setup_args(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required();
    parser->add_option("-w,--window", window_str, "Choose window")->required();
}

void CalculateRawTraceApp::setup() {
    utils::parse_duration(window_str, window);
    log()->info("Splitting with window = {}", window);
    
    output = fs::weakly_canonical(output);
    fs::create_directories(output);
}

void CalculateRawTraceApp::run([[maybe_unused]] CLI::App* app) {
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);


    utils::f_sec dur = utils::get_time([&] {
        indicators::show_console_cursor(false);
        defer {
            indicators::show_console_cursor(true);
        };
        indicators::BlockProgressBar pbar{
            indicators::option::ForegroundColor{indicators::Color::yellow},
            indicators::option::FontStyles{
                std::vector<indicators::FontStyle>{
                    indicators::FontStyle::bold
                }
            },
            indicators::option::MaxProgress{paths.size()},
            indicators::option::PrefixText{"Generating stats... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };

        oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
            trace::ReplayerTrace trace(path);
            auto characteristic = RawCharacteristic::from(trace, true);
            pbar.tick();
        });

        pbar.mark_as_completed();
    });

    log()->info("Generating stats took {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
}
} // namespace trace_utils::app::stats::calculate
