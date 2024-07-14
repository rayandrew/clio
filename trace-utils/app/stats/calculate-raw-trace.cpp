#include "calculate-raw-trace.hpp"

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/utils.hpp>
#include <trace-utils/exception.hpp>

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
    log()->info("window str {}", window_str);

    utils::parse_duration(window_str, window);
    log()->info("Splitting with window = {}", window);
    
    output = fs::weakly_canonical(output);
    fs::create_directories(output);
}

void CalculateRawTraceApp::run([[maybe_unused]] CLI::App* app) {

}
} // namespace trace_utils::app::stats::calculate
