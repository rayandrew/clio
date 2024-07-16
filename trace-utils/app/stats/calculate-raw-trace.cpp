#include "calculate-raw-trace.hpp"

#include <string_view>
#include <charconv>

#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>

#include <fmt/std.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>

#include <csv2/writer.hpp>
#include <csv2/reader.hpp>

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

    oneapi::tbb::concurrent_vector<std::vector<std::string>> v;
    std::vector<std::string> header;

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

        std::atomic_size_t i = 0;        
        
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, paths.size()),
                                  [&](const auto& r) {

            for (std::size_t chunk = r.begin(); chunk < r.end(); ++chunk) {
                trace::ReplayerTrace trace(paths[chunk]);
                auto characteristic = RawCharacteristic::from(trace, true);
                if (i == 0) {
                    header = characteristic.header();
                    header.insert(header.begin(), "chunk");
                    i++;
                }

                auto values = characteristic.values();
                auto chunk_str = utils::to_string(chunk);
                values.insert(values.begin(), chunk_str);
                v.push_back(values);
                pbar.tick();
            }
        });

        pbar.mark_as_completed();
    });

    log()->info("Generating stats took {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));

    dur = utils::get_time([&] {
        indicators::show_console_cursor(false);
        defer {
            indicators::show_console_cursor(true);
        };
        
        if (v.size() == 0) {
            throw Exception("cannot read to sort characteristic file!");
        }
        
        oneapi::tbb::parallel_sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            auto num_a = std::stoi(a[0]);
            auto num_b = std::stoi(b[0]);
            auto comp = num_a < num_b;
            return comp;
        });
        
        std::fstream stream(output / "characteristic.csv",
                            std::fstream::out);
        csv2::Writer<csv2::delimiter<','>, std::fstream> writer(stream);
        writer.write_row(header);
        writer.write_rows(v);
    });

    log()->info("Sorting generated stats took {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
}
} // namespace trace_utils::app::stats::calculate
