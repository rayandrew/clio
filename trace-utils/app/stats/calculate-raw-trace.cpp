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

        std::fstream stream(output / "temp_characteristic.csv",
                            std::fstream::out);
        csv2::Writer<csv2::delimiter<','>, std::fstream> writer(stream);

        std::atomic_size_t i = 0;
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_t>(0, paths.size()),
                                  [&](const auto& r) {

            for (std::size_t chunk = r.begin(); chunk < r.end(); chunk++) {
                trace::ReplayerTrace trace(paths[chunk]);
                auto characteristic = RawCharacteristic::from(trace, true);
                if (i == 0) {
                    auto header = characteristic.header();
                    header.insert(header.begin(), "chunk");
                    writer.write_row(header);
                    i++;
                }

                // TODO: BUG HERE, chunk is not written somehow!!!!
                auto values = characteristic.values();
                values.insert(values.begin(), utils::to_string(chunk));
                writer.write_row(values);
                pbar.tick();
            }
        });

        pbar.mark_as_completed();
    });

    log()->info("Generating stats took {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));

    dur = utils::get_time([&] {
        std::fstream istream(output / "temp_characteristic.csv", std::fstream::in);
        std::string line;
        std::vector<std::string> lines;

        std::size_t i = 0;
        while (std::getline(istream, line)) {
            if (line.empty()) {
                continue;
            }
            if (line.find_first_not_of(' ') == std::string::npos) {
                continue;
            }
            if (i == 0) {
                i++;
                continue;
            }
            lines.push_back(line);
            i++;
        }

        std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) {
            std::string_view view_a(a);
            std::string_view view_b(b);
        
            size_t pos_a = view_a.find(',');
            size_t pos_b = view_b.find(',');

            auto substr_a = view_a.substr(0, pos_a);
            auto substr_b = view_b.substr(0, pos_b);

            log()->info("AA {} |||| BB {}", a, b);
            log()->info("A {} | B {}", substr_a, substr_b);

            int num_a = std::stoi(std::string(view_a.substr(0, pos_a)));
            int num_b = std::stoi(std::string(view_b.substr(0, pos_b)));
            log()->info("A {} {} | B {} {}", substr_a, num_a, substr_b, num_b);
            // int num_a;
            // int num_b;
            // auto result = std::from_chars(substr_a.data(), substr_a.data() + substr_a.size(), num_a);
            // if (result.ec == std::errc::invalid_argument) {
            //     log()->error("Could not convert num a {}", substr_a);
            // }
            // result = std::from_chars(substr_b.data(), substr_b.data() + substr_b.size(), num_b);
            // if (result.ec == std::errc::invalid_argument) {
            //     log()->error("Could not convert num b {}", substr_b);
            // }
            
            return num_a < num_b;
            // return false;
        });

        istream.close();
        std::fstream ostream(output / "characteristic.csv", std::fstream::out);
        for (auto l: lines) {
            ostream << l << "\n";
        }

        fs::remove(output / "temp_characteristic.csv");
    });

    log()->info("Sorting generated stats took {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
}
} // namespace trace_utils::app::stats::calculate
