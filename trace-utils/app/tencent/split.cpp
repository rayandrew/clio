#include "split.hpp"

#include <cstdio>
#include <algorithm>
#include <limits>

#include <archive.h>
#include <archive_entry.h>

#include <fmt/std.h>
#include <fmt/chrono.h>
#include <glob/glob.h>
#include <natural_sort.hpp>
#include <oneapi/tbb.h>
#include <csv2/mio.hpp>
#include <csv2/writer.hpp>

#include <mp-units/format.h>
#include <mp-units/math.h>
#include <mp-units/systems/si/si.h>

#include <indicators/progress_bar.hpp>
#include <indicators/dynamic_progress.hpp>
#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/utils.hpp>

namespace trace_utils::app::tencent {
namespace split {
const char* name = "split";
const char* description = "Tencent Split";
}
    
SplitApp::SplitApp(): App(split::name, split::description) {

}
    
void SplitApp::setup_args(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required();
    parser->add_option("-w,--window", window_str, "Choose window")->required();
    log()->info("window str {}", window_str);
}

void SplitApp::setup() {
    log()->info("window str {}", window_str);

    window = utils::parse_duration(window_str);
    log()->info("Splitting with window = {}", window);

    
    tmp_dir_path = fs::temp_directory_path() / fmt::format("{}-{}", split::name, utils::random_string(50));
    log()->info("Creating temporary directory {}", tmp_dir_path);
    fs::create_directories(tmp_dir_path);

    output = fs::weakly_canonical(output);
    fs::create_directories(output);
}


template<typename Trace, typename ProgressBar>
class FindMinTimestampReducer {
public:
    FindMinTimestampReducer(const std::vector<fs::path>& paths,
                            ProgressBar* pbar = nullptr):
        paths(paths),
        min_ts{std::numeric_limits<float>::max()},
        pbar{pbar} {}

    FindMinTimestampReducer(FindMinTimestampReducer& x, oneapi::tbb::split):
        paths(x.paths), min_ts(std::numeric_limits<float>::max()),
        pbar{x.pbar} {}

    void join(const FindMinTimestampReducer& y) {
        if (y.min_ts < min_ts) {
            min_ts = y.min_ts;
        }
    }

    void operator()(const oneapi::tbb::blocked_range<std::size_t>& r) {
        const auto& paths = this->paths;
        float min_ts = this->min_ts;

        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            auto trace_min_ts = do_work(paths[i]);
            if (pbar) pbar->tick();
            if (min_ts > trace_min_ts) min_ts = trace_min_ts;
        }
        this->min_ts = min_ts;
    }

    inline float get() const { return min_ts; }

private:
    float do_work(const fs::path& path) {
        float trace_start_time = std::numeric_limits<float>::max();
        Trace trace(path);
        trace.stream([&](const auto& item) {
            if (item.timestamp < trace_start_time) {
                trace_start_time = item.timestamp;
            }
        });
        return trace_start_time;
    }

private:
    std::vector<fs::path> paths;
    float min_ts;
    ProgressBar* pbar;
};

template<typename T = char>
class mmap_stream : public mio::basic_mmap_sink<T> {
    using base = mio::basic_mmap_sink<T>;
    using size_type = mio::basic_mmap_sink<T>::size_type;
    // using map = mio::basic_mmap_sink<T>::map;
public:
    // using base::base;

    template <typename String>
    mmap_stream(const String &path, const size_type offset = 0,
                const size_type length = mio::map_entire_file): base::basic_mmap(path, offset, length) {}
    
    void close() {
        sync();
    }
};

template <class delimiter = csv2::delimiter<','>, typename Stream = std::ofstream, typename String = std::string>
class Writer {
  String filename_;
  Stream stream_;

public:
  Writer(const String& filename): filename_(filename), stream_(Stream(filename)) {}

  ~Writer() {
    stream_.close();
  }

  template <typename Container> void write_row(Container &&row) {
    const auto &strings = std::forward<Container>(row);
    const auto delimiter_string = std::string(1, delimiter::value);
    std::copy(strings.begin(), strings.end() - 1,
              std::ostream_iterator<std::string>(stream_, delimiter_string.c_str()));
    stream_ << strings.back() << "\n";
  }

  template <typename Container> void write_rows(Container &&rows) {
    const auto &container_of_rows = std::forward<Container>(rows);
    for (const auto &row : container_of_rows) {
      write_row(row);
    }
  }
};

void SplitApp::run([[maybe_unused]] CLI::App* app) {
    // using namespace csv2;
    using namespace mp_units;
    using namespace mp_units::si;
    using namespace mp_units::si::unit_symbols;
    // using ConcurrentTable = oneapi::tbb::concurrent_hash_map<std::size_t, std::shared_ptr<Writer<csv2::delimiter<' '>>>>;
    // using ConcurrentTable = oneapi::tbb::concurrent_hash_map<std::size_t, fs::path>;
    using ConcurrentTable = oneapi::tbb::concurrent_hash_map<std::size_t, std::ofstream>;
    
    defer {       
        log()->info("Removing temporary directory", tmp_dir_path);
        fs::remove_all(tmp_dir_path);
        indicators::show_console_cursor(true);
    };
        
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);

    float trace_start_time = std::numeric_limits<float>::max();

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
            indicators::option::PrefixText{"Finding min timestamp... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };
        FindMinTimestampReducer<trace_utils::trace::TencentTrace, decltype(pbar)> r(paths, &pbar);
        parallel_reduce(oneapi::tbb::blocked_range<size_t>(0, paths.size()), r);
        trace_start_time = r.get();
        pbar.mark_as_completed();
    });

    if (trace_start_time == std::numeric_limits<float>::max()) {
        throw Exception("Cannot find min ts");
    }

    log()->info("Finding min stamp, duration {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
    
    ConcurrentTable map;
    dur = utils::get_time([&] {
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
            indicators::option::PrefixText{"Splitting and converting... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };        
        oneapi::tbb::parallel_for_each(paths.cbegin(), paths.cend(), [&](const auto& path) {
            auto prev_jitter = 0.0 * ms;
            auto prev_time = 0.0 * ms;
            std::random_device rd;
            std::mt19937 e2(rd());

            trace_utils::trace::TencentTrace trace(path);
            trace.stream([&](const auto& item) {
                auto it = item;
                auto timestamp = (it.timestamp - trace_start_time) * ms;
                if (timestamp == prev_time) {
                    std::uniform_real_distribution<> dist((prev_jitter).numerical_value_in(ms) + 1.0, (prev_jitter).numerical_value_in(ms) + 5.0);
                    auto jitter = dist(e2) * ms;
                    timestamp += jitter;
                    prev_jitter = jitter;
                } else {
                    prev_jitter = 0.0 * ms;
                }

                prev_time = timestamp;
                it.timestamp = timestamp.numerical_value_in(ms);
                auto chunk_d = std::floor((timestamp / window.in(ms)).numerical_value_in(mp_units::one));
                auto current_chunk = static_cast<std::size_t>(chunk_d);
                // log()->info("Current chunk {} ts={} window={}", current_chunk, timestamp, window.in(ms));

                ConcurrentTable::accessor accessor;
                bool res = map.insert(accessor, current_chunk);
                // if (res && accessor->second.empty()) {
                if (res && !accessor->second.is_open()) {
                    // log()->info("here");
                    auto stem_path = path.stem();
                    auto out_path = fs::weakly_canonical(output / stem_path);
                    auto archive_file_path = out_path.replace_extension(".tgz");
                    auto temp_path = (tmp_dir_path / fmt::format("chunk-{}", current_chunk)).replace_extension(".csv");
                    log()->debug("Creating temp file: {}", temp_path);
                    accessor->second.open(temp_path, std::ios_base::app);
                    // accessor->second.reset(new Writer<csv2::delimiter<' '>>(temp_path));
                }
                // std::ofstream stream;
                // stream.open(accessor->second, std::ios_base::app);
                // csv2::Writer<csv2::delimiter<' '>> writer(stream);
                // writer.write_row(it.to_vec());
                // if (found) {
                //     log()->info("here");
                //     accessor->second->write_row(it.to_vec());
                // } else {
                //     if (map.insert(accessor, current_chunk)) {
                //         // log()->info("Creating temporary file");
                        
                //     }
                // }
            });

            pbar.tick();
        });
    });

    log()->info("Splitting takes {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
}
} // namespace trace_utils::app::tencent
