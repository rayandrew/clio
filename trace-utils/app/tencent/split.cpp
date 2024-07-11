#include "split.hpp"

#include <cstdio>
#include <algorithm>
#include <limits>
#include <utility>
#include <tuple>
#include <atomic>
#include <span>

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

#include <indicators/block_progress_bar.hpp>
#include <indicators/cursor_control.hpp>
#include <indicators/termcolor.hpp>

#include <trace-utils/logger.hpp>
#include <trace-utils/trace.hpp>
#include <trace-utils/trace/replayer.hpp>
#include <trace-utils/trace/tencent.hpp>
#include <trace-utils/utils.hpp>

namespace trace_utils::app::tencent {
namespace split {
const char* name = "split";
const char* description = "Tencent Split";
}
    
SplitApp::SplitApp(): App(split::name, split::description) {

}

SplitApp::~SplitApp() {
    log()->info("Removing temporary directory", tmp_dir_path);
    fs::remove_all(tmp_dir_path);
    indicators::show_console_cursor(true);
}
    
void SplitApp::setup_args(CLI::App *app) {
    parser = create_subcommand(app);
    parser->add_option("-i,--input", input, "Input directory")->required()->check(CLI::ExistingDirectory);
    parser->add_option("-o,--output", output, "Output directory")->required();
    parser->add_option("-w,--window", window_str, "Choose window")->required();
}

void SplitApp::setup() {
    log()->info("window str {}", window_str);

    utils::parse_duration(window_str, window);
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
        min_ts{std::numeric_limits<double>::max()},
        pbar{pbar} {}

    FindMinTimestampReducer(FindMinTimestampReducer& x, oneapi::tbb::split):
        paths(x.paths), min_ts(std::numeric_limits<double>::max()),
        pbar{x.pbar} {}

    void join(const FindMinTimestampReducer& y) {
        if (y.min_ts < min_ts) {
            min_ts = y.min_ts;
        }
    }

    void operator()(const oneapi::tbb::blocked_range<std::size_t>& r) {
        const auto& paths = this->paths;
        double min_ts = this->min_ts;

        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            auto trace_min_ts = do_work(paths[i]);
            if (pbar) pbar->tick();
            if (min_ts > trace_min_ts) min_ts = trace_min_ts;
        }
        this->min_ts = min_ts;
    }

    inline double get() const { return min_ts; }

private:
    double do_work(const fs::path& path) {
        double trace_start_time = std::numeric_limits<double>::max();
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
    double min_ts;
    ProgressBar* pbar;
};

void SplitApp::run([[maybe_unused]] CLI::App* app) {
    using namespace mp_units;
    using namespace mp_units::si;
    using namespace mp_units::si::unit_symbols;
    using Mutex = oneapi::tbb::mutex;
    constexpr std::size_t total_buffer_size = 10000;
    using ConcurrentTable = oneapi::tbb::concurrent_hash_map<std::size_t, std::tuple<Mutex, std::vector<std::vector<std::string>>, std::atomic_size_t>>;
        
    auto input_path = fs::canonical(input) / "*.tgz";
    log()->info("Globbing over {}", input_path);
    auto paths = glob::glob(input_path);
    std::sort(paths.begin(), paths.end(), SI::natural::compare<std::string>);

    auto unsorted_tmp_dir_path = tmp_dir_path / "unsorted";
    fs::create_directories(unsorted_tmp_dir_path);

    auto sorted_tmp_dir_path = tmp_dir_path / "sorted";
    fs::create_directories(sorted_tmp_dir_path);
    

    double trace_start_time = std::numeric_limits<double>::max();

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
        FindMinTimestampReducer<trace::TencentTrace, decltype(pbar)> r(paths, &pbar);
        oneapi::tbb::parallel_reduce(oneapi::tbb::blocked_range<size_t>(0, paths.size()), r);
        trace_start_time = r.get();
        pbar.mark_as_completed();
    });

    if (trace_start_time == std::numeric_limits<double>::max()) {
        throw Exception("Cannot find min ts");
    }

    log()->info("Finding min stamp takes {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
    
    dur = utils::get_time([&] {
        ConcurrentTable map;
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
            trace::TencentTrace trace(path);
            trace.stream([&](const auto& item) {
                auto it = item;
                it.timestamp -= trace_start_time;
                auto timestamp = it.timestamp * ms;
                double chunk_d = std::floor((timestamp / window.in(ms)).numerical_value_in(mp_units::one));
                auto current_chunk = static_cast<std::size_t>(chunk_d);

                ConcurrentTable::accessor accessor;
                bool new_chunk = map.insert(accessor, current_chunk);
                if (new_chunk) {
                    auto&& l = std::get<0>(accessor->second);
                    Mutex::scoped_lock lock(l); // , /* is_writer */ true);
                    auto&& v = std::get<1>(accessor->second);
                    auto&& c = std::get<2>(accessor->second);
                    v.push_back(it.to_vec());
                    c++;
                } else {
                    auto&& l = std::get<0>(accessor->second);              
                    Mutex::scoped_lock lock(l); // , /* is_writer */ false);
                    auto&& c = std::get<2>(accessor->second);
                    if (c > total_buffer_size - 1) {
                        auto stem_path = path.stem();
                        auto out_path = fs::weakly_canonical(output / stem_path);
                        auto temp_path = (unsorted_tmp_dir_path / fmt::format("chunk-{}", current_chunk)).replace_extension(".csv");

                        auto&& v = std::get<1>(accessor->second);
                            
                        log()->debug("Creating temp file: {}", temp_path);

                        std::fstream stream(temp_path, std::fstream::in | std::fstream::out | std::fstream::app);
                        csv2::Writer<csv2::delimiter<' '>, std::fstream> writer(stream);
                        writer.write_rows(std::span(v).subspan(0, c));
                        
                        v.clear();
                        v.resize(0);
                        std::vector<std::vector<std::string>>().swap(v);
                        c = 0;

                    } else {
                        auto&& v = std::get<1>(accessor->second);
                        auto&& c = std::get<2>(accessor->second);
                        if (v.size() >= total_buffer_size && c < total_buffer_size - 1) {
                            v.at(++c) = it.to_vec();
                        } else {
                            v.push_back(it.to_vec());
                            c++;
                        }
                    }
                }
            });

            pbar.tick();
        });

        pbar.mark_as_completed();

        indicators::BlockProgressBar pbar2{
            indicators::option::ForegroundColor{indicators::Color::yellow},
            indicators::option::FontStyles{
                std::vector<indicators::FontStyle>{
                    indicators::FontStyle::bold
                }
            },
            indicators::option::MaxProgress{map.size()},
            indicators::option::PrefixText{"Saving split leftovers... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };

        oneapi::tbb::parallel_for(map.range(), [&](auto &r)  {
            for(auto i = r.begin(); i != r.end(); i++) {
                auto&& v = std::get<1>(i->second);
                auto&& c = std::get<2>(i->second);

                if (c > 0) {
                    auto temp_path = (unsorted_tmp_dir_path / fmt::format("chunk-{}", i->first)).replace_extension(".csv");

                    std::fstream stream(temp_path, std::fstream::in | std::fstream::out | std::fstream::app);
                    csv2::Writer<csv2::delimiter<' '>, std::fstream> writer(stream);
                    writer.write_rows(std::span(v).subspan(0, c));
                    // writer.write_rows(v);
                }
                pbar2.tick();
            }
        });

        pbar2.mark_as_completed();
        
        ConcurrentTable().swap(map);

    });

    log()->info("Splitting takes {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));


    dur = utils::get_time([&] {
        // All this data below is already in ReplayerTrace::Entry format
        auto unsorted_temp_paths = glob::glob(unsorted_tmp_dir_path / "*.csv");
        std::sort(unsorted_temp_paths.begin(), unsorted_temp_paths.end(), SI::natural::compare<std::string>);
        
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
            indicators::option::MaxProgress{unsorted_temp_paths.size()},
            indicators::option::PrefixText{"Sorting... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };


        oneapi::tbb::parallel_for_each(unsorted_temp_paths.cbegin(), unsorted_temp_paths.cend(), [&](const auto& path) {
            trace::ReplayerTrace trace(path);

            auto vecs = trace(/* filter */[]([[maybe_unused]] const auto& item) { return true; });
            
            std::sort(vecs.begin(), vecs.end(), [](const auto& a, const auto& b) {
                return a.timestamp < b.timestamp;
            });

            // sorted already
            std::random_device rd;
            std::mt19937 e2(rd());
            for (std::size_t i = 1; i < vecs.size(); ++i) {
                const auto& prev_item = vecs[i - 1];
                auto& it = vecs[i];
                if (it.timestamp <= prev_item.timestamp) {
                    std::uniform_real_distribution<> dist(1.0, 4.0);
                    auto jitter = dist(e2);
                    it.timestamp = prev_item.timestamp + jitter;
                }
            };
            
            auto sorted_path = (sorted_tmp_dir_path / path.stem()).replace_extension(".csv");
            std::ofstream stream(sorted_path);
            csv2::Writer<csv2::delimiter<' '>> writer(stream);
            
            for (const auto& row: vecs) {
                writer.write_row(row.to_vec());
            }
            
            pbar.tick();
        });

        pbar.mark_as_completed();
    });

    log()->info("Sorting takes {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));

    dur = utils::get_time([&] {
        auto sorted_temp_paths = glob::glob(sorted_tmp_dir_path / "*.csv");
        std::sort(sorted_temp_paths.begin(), sorted_temp_paths.end(), SI::natural::compare<std::string>);

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
            indicators::option::MaxProgress{sorted_temp_paths.size()},
            indicators::option::PrefixText{"Archiving... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };
        
        oneapi::tbb::parallel_for_each(sorted_temp_paths.cbegin(), sorted_temp_paths.cend(), [&](const auto& path) {
            auto stem_path = path.stem();
            auto out_path = fs::weakly_canonical(output / stem_path);
            auto archive_file_path = out_path.replace_extension(".tgz");
            
            struct archive *a;
            struct stat st;
            struct archive_entry *entry;

            a = archive_write_new();
            defer { archive_write_free(a); };
            archive_write_add_filter_gzip(a);
            archive_write_set_format_pax_restricted(a);
            archive_write_open_filename(a, archive_file_path.c_str());
            defer { archive_write_close(a); };

            entry = archive_entry_new();
            defer { archive_entry_free(entry); };

            stat(path.c_str(), &st);

            archive_entry_set_pathname(entry, stem_path.c_str());
            archive_entry_set_filetype(entry, AE_IFREG);
            archive_entry_set_perm(entry, 0644);
            archive_entry_copy_stat(entry, &st);
            archive_write_header(a, entry);

            auto file = mio::mmap_source(path.string());
            if (!file.is_open() || !file.is_mapped()) {
                throw Exception(fmt::format("Cannot mmap file {}", path));
            }
            archive_write_data(a, file.data(), file.mapped_length());

            pbar.tick();
        });

        pbar.mark_as_completed();
    });

    log()->info("Archiving takes {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));
}
} // namespace trace_utils::app::tencent
