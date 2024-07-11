#include "split.hpp"

#include <cstdio>
#include <algorithm>
#include <limits>
#include <utility>
#include <tuple>
#include <atomic>

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
    // log()->info("Removing temporary directory", tmp_dir_path);
    // fs::remove_all(tmp_dir_path);
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
    using Mutex = oneapi::tbb::rw_mutex;
    // using ConcurrentTable = oneapi::tbb::concurrent_hash_map<std::size_t, std::pair<Mutex, oneapi::tbb::concurrent_vector<std::vector<std::string>>>>;
    
    const std::size_t total_buffer_size = 10000;
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
                    // auto [l, v, c] = accessor->second;
                    auto&& l = std::get<0>(accessor->second);
                    Mutex::scoped_lock lock(l,
                                           /* is_writer */ true);
                    auto&& v = std::get<1>(accessor->second);
                    auto&& c = std::get<2>(accessor->second);
                    // v.resize(total_buffer_size);
                    // c = 0;
                    v.push_back(it.to_vec());
                    c++;
                    // log()->info("here 1 {}", c);
                    // v.at(++c) = it.to_vec();
                    // v.insert(v.begin() + c + 1, it.to_vec());
                    // c++;
                    
                    // v[++c] = it.to_vec();
                    // c++;
                } else {
                    std::size_t size;
                    {
                        auto&& l = std::get<0>(accessor->second);
                        // auto [l, v, c] = accessor->second;                        
                        Mutex::scoped_lock lock(l,
                                                /* is_writer */ false);
                        // auto&& v = std::get<1>(accessor->second);
                        auto&& c = std::get<2>(accessor->second);
                        size = c;
                    }
                    // if (size == 0) {
                    //     log()->info("Size {}", size);
                    // }
                    if (size > total_buffer_size - 1) {
                        // auto&& [l, v, c] = accessor->second;
                        auto&& l = std::get<0>(accessor->second);
                        auto stem_path = path.stem();
                        auto out_path = fs::weakly_canonical(output / stem_path);
                        auto temp_path = (unsorted_tmp_dir_path / fmt::format("chunk-{}", current_chunk)).replace_extension(".csv");
                        {
                            Mutex::scoped_lock lock(l,
                                                    /* is_writer */ true);
                            auto&& v = std::get<1>(accessor->second);
                            auto&& c = std::get<2>(accessor->second);
                            
                            log()->debug("Creating temp file: {}", temp_path);
                            std::ofstream stream;
                            stream.open(temp_path, std::ios_base::app);
                            csv2::Writer<csv2::delimiter<' '>> writer(stream);
                            // std::fstream stream(temp_path, std::fstream::in | std::fstream::out | std::fstream::app);
                            // csv2::Writer<csv2::delimiter<' '>, std::fstream> writer(stream);

                            for (std::size_t i =  0; i < c; ++i) {
                                auto row = v[i];
                                // const auto &strings = std::forward<std::string>(row);
                                const auto delimiter_string = std::string(1, ',');
                                std::copy(row.begin(), row.end() - 1,
                                          std::ostream_iterator<std::string>(stream, delimiter_string.c_str()));
                                stream << row.back() << "\n";
                            }
                            // writer.write_rows(accessor->second.second);
                            // for (auto i = accessor->second.second.cbegin(); i < accessor->second.second.cend(); ++i) {
                                // writer.write_row(*i);
                            // }

                            // accessor->second.second.resize(0);
                            // accessor->second.second.clear();
                            v.clear();
                            v.resize(0);
                            std::vector<std::vector<std::string>>().swap(v);
                            c = 0;
                            // oneapi::tbb::concurrent_vector<std::vector<std::string>>().swap(accessor->second.second);
                        }

                    } else {
                        // auto [l, v, c] = accessor->second;
                        auto&& l = std::get<0>(accessor->second);
                        Mutex::scoped_lock lock(l,
                                                /* is_writer */ true);
                        auto&& v = std::get<1>(accessor->second);
                        auto&& c = std::get<2>(accessor->second);
                        // log()->info("here 2 {}", c);
                        if (v.size() >= total_buffer_size && c < total_buffer_size - 1) {
                            v.at(++c) = it.to_vec();
                        } else {
                            v.push_back(it.to_vec());
                            c++;
                        }
                        // v.at(++c) = it.to_vec();
                        // v.insert(v.begin() + c + 1, it.to_vec());
                        // c++;
                        // v[++c] = it.to_vec();
                                                
                        // accessor->second.second.push_back(it.to_vec());
                        // c++;
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
                // auto&& l = std::get<0>(accessor->);
                auto&& v = std::get<1>(i->second);
                auto&& c = std::get<2>(i->second);
                // log()->info("Chunk {}", i->first);

                if (c > 0) {
                    // auto stem_path = path.stem();
                    // auto out_path = fs::weakly_canonical(output / stem_path);
                    auto temp_path = (unsorted_tmp_dir_path / fmt::format("chunk-{}", i->first)).replace_extension(".csv");

                    std::fstream stream(temp_path, std::fstream::in | std::fstream::out | std::fstream::app);
                    csv2::Writer<csv2::delimiter<' '>, std::fstream> writer(stream);
                    writer.write_rows(v);
                    
                    // for (std::size_t i =  0; i < c; ++i) {
                    //     auto row = v[i];
                    //     // const auto &strings = std::forward<std::string>(row);
                    //     const auto delimiter_string = std::string(1, ',');
                    //     std::copy(row.begin(), row.end() - 1,
                    //               std::ostream_iterator<std::string>(stream, delimiter_string.c_str()));
                    //     stream << row.back() << "\n";
                    // }
                
                    // for (auto row = i->second.second.cbegin(); row < i->second.second.cend(); ++row) {
                    //     writer.write_row(*row);
                    // }
                }
                pbar2.tick();
            }
        });

        pbar2.mark_as_completed();
    });

    // oneapi::tbb::parallel_for(map.range(), [&](auto &r)  {
    //     for(auto i = r.begin(); i != r.end(); i++) {
    //         // auto stem_path = path.stem();
    //         // auto out_path = fs::weakly_canonical(output / stem_path);
    //         auto temp_path = (unsorted_tmp_dir_path / fmt::format("chunk-{}", i->first)).replace_extension(".csv");

    //         std::fstream stream(temp_path, std::fstream::in | std::fstream::out | std::fstream::app);
    //         csv2::Writer<csv2::delimiter<' '>, std::fstream> writer(stream);
                        
    //         for (auto row = i->second.second.cbegin(); row < i->second.second.cend(); ++row) {
    //             writer.write_row(*row);
    //         }
    //     }
    // });

    log()->info("Splitting takes {}", std::chrono::duration_cast<std::chrono::milliseconds>(dur));


    // All this data below is already in ReplayerTrace::Entry format
    auto temp_paths = glob::glob(unsorted_tmp_dir_path / "*.csv");
    std::sort(temp_paths.begin(), temp_paths.end(), SI::natural::compare<std::string>);

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
            indicators::option::MaxProgress{temp_paths.size()},
            indicators::option::PrefixText{"Sorting... "},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
        };


        oneapi::tbb::parallel_for_each(temp_paths.cbegin(), temp_paths.cend(), [&](const auto& path) {
            trace::ReplayerTrace trace(path);

            auto vecs = trace(/* filter */[]([[maybe_unused]] const auto& item) { return true; });
            // oneapi::tbb::parallel_sort(vecs.begin(), vecs.end(), [](const auto& a, const auto& b) {
            std::sort(vecs.begin(), vecs.end(), [](const auto& a, const auto& b) {
                return a.timestamp < b.timestamp;
            });

            // sorted already
            std::random_device rd;
            std::mt19937 e2(rd());
            
            // std::transform(vecs.cbegin(), vecs.cend(), vecs.begin(), [&](const auto& item) {
            for (std::size_t i = 1; i < vecs.size(); ++i) {
                const auto& prev_item = vecs[i - 1];
                auto& it = vecs[i];
                if (it.timestamp <= prev_item.timestamp) {
                    // double prev_jitter_d = prev_jitter;
                    std::uniform_real_distribution<> dist(0.1, 2.0);
                    auto jitter = dist(e2);
                    it.timestamp = prev_item.timestamp + jitter;
                }

                // double new_ts = timestamp;

                // prev_time = timestamp;
            }
            //     return it;
            // });
            
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
}
} // namespace trace_utils::app::tencent
