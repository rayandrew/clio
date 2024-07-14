#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <chrono>
#include <type_traits>

#include <oneapi/tbb.h>

#include <mp-units/systems/si/si.h>
#include <mp-units/systems/isq/isq.h>

#include <regex>

#include <scope_guard.hpp>

#include <trace-utils/internal/filesystem.hpp>
#include <trace-utils/exception.hpp>

#define defer DEFER

namespace trace_utils {
namespace utils {
std::string random_string(std::size_t length);

// mp_units::QuantityOf<mp_units::isq::time> decltype(auto)
template<mp_units::QuantityOf<mp_units::isq::time> D>
void parse_duration(const std::string& duration_str, D& dest) {
    using namespace mp_units;
    using namespace mp_units::si;
    using namespace mp_units::si::unit_symbols;

    constexpr auto q = R"((\d+\.?\d*)\s*([a-zA-Z]+))";
    static std::regex regex(q);
    std::smatch match;

    if (!std::regex_match(duration_str, match, regex)) {
        throw std::invalid_argument("Invalid duration format: " + duration_str);
    }

    // Extract the numerical value and unit from the regex match
    double value = std::stod(match[1].str());
    std::string unit = match[2].str();
    
    if (unit.find("ms") != std::string::npos) {
        dest = value * ms;
        return;
    }

    if (unit.find("us") != std::string::npos) {
        dest = value * us;
        return;
    }

    if (unit.find("ns") != std::string::npos) {
        dest = value * ns;
        return;
    }

    if (unit.find("s") != std::string::npos) {
        dest = value * second;
        return;
    }

    if (unit.find("m") != std::string::npos) {
        dest = value * minute;
        return;
    }

    if (unit.find("h") != std::string::npos) {
        dest = value * hour;
        return;
    }

    if (unit.find("d") != std::string::npos) {
        dest = value * day;
        return;
    }
    
    throw Exception(fmt::format("Unit {} is not defined!", unit));
}

inline std::chrono::time_point<std::chrono::steady_clock> get_time() {
    return std::chrono::steady_clock::now();
}

template<typename Func>
auto get_time(Func&& func) {
    auto start = get_time();
    func();
    auto end = get_time();
    return end - start;
}

using f_sec = std::chrono::duration<float>;

template<typename Class, typename Enabled = void> 
struct is_progress_bar_s
{
    static constexpr bool value = false;  
};

template<typename Class> 
struct is_progress_bar_s<Class, std::enable_if_t<std::is_member_function_pointer_v<decltype(&Class::tick)>>> 
{
    static constexpr bool value = std::is_member_function_pointer_v<decltype(&Class::tick)>;
    using type = bool;
};

template<typename Class> 
constexpr bool is_progress_bar()
{
    return is_progress_bar_s<Class>::value;
};

    
template<typename Trace, typename ProgressBar = void>
class FindMinTimestampReducer {
public:

    template<typename... Dummy, typename P = ProgressBar, std::enable_if_t<is_progress_bar_s<P>::value, bool> = true>
    FindMinTimestampReducer(const std::vector<fs::path>& paths,
                            P* pbar = nullptr):
        paths(paths),
        min_ts{std::numeric_limits<double>::max()},
        pbar{pbar} {
        static_assert(sizeof...(Dummy)==0, "Do not specify template arguments!");
    }

    template<typename... Dummy, typename P = ProgressBar, std::enable_if_t<!is_progress_bar_s<P>::value, bool> = true>
    FindMinTimestampReducer(const std::vector<fs::path>& paths):
        paths(paths),
        min_ts{std::numeric_limits<double>::max()} {
        static_assert(sizeof...(Dummy)==0, "Do not specify template arguments!");
    }

    template<typename... Dummy, typename P = ProgressBar, std::enable_if_t<is_progress_bar_s<P>::value, bool> = true>
    FindMinTimestampReducer(FindMinTimestampReducer& x, oneapi::tbb::split):
        paths(x.paths), min_ts(std::numeric_limits<double>::max()),
        pbar{x.pbar} {
        static_assert(sizeof...(Dummy)==0, "Do not specify template arguments!");
    }

    template<typename... Dummy, typename P = ProgressBar, std::enable_if_t<!is_progress_bar_s<P>::value, bool> = true>
    FindMinTimestampReducer(FindMinTimestampReducer& x, oneapi::tbb::split):
        paths(x.paths), min_ts(std::numeric_limits<double>::max()) {
        static_assert(sizeof...(Dummy)==0, "Do not specify template arguments!");
    }

    void join(const FindMinTimestampReducer& y) {
        if (y.min_ts < min_ts) {
            min_ts = y.min_ts;
        }
    }
    
    template<typename P = ProgressBar, std::enable_if_t<is_progress_bar_s<P>::value, bool> = true>
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

    template<typename P = ProgressBar, std::enable_if_t<!is_progress_bar_s<P>::value, bool> = true>
    void operator()(const oneapi::tbb::blocked_range<std::size_t>& r) {
        const auto& paths = this->paths;
        double min_ts = this->min_ts;

        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            auto trace_min_ts = do_work(paths[i]);
            if (min_ts > trace_min_ts) min_ts = trace_min_ts;
        }
        this->min_ts = min_ts;
    }

    inline decltype(Trace::Entry::timestamp) get() const { return min_ts; }

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
    decltype(Trace::Entry::timestamp) min_ts;
    ProgressBar* pbar;
};

template<typename Trace, typename ProgressBar>
auto get_min_timestamp(const std::vector<fs::path>& paths, ProgressBar& pbar) {
    static_assert(is_progress_bar<ProgressBar>(), "pbar needs to have `tick` function!");
    FindMinTimestampReducer<Trace, ProgressBar> r(paths, &pbar);
    oneapi::tbb::parallel_reduce(oneapi::tbb::blocked_range<size_t>(0, paths.size()), r);
    return r.get();
}

template<typename Trace>
auto get_min_timestamp(const std::vector<fs::path>& paths) {
    FindMinTimestampReducer<Trace, void> r(paths);
    oneapi::tbb::parallel_reduce(oneapi::tbb::blocked_range<size_t>(0, paths.size()), r);
    return r.get();
}
} // namespace utils
} // namespace trace_utils

#endif
