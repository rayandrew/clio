#ifndef __TRACE_UTILS_CHARACTERISTIC_HPP__
#define __TRACE_UTILS_CHARACTERISTIC_HPP__

#include <string>

#include <trace-utils/stats.hpp>
#include <trace-utils/trace/replayer.hpp>
#include <trace-utils/trace/replayed.hpp>

namespace trace_utils {
struct RawCharacteristic {
    // count
    std::size_t num_io;
    std::size_t read_count;
    std::size_t write_count;

    // time
    double start_time;
    double end_time;
    std::string ts_unit;
    double duration;
    double duration_in_sec;

    // size
    Statistic size;
    Statistic read_size;
    Statistic write_size;

    // ofset
    Statistic offset;

    // iat
    Statistic iat;
    Statistic read_iat;
    Statistic write_iat;

    // bandwidth
    double raw_bandwidth;
    double raw_read_bandwidth;
    double raw_write_bandwidth;

    static RawCharacteristic from(const trace::ReplayerTrace& trace, bool parallel = true);
};

struct ReplayedCharacteristic : public RawCharacteristic {
    // latency
    Statistic latency;
    Statistic read_latency;
    Statistic write_latency;

    // throughput
    Statistic throughput;
    Statistic read_throughput;
    Statistic write_throughput;

    // empirical bandwidth per I/O
    Statistic emp_bandwidth;
    Statistic emp_read_bandwidth;
    Statistic emp_write_bandwidth;

    static ReplayedCharacteristic from(const trace::ReplayedTrace& trace);
};
} // namespace trace_utils

#endif
