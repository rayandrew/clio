#ifndef __TRACE_UTILS_CHARACTERISTIC_HPP__
#define __TRACE_UTILS_CHARACTERISTIC_HPP__

#include <string>

#include <trace-utils/stats.hpp>
#include <trace-utils/trace.hpp>
#include <trace-utils/trace/replayer.hpp>
#include <trace-utils/trace/replayed.hpp>

namespace trace_utils {
struct RawCharacteristic {
    // count
    std::size_t num_io;
    std::size_t read_count;
    std::size_t write_count;

    // iops
    double iops;
    double read_iops;
    double write_iops;

    // time
    double start_time;
    double end_time;
    std::string ts_unit;
    double duration;
    double duration_in_sec;

    // bandwidth
    double raw_bandwidth;
    double raw_read_bandwidth;
    double raw_write_bandwidth;

    // I/O type ratio
    double write_ratio;
    double read_ratio;
    double write_over_read_ratio;
    double read_over_write_ratio;

    /// ----------------
    /// statistic
    /// ----------------

    // size
    Statistic size;
    Statistic read_size;
    Statistic write_size;

    // size ratio
    Statistic write_size_ratio;
    Statistic read_size_ratio;
    Statistic write_size_over_read_size_ratio;
    Statistic read_size_over_write_size_ratio;

    // offset
    Statistic offset;
    Statistic read_offset;
    Statistic write_offset;

    // iat
    Statistic iat;
    Statistic read_iat;
    Statistic write_iat;

    static RawCharacteristic from(const TraceCombiner<trace::ReplayerTrace>& trace, bool parallel = true);
    static RawCharacteristic from(const trace::ReplayerTrace& trace, bool parallel = true);
    
    virtual std::vector<std::string> header();
    virtual std::vector<std::string> values();
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

    // empirical IAT
    Statistic emp_iat;
    Statistic emp_read_iat;
    Statistic emp_write_iat;

    static ReplayedCharacteristic from(const TraceCombiner<trace::ReplayedTrace>& trace, bool parallel = true);
    static ReplayedCharacteristic from(const trace::ReplayedTrace& trace, bool parallel = true);
    
    virtual std::vector<std::string> header();
    virtual std::vector<std::string> values();
};
} // namespace trace_utils

#endif
