#include <trace-utils/characteristic.hpp>

#include <string_view>

#include <oneapi/tbb.h>

#include <trace-utils/utils.hpp>

namespace trace_utils {
constexpr const char* non_stats_col[] = {
    "num_io",
    "read_count",
    "write_count",

    "iops",
    "read_iops",
    "write_iops",

    "start_time",
    "end_time",
    "ts_unit",
    "duration",
    "duration_in_sec",

    "write_ratio",
    "read_ratio",
    "write_over_read_ratio",
    "read_over_write_ratio",
};

namespace raw_characteristic {
constexpr const char* stats_col[] = {

    "raw_bandwidth",
    "raw_read_bandwidth",
    "raw_write_bandwidth",

    "size",
    "read_size",
    "write_size",

    "write_size_ratio",
    "read_size_ratio",
    "write_size_over_read_size_ratio",
    "read_size_over_write_size_ratio",

    "offset",
    "read_offset",
    "write_offset",

    "iat",
    "read_iat",
    "write_iat",
};
} // namespace raw_characteristic

namespace replayed_characteristic {
constexpr const char* stats_col[] = {
    "raw_bandwidth",
    "raw_read_bandwidth",
    "raw_write_bandwidth",

    "size",
    "read_size",
    "write_size",

    "write_size_ratio",
    "read_size_ratio",
    "write_size_over_read_size_ratio",
    "read_size_over_write_size_ratio",

    "offset",
    "read_offset",
    "write_offset",

    "iat",
    "read_iat",
    "write_iat",

    "latency",
    "read_latency",
    "write_latency",

    "throughput",
    "read_throughput",
    "write_throughput",

    "emp_bandwidth",
    "emp_read_bandwidth",
    "emp_write_bandwidth",

    "emp_iat",
    "emp_read_iat",
    "emp_write_iat"
};
} // namespace replayed_characteristic

template<typename T>
static RawCharacteristic calc_raw_characteristic(const T& trace, bool parallel) {
    std::vector<double> offset;
    std::vector<double> read_offset;
    std::vector<double> write_offset;
    
    std::vector<double> iat;
    std::vector<double> read_iat;
    std::vector<double> write_iat;
    
    std::vector<double> size;
    std::vector<double> read_size;
    std::vector<double> write_size;

    std::size_t num_io = 0;
    std::size_t read_count = 0;
    std::size_t write_count = 0;

    double start_time = 0.0;
    double end_time = 0.0;
    std::string ts_unit;
    double duration = 0.0;
    double duration_in_sec = 0.0;
    
    double last_time = 0.0;
    double last_read_time = 0.0;
    double last_write_time = 0.0;
        
    trace.raw_stream([&](const auto& item) {
        if (num_io == 0) {
            start_time = item.timestamp;
        }
        
        num_io += 1;
        end_time = item.timestamp;
        iat.push_back(item.timestamp - last_time);
        offset.push_back(item.offset);
        size.push_back(item.size);

        if (item.read) {
            read_count += 1;
            read_iat.push_back(item.timestamp - last_read_time);
            read_offset.push_back(item.offset);
            read_size.push_back(item.size);
            last_read_time = item.timestamp;
        } else {
            write_count += 1;
            write_iat.push_back(item.timestamp - last_write_time);
            write_offset.push_back(item.offset);
            write_size.push_back(item.size);
            last_write_time = item.timestamp;
        }
        last_time = item.timestamp;    
    });

    if (num_io > 0) {
        duration = end_time - start_time;
        ts_unit = "ms";
        duration_in_sec = duration / 1000.0;
    }

    RawCharacteristic characteristic;
    characteristic.num_io = num_io;
    characteristic.read_count = read_count;
    characteristic.write_count = write_count;
    characteristic.start_time = start_time;
    characteristic.end_time = end_time;
    characteristic.ts_unit = ts_unit;
    characteristic.duration = duration;
    characteristic.duration_in_sec = duration_in_sec;

    if (parallel) {
        oneapi::tbb::parallel_invoke(
            [&]() { characteristic.size = Statistic::from(size, true); },
            [&]() { characteristic.read_size = Statistic::from(read_size, true); },
            [&]() { characteristic.write_size = Statistic::from(write_size, true); },
            [&]() { characteristic.offset = Statistic::from(offset, true); },
            [&]() { characteristic.read_offset = Statistic::from(read_offset, true); },
            [&]() { characteristic.write_offset = Statistic::from(write_offset, true); },
            [&]() { characteristic.iat = Statistic::from(iat, true); },
            [&]() { characteristic.read_iat = Statistic::from(read_iat, true); },
            [&]() { characteristic.write_iat = Statistic::from(write_iat, true); }
        );
    } else {
        characteristic.size = Statistic::from(size, false);
        characteristic.read_size = Statistic::from(read_size, false);
        characteristic.write_size = Statistic::from(write_size, false);
        characteristic.offset = Statistic::from(offset, false);
        characteristic.read_offset = Statistic::from(read_offset, false);
        characteristic.write_offset = Statistic::from(write_offset, false);
        characteristic.iat = Statistic::from(iat, false);
        characteristic.read_iat = Statistic::from(read_iat, false);
        characteristic.write_iat = Statistic::from(write_iat, false);
    }

    characteristic.iops = characteristic.num_io / duration_in_sec;
    characteristic.read_iops = characteristic.read_count / duration_in_sec;
    characteristic.write_iops = characteristic.write_count / duration_in_sec;

    characteristic.read_size_ratio = characteristic.read_size / characteristic.size;
    characteristic.write_size_ratio = characteristic.write_size / characteristic.size;
    characteristic.write_size_over_read_size_ratio = characteristic.write_size / characteristic.read_size;
    characteristic.read_size_over_write_size_ratio = characteristic.read_size / characteristic.write_size;
    
    characteristic.read_ratio = characteristic.read_count / characteristic.num_io;
    characteristic.write_ratio = characteristic.write_count / characteristic.num_io;
    characteristic.write_over_read_ratio = characteristic.write_count / characteristic.read_count;
    characteristic.read_over_write_ratio = characteristic.read_count / characteristic.write_count;

    characteristic.raw_bandwidth = characteristic.size / duration_in_sec; // bytes/sec
    characteristic.raw_read_bandwidth = characteristic.read_size / duration_in_sec; // bytes/sec
    characteristic.raw_write_bandwidth = characteristic.write_size / duration_in_sec; // bytes/sec

    return characteristic;
}

RawCharacteristic RawCharacteristic::from(const TraceCombiner<trace::ReplayerTrace>& trace, bool parallel) {
    return calc_raw_characteristic(trace, parallel);
}
    
RawCharacteristic RawCharacteristic::from(const trace::ReplayerTrace& trace, bool parallel) {
    return calc_raw_characteristic(trace, parallel);
}
    
std::vector<std::string> RawCharacteristic::header() {
    using std::operator""sv;
    
    std::vector<std::string> v(non_stats_col, non_stats_col + sizeof(non_stats_col) / sizeof(non_stats_col[0]));
    for (const auto& col: raw_characteristic::stats_col) {
        tsl::ordered_map<std::string, double> m;
        auto col_sv = std::string_view{col};
        if (col_sv == "raw_bandwidth"sv) {
            m = raw_bandwidth.to_map();
        } else if (col_sv == "raw_read_bandwidth"sv) {
            m = raw_read_bandwidth.to_map();
        } else if (col_sv == "raw_write_bandwidth"sv) {
            m = raw_write_bandwidth.to_map();
        } else if (col_sv == "size"sv) {
            m = size.to_map();
        } else if (col_sv == "read_size"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_size"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_size_over_read_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_size_over_write_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "offset"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_offset"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_offset"sv) {
            m = read_size.to_map();
        } else if (col_sv == "iat"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_iat"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_iat"sv) {
            m = read_size.to_map();
        }
        

        for (const auto& p: m) {
            v.push_back(fmt::format("{}_{}", col, p.first));
        }
    }

    return v;
}
    
std::vector<std::string> RawCharacteristic::values() {
    using std::operator""sv;
    
    std::vector<std::string> v;

    v.push_back(utils::to_string(num_io));
    v.push_back(utils::to_string(read_count));
    v.push_back(utils::to_string(write_count));
    
    v.push_back(utils::to_string(iops));
    v.push_back(utils::to_string(read_iops));
    v.push_back(utils::to_string(write_iops));
    
    v.push_back(utils::to_string(start_time));
    v.push_back(utils::to_string(end_time));
    v.push_back(ts_unit);
    v.push_back(utils::to_string(duration));
    v.push_back(utils::to_string(duration_in_sec));

    v.push_back(utils::to_string(write_ratio));
    v.push_back(utils::to_string(read_ratio));
    v.push_back(utils::to_string(write_over_read_ratio));
    v.push_back(utils::to_string(read_over_write_ratio));
    
    for (const auto& col: raw_characteristic::stats_col) {
        tsl::ordered_map<std::string, double> m;
        auto col_sv = std::string_view{col};
        if (col_sv == "raw_bandwidth"sv) {
            m = raw_bandwidth.to_map();
        } else if (col_sv == "raw_read_bandwidth"sv) {
            m = raw_read_bandwidth.to_map();
        } else if (col_sv == "raw_write_bandwidth"sv) {
            m = raw_write_bandwidth.to_map();
        } else if (col_sv == "size"sv) {
            m = size.to_map();
        } else if (col_sv == "read_size"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_size"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_size_over_read_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_size_over_write_size_ratio"sv) {
            m = read_size.to_map();
        } else if (col_sv == "offset"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_offset"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_offset"sv) {
            m = read_size.to_map();
        } else if (col_sv == "iat"sv) {
            m = read_size.to_map();
        } else if (col_sv == "read_iat"sv) {
            m = read_size.to_map();
        } else if (col_sv == "write_iat"sv) {
            m = read_size.to_map();
        }
        

        for (const auto& p: m) {
            v.push_back(utils::to_string(p.second));
        }
    }

    return v;
}

ReplayedCharacteristic ReplayedCharacteristic::from(const trace::ReplayedTrace& trace, bool parallel) {
}

ReplayedCharacteristic ReplayedCharacteristic::from(const TraceCombiner<trace::ReplayedTrace>& trace, bool parallel) {
}

std::vector<std::string> ReplayedCharacteristic::header() {

}

std::vector<std::string> ReplayedCharacteristic::values() {

}
} // namespace trace_utils
