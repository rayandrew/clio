#include <trace-utils/characteristic.hpp>

#include <string_view>

#include <oneapi/tbb.h>

#include <trace-utils/utils.hpp>
#include <iostream>
namespace trace_utils
{
    constexpr const char *non_stats_col[] = {
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

    namespace raw_characteristic
    {
        constexpr const char *stats_col[] = {

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

    namespace replayed_characteristic
    {
        constexpr const char *stats_col[] = {
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
            "emp_write_iat"};
    } // namespace replayed_characteristic

    template <typename T>
    static RawCharacteristic calc_raw_characteristic(const T &trace, bool parallel)
    {
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

        trace.raw_stream([&](const auto &item)
                         {
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
        last_time = item.timestamp; });

        if (num_io > 0)
        {
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

        if (parallel)
        {
            oneapi::tbb::parallel_invoke(
                [&]()
                { characteristic.size = Statistic::from(size, true); },
                [&]()
                { characteristic.read_size = Statistic::from(read_size, true); },
                [&]()
                { characteristic.write_size = Statistic::from(write_size, true); },
                [&]()
                { characteristic.offset = Statistic::from(offset, true); },
                [&]()
                { characteristic.read_offset = Statistic::from(read_offset, true); },
                [&]()
                { characteristic.write_offset = Statistic::from(write_offset, true); },
                [&]()
                { characteristic.iat = Statistic::from(iat, true); },
                [&]()
                { characteristic.read_iat = Statistic::from(read_iat, true); },
                [&]()
                { characteristic.write_iat = Statistic::from(write_iat, true); });
        }
        else
        {
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

        if (duration_in_sec > 1e-6)
        {
            characteristic.iops = characteristic.num_io / duration_in_sec;
            characteristic.read_iops = characteristic.read_count / duration_in_sec;
            characteristic.write_iops = characteristic.write_count / duration_in_sec;
            characteristic.raw_bandwidth = characteristic.size / duration_in_sec;             // bytes/sec
            characteristic.raw_read_bandwidth = characteristic.read_size / duration_in_sec;   // bytes/sec
            characteristic.raw_write_bandwidth = characteristic.write_size / duration_in_sec; // bytes/sec
        }

        // Div by characteristic, div by 0 handled by operator override '/'
        characteristic.read_size_ratio = characteristic.read_size / characteristic.size;
        characteristic.write_size_ratio = characteristic.write_size / characteristic.size;
        characteristic.write_size_over_read_size_ratio = characteristic.write_size / characteristic.read_size;
        characteristic.read_size_over_write_size_ratio = characteristic.read_size / characteristic.write_size;

        if (read_count > 1e-6)
        {
            characteristic.read_ratio = static_cast<double>(characteristic.read_count) / characteristic.num_io;
            characteristic.write_over_read_ratio = static_cast<double>(characteristic.write_count) / characteristic.read_count;
        }
        if (write_count > 1e-6)
        {
            characteristic.write_ratio = static_cast<double>(characteristic.write_count) / characteristic.num_io;
            characteristic.read_over_write_ratio = static_cast<double>(characteristic.read_count) / characteristic.write_count;
        }

        return characteristic;
    }

    RawCharacteristic RawCharacteristic::from(const TraceCombiner<trace::ReplayerTrace> &trace, bool parallel)
    {
        return calc_raw_characteristic(trace, parallel);
    }

    RawCharacteristic RawCharacteristic::from(const trace::ReplayerTrace &trace, bool parallel)
    {
        return calc_raw_characteristic(trace, parallel);
    }

    std::vector<std::string> RawCharacteristic::header()
    {
        using std::operator""sv;

        std::vector<std::string> v(non_stats_col, non_stats_col + sizeof(non_stats_col) / sizeof(non_stats_col[0]));
        for (const auto &col : raw_characteristic::stats_col)
        {
            tsl::ordered_map<std::string, double> m;
            auto col_sv = std::string_view{col};
            if (col_sv == "raw_bandwidth"sv)
            {
                m = raw_bandwidth.to_map();
            }
            else if (col_sv == "raw_read_bandwidth"sv)
            {
                m = raw_read_bandwidth.to_map();
            }
            else if (col_sv == "raw_write_bandwidth"sv)
            {
                m = raw_write_bandwidth.to_map();
            }
            else if (col_sv == "size"sv)
            {
                m = size.to_map();
            }
            else if (col_sv == "read_size"sv)
            {
                m = read_size.to_map();
            }
            else if (col_sv == "write_size"sv)
            {
                m = write_size.to_map();
            }
            else if (col_sv == "write_size_ratio"sv)
            {
                m = write_size_ratio.to_map();
            }
            else if (col_sv == "read_size_ratio"sv)
            {
                m = read_size_ratio.to_map();
            }
            else if (col_sv == "write_size_over_read_size_ratio"sv)
            {
                m = write_size_over_read_size_ratio.to_map();
            }
            else if (col_sv == "read_size_over_write_size_ratio"sv)
            {
                m = read_size_over_write_size_ratio.to_map();
            }
            else if (col_sv == "offset"sv)
            {
                m = offset.to_map();
            }
            else if (col_sv == "read_offset"sv)
            {
                m = read_offset.to_map();
            }
            else if (col_sv == "write_offset"sv)
            {
                m = write_offset.to_map();
            }
            else if (col_sv == "iat"sv)
            {
                m = iat.to_map();
            }
            else if (col_sv == "read_iat"sv)
            {
                m = read_iat.to_map();
            }
            else if (col_sv == "write_iat"sv)
            {
                m = write_iat.to_map();
            }

            for (const auto &p : m)
            {
                v.push_back(fmt::format("{}_{}", col, p.first));
            }
        }

        return v;
    }

    std::vector<std::string> RawCharacteristic::values()
    {
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

        for (const auto &col : raw_characteristic::stats_col)
        {
            tsl::ordered_map<std::string, double> m;
            auto col_sv = std::string_view{col};
            if (col_sv == "raw_bandwidth"sv)
            {
                m = raw_bandwidth.to_map();
            }
            else if (col_sv == "raw_read_bandwidth"sv)
            {
                m = raw_read_bandwidth.to_map();
            }
            else if (col_sv == "raw_write_bandwidth"sv)
            {
                m = raw_write_bandwidth.to_map();
            }
            else if (col_sv == "size"sv)
            {
                m = size.to_map();
            }
            else if (col_sv == "read_size"sv)
            {
                m = read_size.to_map();
            }
            else if (col_sv == "write_size"sv)
            {
                m = write_size.to_map();
            }
            else if (col_sv == "write_size_ratio"sv)
            {
                m = write_size_ratio.to_map();
            }
            else if (col_sv == "read_size_ratio"sv)
            {
                m = read_size_ratio.to_map();
            }
            else if (col_sv == "write_size_over_read_size_ratio"sv)
            {
                m = write_size_over_read_size_ratio.to_map();
            }
            else if (col_sv == "read_size_over_write_size_ratio"sv)
            {
                m = read_size_over_write_size_ratio.to_map();
            }
            else if (col_sv == "offset"sv)
            {
                m = offset.to_map();
            }
            else if (col_sv == "read_offset"sv)
            {
                m = read_offset.to_map();
            }
            else if (col_sv == "write_offset"sv)
            {
                m = write_offset.to_map();
            }
            else if (col_sv == "iat"sv)
            {
                m = iat.to_map();
            }
            else if (col_sv == "read_iat"sv)
            {
                m = read_iat.to_map();
            }
            else if (col_sv == "write_iat"sv)
            {
                m = write_iat.to_map();
            }

            for (const auto &p : m)
            {
                v.push_back(utils::to_string(p.second));
            }
        }

        return v;
    }

    template <typename T>
    static ReplayedCharacteristic calc_replayed_characteristic(const T &trace, bool parallel)
    {
        // Replayed is not guaranteed to be sorted.
        // The raw streaming sorts by timestamp (in read csv)
        // This one need to sort by timestamp_submit, for emp iat
        auto raw_char = calc_raw_characteristic(trace, parallel);
        ReplayedCharacteristic replayed_char;
        static_cast<RawCharacteristic &>(replayed_char) = raw_char;

        std::vector<trace_utils::trace::replayed::Entry> entries;

        trace.raw_stream([&](const auto &item)
                         { entries.push_back(item); });

        if (parallel)
        {
            oneapi::tbb::parallel_sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
                                       { return a.timestamp_submit < b.timestamp_submit; });
        }
        else
        {
            std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
                      { return a.timestamp_submit < b.timestamp_submit; });
        }

        std::vector<double> latencies;
        std::vector<double> read_latencies;
        std::vector<double> write_latencies;

        std::vector<double> throughputs;
        std::vector<double> read_throughputs;
        std::vector<double> write_throughputs;

        std::vector<double> emp_bandwidths;
        std::vector<double> emp_read_bandwidths;
        std::vector<double> emp_write_bandwidths;

        std::vector<double> emp_iats;
        std::vector<double> emp_read_iats;
        std::vector<double> emp_write_iats;

        double last_time = 0;
        double last_read_time = 0;
        double last_write_time = 0;

        for (const auto &item : entries)
        {
            double latency = item.latency;
            latencies.push_back(latency);
            if (item.read)
            {
                read_latencies.push_back(latency);
            }
            else
            {
                write_latencies.push_back(latency);
            }

            double throughput = item.size / (item.latency);
            throughputs.push_back(throughput);
            if (item.read)
            {
                read_throughputs.push_back(throughput);
            }
            else
            {
                write_throughputs.push_back(throughput);
            }

            double emp_bandwidth = item.size_after_replay / latency;
            emp_bandwidths.push_back(emp_bandwidth);
            if (item.read)
            {
                emp_read_bandwidths.push_back(emp_bandwidth);
            }
            else
            {
                emp_write_bandwidths.push_back(emp_bandwidth);
            }

            double emp_iat = item.timestamp_submit - last_time;
            emp_iats.push_back(emp_iat);
            if (item.read)
            {
                double emp_iat_read = item.timestamp_submit - last_read_time;
                emp_read_iats.push_back(emp_iat_read);
                last_read_time = item.timestamp_submit;
            }
            else
            {
                double emp_iat_write = item.timestamp_submit - last_write_time;
                emp_write_iats.push_back(emp_iat_write);
                last_write_time = item.timestamp_submit;
            }
            last_time = item.timestamp_submit;
        }

        if (parallel)
        {
            oneapi::tbb::parallel_invoke(
                [&]()
                { replayed_char.latency = Statistic::from(latencies, true); },
                [&]()
                { replayed_char.read_latency = Statistic::from(read_latencies, true); },
                [&]()
                { replayed_char.write_latency = Statistic::from(write_latencies, true); },
                [&]()
                { replayed_char.throughput = Statistic::from(throughputs, true); },
                [&]()
                { replayed_char.read_throughput = Statistic::from(read_throughputs, true); },
                [&]()
                { replayed_char.write_throughput = Statistic::from(write_throughputs, true); },
                [&]()
                { replayed_char.emp_bandwidth = Statistic::from(emp_bandwidths, true); },
                [&]()
                { replayed_char.emp_read_bandwidth = Statistic::from(emp_read_bandwidths, true); },
                [&]()
                { replayed_char.emp_write_bandwidth = Statistic::from(emp_write_bandwidths, true); },
                [&]()
                { replayed_char.emp_iat = Statistic::from(emp_iats, true); },
                [&]()
                { replayed_char.emp_read_iat = Statistic::from(emp_read_iats, true); },
                [&]()
                { replayed_char.emp_write_iat = Statistic::from(emp_write_iats, true); });
        }
        else
        {
            replayed_char.latency = Statistic::from(latencies, false);
            replayed_char.read_latency = Statistic::from(read_latencies, false);
            replayed_char.write_latency = Statistic::from(write_latencies, false);
            replayed_char.throughput = Statistic::from(throughputs, false);
            replayed_char.read_throughput = Statistic::from(read_throughputs, false);
            replayed_char.write_throughput = Statistic::from(write_throughputs, false);
            replayed_char.emp_bandwidth = Statistic::from(emp_bandwidths, false);
            replayed_char.emp_read_bandwidth = Statistic::from(emp_read_bandwidths, false);
            replayed_char.emp_write_bandwidth = Statistic::from(emp_write_bandwidths, false);
            replayed_char.emp_iat = Statistic::from(emp_iats, false);
            replayed_char.emp_read_iat = Statistic::from(emp_read_iats, false);
            replayed_char.emp_write_iat = Statistic::from(emp_write_iats, false);
        }

        return replayed_char;
    }
    // Additional columns
    //     "latency",
    //     "read_latency",
    //     "write_latency",

    //     "throughput",
    //     "read_throughput",
    //     "write_throughput",

    //     "emp_bandwidth",
    //     "emp_read_bandwidth",
    //     "emp_write_bandwidth",

    //     "emp_iat",
    //     "emp_read_iat",
    //     "emp_write_iat"

    ReplayedCharacteristic ReplayedCharacteristic::from(const trace::ReplayedTrace &trace, bool parallel)
    {
        return calc_replayed_characteristic(trace, parallel);
    }

    ReplayedCharacteristic ReplayedCharacteristic::from(const TraceCombiner<trace::ReplayedTrace> &trace, bool parallel)
    {
        return calc_replayed_characteristic(trace, parallel);
    }

    std::vector<std::string> new_fields = {
        "latency", "read_latency", "write_latency",
        "throughput", "read_throughput", "write_throughput",
        "emp_bandwidth", "emp_read_bandwidth", "emp_write_bandwidth",
        "emp_iat", "emp_read_iat", "emp_write_iat"};

    std::vector<std::string> ReplayedCharacteristic::header()
    {
        auto v = RawCharacteristic::header();
        // Add new fields to header
        using std::operator""sv;

        for (const auto &field : new_fields)
        {
            // Add detailed headers for the new fields
            tsl::ordered_map<std::string, double> m;

            if (field == "latency"sv)
            {
                m = latency.to_map();
            }
            else if (field == "read_latency"sv)
            {
                m = read_latency.to_map();
            }
            else if (field == "write_latency"sv)
            {
                m = write_latency.to_map();
            }
            else if (field == "throughput"sv)
            {
                m = throughput.to_map();
            }
            else if (field == "read_throughput"sv)
            {
                m = read_throughput.to_map();
            }
            else if (field == "write_throughput"sv)
            {
                m = write_throughput.to_map();
            }
            else if (field == "emp_bandwidth"sv)
            {
                m = emp_bandwidth.to_map();
            }
            else if (field == "emp_read_bandwidth"sv)
            {
                m = emp_read_bandwidth.to_map();
            }
            else if (field == "emp_write_bandwidth"sv)
            {
                m = emp_write_bandwidth.to_map();
            }
            else if (field == "emp_iat"sv)
            {
                m = emp_iat.to_map();
            }
            else if (field == "emp_read_iat"sv)
            {
                m = emp_read_iat.to_map();
            }
            else if (field == "emp_write_iat"sv)
            {
                m = emp_write_iat.to_map();
            }

            for (const auto &p : m)
            {
                v.push_back(fmt::format("{}_{}", field, p.first));
            }
        }

        return v;
    }

    std::vector<std::string> ReplayedCharacteristic::values()
    {
        auto v = RawCharacteristic::values();

        using std::operator""sv;

        std::vector<std::string> new_fields = {
            "latency", "read_latency", "write_latency",
            "throughput", "read_throughput", "write_throughput",
            "emp_bandwidth", "emp_read_bandwidth", "emp_write_bandwidth",
            "emp_iat", "emp_read_iat", "emp_write_iat"};

        for (const auto &field : new_fields)
        {
            tsl::ordered_map<std::string, double> m;

            if (field == "latency"sv)
            {
                m = latency.to_map();
            }
            else if (field == "read_latency"sv)
            {
                m = read_latency.to_map();
            }
            else if (field == "write_latency"sv)
            {
                m = write_latency.to_map();
            }
            else if (field == "throughput"sv)
            {
                m = throughput.to_map();
            }
            else if (field == "read_throughput"sv)
            {
                m = read_throughput.to_map();
            }
            else if (field == "write_throughput"sv)
            {
                m = write_throughput.to_map();
            }
            else if (field == "emp_bandwidth"sv)
            {
                m = emp_bandwidth.to_map();
            }
            else if (field == "emp_read_bandwidth"sv)
            {
                m = emp_read_bandwidth.to_map();
            }
            else if (field == "emp_write_bandwidth"sv)
            {
                m = emp_write_bandwidth.to_map();
            }
            else if (field == "emp_iat"sv)
            {
                m = emp_iat.to_map();
            }
            else if (field == "emp_read_iat"sv)
            {
                m = emp_read_iat.to_map();
            }
            else if (field == "emp_write_iat"sv)
            {
                m = emp_write_iat.to_map();
            }

            for (const auto &p : m)
            {
                v.push_back(utils::to_string(p.second));
            }
        }

        return v;
    }
} // namespace trace_utils
