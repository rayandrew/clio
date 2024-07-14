#include <trace-utils/characteristic.hpp>

#include <oneapi/tbb.h>

namespace trace_utils {
RawCharacteristic RawCharacteristic::from(const trace::ReplayerTrace& trace, bool parallel) {
    std::vector<double> offset;
    
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
            read_size.push_back(item.size);
            last_read_time = item.timestamp;
        } else {
            write_count += 1;
            write_iat.push_back(item.timestamp - last_write_time);
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
            [&]() { characteristic.iat = Statistic::from(iat, true); },
            [&]() { characteristic.read_iat = Statistic::from(read_iat, true); },
            [&]() { characteristic.write_iat = Statistic::from(write_iat, true); }
        );
    } else {
        characteristic.size = Statistic::from(size, false);
        characteristic.read_size = Statistic::from(read_size, false);
        characteristic.write_size = Statistic::from(write_size, false);
        characteristic.offset = Statistic::from(offset, false);
        characteristic.iat = Statistic::from(iat, false);
        characteristic.read_iat = Statistic::from(read_iat, false);
        characteristic.write_iat = Statistic::from(write_iat, false);
    }

    characteristic.raw_bandwidth = characteristic.size.avg / duration_in_sec; // bytes/sec
    characteristic.raw_read_bandwidth = characteristic.read_size.avg / duration_in_sec; // bytes/sec
    characteristic.raw_write_bandwidth = characteristic.write_size.avg / duration_in_sec; // bytes/sec

    return characteristic;
}

ReplayedCharacteristic ReplayedCharacteristic::from(const trace::ReplayedTrace& trace) {
    ReplayedCharacteristic characteristic;

    trace.raw_stream([&](const auto& item) {
        auto buff = item.to_vec();
        // writer.write_row(buff);
    });


    return characteristic;
}
} // namespace trace_utils
