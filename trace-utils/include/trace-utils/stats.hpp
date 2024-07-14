#ifndef __TRACE_UTILS_STATS_HPP__
#define __TRACE_UTILS_STATS_HPP__

#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <trace-utils/exception.hpp>

namespace trace_utils {
namespace stats {

// CC BY-SA 4.0
// https://stackoverflow.com/a/37708864/2418586
template<typename T>
inline double lerp(T v0, T v1, T t)
{
    return (1 - t)*v0 + t*v1;
}

// CC BY-SA 4.0
// https://stackoverflow.com/a/37708864/2418586
template<typename T>
T quantile(const std::vector<T>& data, T prob, bool check_sorted = false) {
    if (check_sorted) {
        if (std::is_sorted(data.begin(), data.end())) {
            throw Exception("Data needs to be sorted!");
        }
    }
    
    T poi = lerp<T>(-0.5, data.size() - 0.5, prob);

    size_t left = std::max(std::int64_t(std::floor(poi)), std::int64_t(0));
    size_t right = std::min(std::int64_t(std::ceil(poi)), std::int64_t(data.size() - 1));

    T data_left = data.at(left);
    T data_right = data.at(right);

    T quantile = lerp<T>(data_left, data_right, poi - left);

    return quantile;
}
    
// CC BY-SA 4.0
// https://stackoverflow.com/a/37708864/2418586
template<typename T>
inline std::vector<T> quantile(const std::vector<T>& data_, const std::vector<T>& probs, bool sort_data = false)
{
    if (data_.empty())
    {
        return std::vector<T>();
    }

    if (1 == data_.size())
    {
        return std::vector<T>(1, data_[0]);
    }

    std::vector<T> data = data_;
    if (sort_data) {
        std::sort(data.begin(), data.end());
    }
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        quantiles.push_back(quantile<T>(data, probs[i], false));
    }

    return quantiles;
}
} // namespace stats

struct Statistic {
    double avg;
    double max;
    double min;
    double mode;
    double count;
    double median;
    double variance;
    double std_dev;

    // percentiles
    // double p10;
    // double p20;
    // double p25;
    // double p30;
    // double p35;
    // double p40;
    // double p45;
    // double p50;
    // double p55;
    // double p60;
    // double p65;
    // double p70;
    // double p75;
    // double p80;
    // double p85;
    // double p90;
    // double p95;
    // double p99;
    // double p999;
    // double p9999;
    // double p100;
    std::unordered_map<std::string, double> percentiles;
    
    static Statistic from(const std::vector<double>& data, bool parallel);

    std::unordered_map<std::string, double> to_map();
};
} // namespace trace_utils

#endif
