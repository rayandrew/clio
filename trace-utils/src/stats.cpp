#include <trace-utils/stats.hpp>

#include <fmt/core.h>
#include <oneapi/tbb.h>

#include <trace-utils/logger.hpp>

template<typename T>
struct SumReduce {
    const std::vector<T>& it;
    T value;
   
    SumReduce(const std::vector<T>& it): it{it}, value{0} {}
    SumReduce(SumReduce<T>& body, tbb::split): it{body.it}, value{0} {}
  
    void operator()(const oneapi::tbb::blocked_range<size_t>& r) {
        // value += std::accumulate(it.begin() + r.begin(), it.begin() + r.end(), 0);
        
        // for (std::size_t i = r.begin(); i != r.end(); ++i) {
        //     auto d = it[i];
        //     value += std::accumulate(d.cbegin(), d.cend(), 0);
        // }
        // value += std::accumulate(it + r.begin(), it + r.end(), 0);
    }
    
    void join(SumReduce<T>& rhs) { value += rhs.value; }
};

static constexpr double quantiles[] = {
    0.1, 0.15,
    0.2, 0.25,
    0.3, 0.35,
    0.4, 0.45,
    0.5, 0.55,
    0.6, 0.65,
    0.7, 0.75,
    0.8, 0.85,
    0.9, 0.95,
    0.99, 0.999,
    0.9999
};

namespace trace_utils {
Statistic Statistic::from(const std::vector<double>& data, bool parallel) {
    Statistic stats;

    stats.count = data.size();

    auto sorted_data = data;
    if (parallel) {
        oneapi::tbb::parallel_sort(sorted_data.begin(), sorted_data.end(),
                           [=](const auto &a, const auto& b)
                           {
                               return a < b;
                           });
    } else {
        std::sort(sorted_data.begin(), sorted_data.end());
    }

    if (parallel) {
        SumReduce<double> sum(data);
        oneapi::tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, data.size()), sum);
        stats.avg = sum.value / stats.count;
    } else {
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        stats.avg = sum / stats.count;
    }

    auto variance_func = [](double mean, double size) {
        return [&](double accumulator, const double& val) {
            return accumulator + ((val - mean) * (val - mean) / (size - 1));
        };
    };

    stats.variance = std::accumulate(data.begin(), data.end(), 0.0, variance_func(stats.avg, stats.count));
    stats.std_dev = std::sqrt(stats.variance);

    stats.min = sorted_data.front();
    stats.max = sorted_data.back();

    if (parallel) {
        using Map = oneapi::tbb::concurrent_hash_map<
            double,
            std::size_t
        >;

        Map map;
        oneapi::tbb::parallel_for_each(data.cbegin(), data.cend(), [&](const auto& d) {
            Map::accessor accessor;
            map.insert(accessor, d);
            accessor->second += 1;
        });

        auto mode = std::max_element(map.begin(), map.end(),
                                     [] (const auto &p1, const auto &p2) {
                                         return p1.second < p2.second;
                                     });
        stats.mode = mode->first;
    } else {
        std::unordered_map<double, std::size_t> map;
        for (const auto& d: sorted_data) {
            if (map.find(d) == map.end()){
                map[d] = 0;
            } else {
                map[d] += 1;
            }
        }

        auto mode = std::max_element(map.begin(), map.end(),
                                     [] (const auto &p1, const auto &p2) {
                                         return p1.second < p2.second;
                                     });
        stats.mode = mode->first;
    }

    if (parallel) {
        oneapi::tbb::parallel_for_each(std::begin(quantiles), std::end(quantiles), [&](const auto& q) {
            stats.percentiles[fmt::format("p{}", q * 100.0)] = stats::quantile(sorted_data, q);
        });      
    } else {
        for (auto q: quantiles) {
            stats.percentiles[fmt::format("p{}", q * 100.0)] = stats::quantile(sorted_data, q);
        }
    }

    stats.median = stats.percentiles["p50"];

    stats.percentiles["p100"] = sorted_data.back();

    return stats;
}

std::unordered_map<std::string, double> Statistic::to_map() {
    auto map = std::unordered_map<std::string, double>();
    map["avg"] = avg;
    map["max"] = max;
    map["min"] = min;
    map["mode"] = mode;
    map["count"] = count;
    map["median"] = median;
    map["variance"] = variance;
    map["std_dev"] = std_dev;
    map.insert(percentiles.cbegin(), percentiles.cend());
    return map;
}
} // namespace trace_utils
