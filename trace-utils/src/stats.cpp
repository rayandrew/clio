#include <trace-utils/stats.hpp>

#include <fmt/core.h>
#include <oneapi/tbb.h>

template<typename T>
struct SumReduce {
    const T* it;
    T value;
   
    SumReduce(const T* it): it{it}, value{0} {}
    SumReduce(SumReduce<T>& body, tbb::split): it{body.it}, value{0} {}
  
    void operator()(const oneapi::tbb::blocked_range<size_t>& r) {
        value += std::accumulate(it + r.begin(), it + r.end(), 0);
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
TraceStatistic TraceStatistic::calculate(const std::vector<double>& data, bool parallel) {
    TraceStatistic stats;

    stats.count_ = data.size();

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
        SumReduce<double> sum(data.data());
        oneapi::tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, data.size()), sum);
        stats.avg_ = sum.value / stats.count_;
    } else {
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        stats.avg_ = sum / stats.count_;
    }

    auto variance_func = [](double mean, double size) {
        return [&](double accumulator, const double& val) {
            return accumulator + ((val - mean) * (val - mean) / (size - 1));
        };
    };

    stats.variance_ = std::accumulate(data.begin(), data.end(), 0.0, variance_func(stats.avg_, stats.count_));
    stats.std_dev_ = std::sqrt(stats.variance_);

    stats.min_ = sorted_data.front();
    stats.max_ = sorted_data.back();

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
        stats.mode_ = mode->first;
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
        stats.mode_ = mode->first;
    }

    if (parallel) {
        oneapi::tbb::parallel_for_each(std::begin(quantiles), std::end(quantiles), [&](const auto& q) {
            stats.percentiles_[fmt::format("p{}", q * 100.0)] = stats::quantile(sorted_data, q);
        });      
    } else {
        for (auto q: quantiles) {
            stats.percentiles_[fmt::format("p{}", q * 100.0)] = stats::quantile(sorted_data, q);
        }
    }

    stats.percentiles_["p100"] = sorted_data.back();

    return stats;
}

std::unordered_map<std::string, std::string> to_map() {
    auto map = std::unordered_map<std::string, std::string>();

    return map;
}
} // namespace trace_utils
