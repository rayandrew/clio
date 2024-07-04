#ifndef __TRACE_UTILS_TRACE_HPP__
#define __TRACE_UTILS_TRACE_HPP__

#include <vector>
#include <type_traits>

#include <fmt/base.h>
#include <function2/function2.hpp>

#include <trace-utils/exception.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils {
namespace trace {
struct Entry {
    float timestamp = 0.0;
    unsigned long disk_id = 0;
    unsigned long offset = 0;
    unsigned long size = 0;
    bool read = false;
};

class IEntry {
public:
    virtual Entry convert() = 0;
    inline Entry operator()() { return convert(); };
};
} // namespace trace

template <typename T, std::enable_if_t<std::is_base_of_v<trace::IEntry, T>, bool> = true>
class Trace {
public:
    using Entry = T;
    using FilterFn = fu2::function<bool(const Entry&) const>;
    using ReadFn = fu2::function<void(const Entry&) const>;
    
    /**
     * @brief Constructor
     */
    Trace() = default;

    /**
     * @brief Constructor
     */
    Trace(const fs::path& path): path(path) {
        if (!fs::exists(path)) { throw new Exception("Path does not exist"); }
    }
    
    /**
     * @brief Move-constructor.
     */
    Trace(Trace&&) = default;

    /**
     * @brief Copy-constructor.
     */
    Trace(const Trace&) = default;

    /**
     * @brief Move-assignment operator.
     */
    Trace& operator=(Trace&&) = default;

    /**
     * @brief Copy-assignment operator.
     */
    Trace& operator=(const Trace&) = default;

    /**
     * @brief Destructor.
     */
    virtual ~Trace() = default;

    virtual void stream(const fs::path& path, ReadFn&& read_fn) const = 0;
    
    virtual void stream(ReadFn&& read_fn) const {
        stream(path, std::forward<ReadFn>(read_fn));
    }

    void stream_filter(const fs::path& path, ReadFn&& read_fn, FilterFn&& filter_fn) const {
        stream(path, [&](auto item) {
            if (filter_fn(item)) {
                read_fn(item);
            }
        });
    }

    inline void stream_filter(ReadFn&& read_fn, FilterFn&& filter_fn) const {
        stream(path, std::forward<ReadFn>(read_fn), std::forward<FilterFn>(filter_fn));
    }

    std::vector<T> get_raw_vector(const fs::path& path, FilterFn&& filter_fn) const {
        std::vector<T> vec;
        stream(path, [&](auto item) {
            if (filter_fn(item)) {
                vec.push_back(item);
            }
        });
        return vec;
    }
    
    inline std::vector<T> get_raw_vector(const std::string& path, FilterFn&& filter_fn) const {
        return get_raw_vector(fs::path{path}, std::forward<FilterFn>(filter_fn));
    }

    std::vector<trace::Entry> get_vector(const fs::path& path, FilterFn&& filter_fn) const {
        std::vector<trace::Entry> vec;
        stream(path, [&](auto item) {
            if (filter_fn(item)) {
                vec.push_back(item.convert());
            }
        });
        return vec;
    }

    inline std::vector<trace::Entry> get_vector(const std::string& path, FilterFn&& filter_fn) const {
        return get_vector(fs::path{path}, std::forward<FilterFn>(filter_fn));
    }

    inline std::vector<trace::Entry> operator()(const std::string& path, ReadFn&& read_fn) const {
        return get_vector(path, std::forward<ReadFn>(read_fn));
    }

    inline std::vector<trace::Entry> operator()(const fs::path& path, ReadFn&& read_fn) const {
        return get_vector(path, std::forward<ReadFn>(read_fn));
    }

protected:
    fs::path path;
};
} // namespace trace_utils

namespace fmt {
template <typename T> class formatter<trace_utils::Trace<T>> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::Trace<T> const& trace, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{length={}}}}", trace.size());
  }
};
}

#endif
