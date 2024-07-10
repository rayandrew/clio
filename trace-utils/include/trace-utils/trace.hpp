#ifndef __TRACE_UTILS_TRACE_HPP__
#define __TRACE_UTILS_TRACE_HPP__

#include <string>
#include <vector>
#include <type_traits>

#include <fmt/core.h>
#include <function2/function2.hpp>

#include <archive_entry.h>

#include <trace-utils/exception.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils {
namespace trace {
struct Entry;
    
class IEntry {
public:
    virtual Entry convert() const = 0;
    virtual std::vector<std::string> to_vec() const = 0;
};
    
struct Entry : public IEntry {
    double timestamp = 0.0;
    unsigned long disk_id = 0;
    unsigned long offset = 0;
    unsigned long size = 0;
    bool read = false;

    inline virtual Entry convert() const override {
        return *this;
    }

    inline std::vector<std::string> to_vec() const override {
        return {
            std::to_string(timestamp),
            std::to_string(disk_id),
            std::to_string(offset),
            std::to_string(size),
            read ? "1" : "0"
        };
    }
};
} // namespace trace

template <typename T, std::enable_if_t<std::is_base_of_v<trace::IEntry, T>, bool> = true>
class Trace {
public:
    using Entry = T;
    using RawReadFn = fu2::function<void(const Entry&) const>;
    using RawFilterFn = fu2::function<bool(const Entry&) const>;
    using ReadFn = fu2::function<void(const trace::Entry&) const>;
    using FilterFn = fu2::function<bool(const trace::Entry&) const>;
    using RawReadColumnFn = fu2::function<void(const std::string&) const>;
    // using ReadArchiveFn = fu2::function<void(archive_entry*) const>;
    
    /**
     * @brief Constructor
     */
    Trace() = default;

    /**
     * @brief Constructor
     */
    Trace(const fs::path& path): path(path) {
        if (!fs::exists(path)) { throw Exception("Path does not exist"); }
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


    // virtual void archive_stream(const fs::path &path,
    //                             ReadArchiveFn&& read_fn) const = 0;

    // inline void archive_stream(ArchiveReadFn&& read_fn) const {
    //     archive_stream(path, std::forward<ReadArchiveFn>(read_fn));
    // }
    
    virtual void raw_stream_column(const fs::path& path,
                                   unsigned int column,
                                   RawReadColumnFn&& read_fn) const = 0;

    inline void raw_stream_column(unsigned int column,
                                  RawReadColumnFn&& read_fn) const {
        raw_stream_column(path, column, std::forward<RawReadColumnFn>(read_fn));
    }
    
    virtual void raw_stream(const fs::path& path, RawReadFn&& read_fn) const = 0;
    
    inline void raw_stream(RawReadFn&& read_fn) const {
        raw_stream(path, std::forward<RawReadFn>(read_fn));
    }
    
    inline void stream(const fs::path& path, ReadFn&& read_fn) const {
        raw_stream(path, [read_fn](const auto& item) {
            read_fn(item.convert()); 
        });
    }
    
    inline void stream(ReadFn&& read_fn) const {
        stream(path, std::forward<ReadFn>(read_fn));
    }

    void stream(const fs::path& path, ReadFn&& read_fn, FilterFn&& filter_fn) const {
        stream(path, [&](const auto& item) {
            if (filter_fn(item)) {
                read_fn(item);
            }
        });
    }

    inline void stream(ReadFn&& read_fn, FilterFn&& filter_fn) const {
        stream(path, std::forward<ReadFn>(read_fn), std::forward<FilterFn>(filter_fn));
    }

    inline void raw_stream(const fs::path& path, RawReadFn&& read_fn, RawFilterFn&& filter_fn) const {
        raw_stream(path, [&](const auto& item) {
            if (filter_fn(item)) {
                read_fn(item);
            }
        });
    }

    inline void raw_stream(RawReadFn&& read_fn, RawFilterFn&& filter_fn) const {
        raw_stream(path, std::forward<RawReadFn>(read_fn), std::forward<RawFilterFn>(filter_fn));
    }

    std::vector<T> get_raw_vector(const fs::path& path, RawFilterFn&& filter_fn) const {
        std::vector<T> vec;
        raw_stream(path, [&](const auto& item) {
            if (filter_fn(item)) {
                vec.push_back(item);
            }
        });
        return vec;
    }

    std::vector<trace::Entry> get_vector(const fs::path& path, FilterFn&& filter_fn) const {
        std::vector<trace::Entry> vec;
        stream(path, [&](const auto& item) {
            if (filter_fn(item)) {
                vec.push_back(item);
            }
        });
        return vec;
    }

    inline std::vector<trace::Entry> operator()(ReadFn&& read_fn) const {
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
        return format_to(ctx.out(), "{{path={}}}}", trace.path);
  }
};

// template <typename T> class formatter<trace_utils::trace::Entry> {
// public:
//     constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
//     template <typename FmtContext>
//     constexpr auto format(trace_utils::trace::Entry const& entry, FmtContext& ctx) const -> format_context::iterator {
//         return format_to(ctx.out(), "{{timestamp={}, disk_id={}, offset={}, size={}, read={}}}", entry.timestamp, entry.disk_id, entry.offset, entry.size, entry.read);
//   }
// };
}

#endif
