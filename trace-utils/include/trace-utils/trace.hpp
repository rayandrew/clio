#ifndef __TRACE_UTILS_TRACE_HPP__
#define __TRACE_UTILS_TRACE_HPP__


#include <vector>
#include <type_traits>

#include <fmt/base.h>

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
}

template <typename T, std::enable_if_t<std::is_base_of_v<trace::IEntry, T>, bool> = true>
class Trace {
public:
    using Entry = T;

    /**
     * @brief Constructor
     */
    Trace() = default;
    
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

    inline size_t size() const { return data.size(); }

    virtual void read(const char* filename) = 0;

protected:
    std::vector<T> data;
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
    
// template <typename T> class formatter<trace_utils::Trace::Entry<T>> {
// public:
//     constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
//     template <typename FmtContext>
//     constexpr auto format(trace_utils::Trace::Entry<T> const& entry, FmtContext& ctx) const -> format_context::iterator {
//         return format_to(ctx.out(), "{{timestamp={}, disk_id={}, offset={}, size={}, read={}}}", entry.timestamp, entry.disk_id, entry.offset, entry.size, entry.read);
//   }
// };
}

#endif
