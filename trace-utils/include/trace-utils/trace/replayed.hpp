#ifndef __TRACE_UTILS_TRACE_REPLAYED_HPP__
#define __TRACE_UTILS_TRACE_REPLAYED_HPP__

#include <string>

#include <fmt/core.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace {
namespace replayed {
struct Entry : public trace::Entry {
    double latency = 0.0;
    double timestamp_submit = 0.0;
    double size_after_replay = 0.0;
    
    std::vector<std::string> to_vec() const override;
};
} // namespace replayed
    
class ReplayedTrace : public Trace<replayed::Entry> {
public:
    using Trace<replayed::Entry>::Trace;
    using Trace<replayed::Entry>::raw_stream;
    using Trace<replayed::Entry>::raw_stream_column;
    using Trace<replayed::Entry>::stream;
    using Trace<replayed::Entry>::get_raw_vector;
    using Trace<replayed::Entry>::get_vector;
    using Trace<replayed::Entry>::operator();

    enum class Column : unsigned int {
        TIMESTAMP = 1,
        LATENCY = 2,
        READ = 3,
        SIZE = 4,
        OFFSET = 5,
        TIMESTAMP_SUBMIT = 6,
        SIZE_AFTER_REPLAY = 7
    };
    
    virtual void raw_stream(const fs::path& path, RawReadFn&& read_fn) const override;

    virtual void raw_stream_column(const fs::path& path,
                                   unsigned int column,
                                   RawReadColumnFn&& read_fn) const override;
};
} // namespace trace_utils::trace

namespace fmt {
template <> class formatter<trace_utils::trace::ReplayedTrace::Entry> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::trace::ReplayedTrace::Entry const& entry, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{timestamp={}, latency={}, read={}, size={}, offset={}, timestamp_submit={}, size_after_replay={}}}", entry.timestamp, entry.latency, entry.read, entry.size, entry.offset, entry.timestamp_submit, entry.size_after_replay);
  }
};
} // namespace fmt


#endif
