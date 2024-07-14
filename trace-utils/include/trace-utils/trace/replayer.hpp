#ifndef __TRACE_UTILS_TRACE_REPLAYER_HPP__
#define __TRACE_UTILS_TRACE_REPLAYER_HPP__

#include <string>

#include <fmt/core.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace {
namespace replayer {
struct Entry : public trace::Entry { };
} // namespace replayer
    
class ReplayerTrace : public Trace<replayer::Entry> {
public:
    using Trace<replayer::Entry>::Trace;
    using Trace<replayer::Entry>::raw_stream;
    using Trace<replayer::Entry>::raw_stream_column;
    using Trace<replayer::Entry>::stream;
    using Trace<replayer::Entry>::get_raw_vector;
    using Trace<replayer::Entry>::get_vector;
    using Trace<replayer::Entry>::operator();

    enum class Column : unsigned int {
        TIMESTAMP = 1,
        DISK_ID = 2,
        OFFSET = 3,
        SIZE = 4,
        READ = 5
    };
    
    virtual void raw_stream(const fs::path& path, RawReadFn&& read_fn) const override;

    virtual void raw_stream_column(const fs::path& path,
                                   unsigned int column,
                                   RawReadColumnFn&& read_fn) const override;
};
} // namespace trace_utils::trace

namespace fmt {
template <> class formatter<trace_utils::trace::ReplayerTrace::Entry> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::trace::ReplayerTrace::Entry const& entry, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{timestamp={}, disk_id={}, offset={}, size={}, read={}}}", entry.timestamp, entry.disk_id, entry.offset, entry.size, entry.read);
  }
};
} // namespace fmt


#endif
