#ifndef __TRACE_UTILS_TRACE_MSFT_HPP__
#define __TRACE_UTILS_TRACE_MSFT_HPP__

#include <fmt/base.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace {
namespace msft {
struct Entry : public trace::Entry, public trace::IEntry {
    trace::Entry convert() {
        return *this;
    }
};
} // namespace tencent
    
class MsftTrace : public Trace<msft::Entry> {
public:
    virtual void read(const char* filename);
};
} // namespace trace_utils::trace

namespace fmt {
template <> class formatter<trace_utils::trace::MsftTrace> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::trace::MsftTrace const& trace, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{length={}}}}", trace.size());
  }
};
    
template <> class formatter<trace_utils::trace::MsftTrace::Entry> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::trace::MsftTrace::Entry const& entry, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{timestamp={}, disk_id={}, offset={}, size={}, read={}}}", entry.timestamp, entry.disk_id, entry.offset, entry.size, entry.read);
  }
};
} // namespace fmt


#endif
