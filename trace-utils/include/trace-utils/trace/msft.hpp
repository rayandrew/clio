#ifndef __TRACE_UTILS_TRACE_MSFT_HPP__
#define __TRACE_UTILS_TRACE_MSFT_HPP__

#include <fmt/core.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace {
namespace msft {
struct Entry : public trace::IEntry {
    float timestamp = 0.0;
    std::string hostname = "";
    unsigned long disk_id = 0;
    std::string type = "";
    unsigned long offset = 0;
    unsigned long size = 0;
    float response_time = 0.0;

    virtual trace::Entry convert() const override;
    virtual std::vector<std::string> to_vec() const override;
};
} // namespace tencent
    
class MsftTrace : public Trace<msft::Entry> {
public:
    using Trace<msft::Entry>::Trace;
    using Trace<msft::Entry>::raw_stream;
    using Trace<msft::Entry>::stream;
    // using Trace<msft::Entry>::stream_filter;
    using Trace<msft::Entry>::get_raw_vector;
    using Trace<msft::Entry>::get_vector;
    using Trace<msft::Entry>::operator();
    virtual void raw_stream(const fs::path& path, RawReadFn&& read_fn) const override;
    virtual void raw_stream_column(const fs::path& path,
                                   unsigned int column,
                                   RawReadColumnFn&& read_fn) const override;
};
} // namespace trace_utils::trace

namespace fmt {
template <> class formatter<trace_utils::trace::MsftTrace::Entry> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::trace::MsftTrace::Entry const& entry, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{timestamp={}, hostname={}, disk_id={}, type={}, size={}, read={}, response_time={}}}", entry.timestamp, entry.hostname, entry.disk_id, entry.type, entry.offset, entry.size, entry.type, entry.response_time);
  }
};
} // namespace fmt


#endif
