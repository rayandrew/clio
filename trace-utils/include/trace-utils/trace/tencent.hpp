#ifndef __TRACE_UTILS_TRACE_TENCENT_HPP__
#define __TRACE_UTILS_TRACE_TENCENT_HPP__

#include <string>

#include <fmt/core.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace {
namespace tencent {
struct Entry : public trace::IEntry {
    double timestamp = 0.0;
    unsigned long offset = 0;
    unsigned long size = 0;
    int read = 0;
    unsigned long volume = 0;

    virtual trace::Entry convert() const override;
    virtual std::vector<std::string> to_vec() const override;
};
} // namespace tencent
    
class TencentTrace : public Trace<tencent::Entry> {
public:
    using Trace<tencent::Entry>::Trace;
    using Trace<tencent::Entry>::raw_stream;
    using Trace<tencent::Entry>::raw_stream_column;
    using Trace<tencent::Entry>::stream;
    using Trace<tencent::Entry>::get_raw_vector;
    using Trace<tencent::Entry>::get_vector;
    using Trace<tencent::Entry>::operator();

    enum class Column : unsigned int {
        TIMESTAMP = 1,
        OFFSET = 2,
        SIZE = 3,
        READ = 4,
        VOLUME = 5,
    };
    
    virtual void raw_stream(const fs::path& path, RawReadFn&& read_fn) const override;

    virtual void raw_stream_column(const fs::path& path,
                                   unsigned int column,
                                   RawReadColumnFn&& read_fn) const override;
};
} // namespace trace_utils::trace

namespace fmt {
template <> class formatter<trace_utils::trace::TencentTrace::Entry> {
public:
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
   
    template <typename FmtContext>
    constexpr auto format(trace_utils::trace::TencentTrace::Entry const& entry, FmtContext& ctx) const -> format_context::iterator {
        return format_to(ctx.out(), "{{timestamp={}, offset={}, size={}, read={}, volume={}}}", entry.timestamp, entry.offset, entry.size, entry.read, entry.volume);
  }
};
} // namespace fmt


#endif
