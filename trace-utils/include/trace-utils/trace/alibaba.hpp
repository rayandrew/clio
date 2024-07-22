#ifndef __TRACE_UTILS_TRACE_ALIBABA_HPP__
#define __TRACE_UTILS_TRACE_ALIBABA_HPP__

#include <string>

#include <fmt/core.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace
{
    namespace alibaba
    {
        struct Entry : public trace::IEntry
        {
            double timestamp = 0.0;
            unsigned long offset = 0;
            unsigned long length = 0;
            char opcode;
            int device_id = 0;

            virtual trace::Entry convert() const override;
            virtual std::vector<std::string> to_vec() const override;
        };
    } // namespace alibaba

    class AlibabaTrace : public Trace<alibaba::Entry>
    {
    public:
        using Trace<alibaba::Entry>::Trace;
        using Trace<alibaba::Entry>::raw_stream;
        using Trace<alibaba::Entry>::raw_stream_column;
        using Trace<alibaba::Entry>::stream;
        using Trace<alibaba::Entry>::get_raw_vector;
        using Trace<alibaba::Entry>::get_vector;
        using Trace<alibaba::Entry>::operator();

        enum class Column : unsigned int
        {
            DEVICE_ID = 1,
            OPCODE = 2,
            OFFSET = 3,
            LENGTH = 4,
            TIMESTAMP = 5,
            //         device_id uint32	0	ID of the virtual disk
            // opcode	char	R	Either of 'R' or 'W', indicating this operation is read or write
            // offset	uint64	126703644672	Offset of this operation, in bytes
            // length	uint32	4096	Length of this operation, in bytes
            // timestamp	uint64	1577808000000626	Timestamp of this operation received by server, in microseconds
        };

        virtual void raw_stream(const fs::path &path, RawReadFn &&read_fn) const override;

        virtual void raw_stream_column(const fs::path &path,
                                       unsigned int column,
                                       RawReadColumnFn &&read_fn) const override;
    };
} // namespace trace_utils::trace

namespace fmt
{
    template <>
    class formatter<trace_utils::trace::AlibabaTrace::Entry>
    {
    public:
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        template <typename FmtContext>
        constexpr auto format(trace_utils::trace::AlibabaTrace::Entry const &entry, FmtContext &ctx) const -> format_context::iterator
        {
            return format_to(ctx.out(), "{{timestamp={}, offset={}, size={}, read={}, volume={}}}", entry.timestamp, entry.offset, entry.length, entry.opcode, entry.device_id);
        }
    };
} // namespace fmt

#endif
