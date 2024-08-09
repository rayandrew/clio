#ifndef __TRACE_UTILS_TRACE_TECTONIC_HPP__
#define __TRACE_UTILS_TRACE_TECTONIC_HPP__

#include <string>

#include <fmt/core.h>

#include <trace-utils/trace.hpp>
#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils::trace
{
    namespace tectonic
    {
        struct Entry : public trace::IEntry // MODIFY THIS ONE, THIS IS THE RAW DATA'S COLUMNS
        {
            double timestamp = 0.0;
            unsigned long offset = 0;
            unsigned long length = 0;
            int opcode; // PUT_OPS = [4, 3, 6]; GET_OPS = [2, 1, 5]
            int device_id = 0;

            virtual trace::Entry convert() const override;
            virtual std::vector<std::string> to_vec() const override;
        };
    } // namespace tectonic

    class TectonicTrace : public Trace<tectonic::Entry>
    {
    public:
        using Trace<tectonic::Entry>::Trace;
        using Trace<tectonic::Entry>::raw_stream;
        using Trace<tectonic::Entry>::raw_stream_column;
        using Trace<tectonic::Entry>::stream;
        using Trace<tectonic::Entry>::get_raw_vector;
        using Trace<tectonic::Entry>::get_vector;
        using Trace<tectonic::Entry>::operator();

        enum class Column : unsigned int  
        {   /*
            DEVICE_ID = 1,
            OPCODE = 2,
            OFFSET = 3,
            LENGTH = 4,
            TIMESTAMP = 5, */
            //         device_id uint32	0	ID of the virtual disk
            // opcode	char	R	Either of 'R' or 'W', indicating this operation is read or write
            // offset	uint64	126703644672	Offset of this operation, in bytes
            // length	uint32	4096	Length of this operation, in bytes
            // timestamp	uint64	1577808000000626	Timestamp of this operation received by server, in microseconds
        
            DEVICE_ID = 1,
            OPCODE = 5,
            OFFSET = 2,
            LENGTH = 3,
            TIMESTAMP = 4,
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
    class formatter<trace_utils::trace::TectonicTrace::Entry>
    {
    public:
        constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

        template <typename FmtContext>
        constexpr auto format(trace_utils::trace::TectonicTrace::Entry const &entry, FmtContext &ctx) const -> format_context::iterator
        {
            return format_to(ctx.out(), "{{timestamp={}, offset={}, size={}, read={}, volume={}}}", entry.timestamp, entry.offset, entry.length, entry.opcode, entry.device_id);
        }
    };
} // namespace fmt

#endif