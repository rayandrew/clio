#ifndef __TRACE_UTILS_EXCEPTION_HPP__
#define __TRACE_UTILS_EXCEPTION_HPP__

#include <exception>
#include <string>
#include <cpptrace/cpptrace.hpp>

namespace trace_utils {
class Exception : public cpptrace::exception_with_message {

    // std::string m_error;

public:

    template<typename ... Args>
    Exception(Args&&... args)
              // cpptrace::raw_trace&& trace = cpptrace::detail::get_raw_trace_and_absorb())
        : exception_with_message(std::forward<Args>(args)...) {}

    // virtual const char* what() const noexcept override {
    //     return m_error.c_str();
    // }
};
} // namespace trace_utils

#endif
