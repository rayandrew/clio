#ifndef __TRACE_UTILS_EXCEPTION_HPP__
#define __TRACE_UTILS_EXCEPTION_HPP__

#include <exception>
#include <string>

namespace trace_utils {
class Exception : public std::exception {

    std::string m_error;

    public:

    template<typename ... Args>
    Exception(Args&&... args)
    : m_error(std::forward<Args>(args)...) {}

    virtual const char* what() const noexcept override {
        return m_error.c_str();
    }
};
} // namespace trace_utils

#endif
