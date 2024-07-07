#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <scope_guard.hpp>

#define defer DEFER

namespace trace_utils {
namespace utils {
std::string random_string(std::size_t length);
} // namespace utils
} // namespace trace_utils

#endif
