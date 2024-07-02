#ifndef __TRACE_UTILS_UTILS_HPP__
#define __TRACE_UTILS_UTILS_HPP__

#include <trace-utils/internal/filesystem.hpp>

namespace trace_utils {
void read_csv(const fs::path& path);
} // namespace trace_utils

#endif
