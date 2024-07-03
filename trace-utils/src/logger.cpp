#include <trace-utils/logger.hpp>

#include <spdlog/sinks/stdout_sinks.h>

const std::string _logger_name = "trace-utils";

namespace trace_utils::logger::impl {
LoggerImpl::LoggerImpl(): spdlog::logger(logger_name()) { }

LoggerImpl::LoggerImpl(spdlog::sinks_init_list sinks): spdlog::logger(logger_name(), sinks.begin(), sinks.end()) {}

LoggerImpl::LoggerImpl(spdlog::sink_ptr single_sink): spdlog::logger(logger_name(), single_sink) {}

std::string LoggerImpl::logger_name() {
    return _logger_name;
}
} // namespace trace_utils::logger::impl
