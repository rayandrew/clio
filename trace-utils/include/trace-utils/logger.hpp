#ifndef __TRACE_UTILS_LOGGER_HPP__
#define __TRACE_UTILS_LOGGER_HPP__

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <trace-utils/internal/allocation.hpp>
#include <trace-utils/exception.hpp>

#ifdef WIN32
#  ifdef BUILD_APPLIB_SHARED
#    define APPLIB_EXPORT __declspec(dllexport)
#  else
#    define APPLIB_EXPORT
#  endif //BUILD_APPLIB_SHARED
#else
#  define APPLIB_EXPORT
#endif // WIN32


namespace trace_utils {
namespace logger {
namespace impl {
class APPLIB_EXPORT LoggerImpl : public spdlog::logger, public internal::StackObj {
  template <class LoggerImpl>
  template <typename... Args>
  friend void internal::StaticObj<LoggerImpl>::create(Args&&... args);
    
public:
    LoggerImpl();
    LoggerImpl(spdlog::sinks_init_list sinks);
    LoggerImpl(spdlog::sink_ptr single_sink);

    template <typename It>
    LoggerImpl(It begin, It end): spdlog::logger(logger_name(), begin, end) {}


    static std::string logger_name();
};
} // namespace impl

inline void create() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%L] %v"); 
    auto logger = std::make_shared<impl::LoggerImpl>(console_sink);
    spdlog::register_logger(logger);
}

inline void create(spdlog::sinks_init_list sinks) {
    auto logger = std::make_shared<impl::LoggerImpl>(sinks);
    spdlog::register_logger(logger);
}
    
inline void create(spdlog::sink_ptr single_sink) {
    auto logger = std::make_shared<impl::LoggerImpl>(single_sink);
    spdlog::register_logger(logger);
}

template <typename It>
inline void create(It begin, It end) {
   auto logger = std::make_shared<impl::LoggerImpl>(begin, end);
   spdlog::register_logger(logger);
}
} // namespace logger

inline std::shared_ptr<spdlog::logger> log() {
    auto logger = spdlog::get(logger::impl::LoggerImpl::logger_name());
    if (!logger) {
        logger::create(); // creating default logger
    }
    return logger;
}
} // namespace trace_utils

#endif
