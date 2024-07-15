set(app-src-files
  main.cpp
  app.cpp
  tencent/tencent.cpp
  tencent/count-volume-map.cpp
  tencent/count-volume-reduce.cpp
  tencent/pick-volume.cpp
  tencent/split.cpp
  stats/stats.cpp
  stats/calculate.cpp
  stats/calculate-raw-trace.cpp
)

set(TRACE_UTILS_LINK_LIBS
  trace-utils::lib
  archive::archive
  CLI11::CLI11
  fmt::fmt
  spdlog::spdlog
  function2::function2
  Glob
  natsort::natsort
  indicators::indicators
  mp-units::mp-units
  TBB::tbb
  csv2::csv2
  scope_guard::scope_guard
  magic_enum::magic_enum
  cpptrace::cpptrace)

add_executable(trace-utils ${app-src-files})
target_link_libraries(trace-utils ${TRACE_UTILS_LINK_LIBS})

target_include_directories(trace-utils BEFORE PRIVATE
  "${libarchive_SOURCE_DIR}/libarchive")

add_executable(trace-utils-stack-trace stacktrace.cpp)
target_link_libraries(trace-utils-stack-trace
  cpptrace::cpptrace
  ${LIBUNWIND_LDFLAGS})
target_include_directories(trace-utils-stack-trace BEFORE PRIVATE
  "${libarchive_SOURCE_DIR}/libarchive")

install(TARGETS trace-utils
  EXPORT trace-utils-targets
  ARCHIVE DESTINATION "bin"
  LIBRARY DESTINATION "bin"
  RUNTIME DESTINATION "bin")

install(
  FILES $<TARGET_RUNTIME_DLLS:trace-utils>
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS trace-utils-stack-trace
  EXPORT trace-utils-stack-trace-targets
  ARCHIVE DESTINATION "bin"
  LIBRARY DESTINATION "bin"
  RUNTIME DESTINATION "bin")

install(
  FILES $<TARGET_RUNTIME_DLLS:trace-utils-stack-trace>
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
