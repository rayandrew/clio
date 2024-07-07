set(app-src-files
  main.cpp
  app.cpp
  tencent/tencent.cpp
  tencent/count-volume-map.cpp
  tencent/count-volume-reduce.cpp
  tencent/pick-volume.cpp
)

add_executable(trace-utils ${app-src-files})
target_link_libraries(trace-utils
  trace-utils::lib
  CLI11::CLI11
  fmt::fmt
  spdlog::spdlog
  function2::function2
  Glob
  natsort::natsort
  mp-units::mp-units
  TBB::tbb
  csv2::csv2
)

install(TARGETS trace-utils
  EXPORT trace-utils-targets
  ARCHIVE DESTINATION "bin"
  LIBRARY DESTINATION "bin"
  RUNTIME DESTINATION "bin")

install(
  FILES $<TARGET_RUNTIME_DLLS:trace-utils>
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
