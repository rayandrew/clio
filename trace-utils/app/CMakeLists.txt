set(app-src-files main.cpp)

add_executable(trace-utils ${app-src-files})
target_link_libraries(trace-utils
  trace-utils::lib
  argparse::argparse
  fmt::fmt
  spdlog::spdlog
  7zip
)

install(TARGETS trace-utils
  EXPORT trace-utils-targets
  ARCHIVE DESTINATION "bin"
  LIBRARY DESTINATION "bin"
  RUNTIME DESTINATION "bin")

install(
  FILES $<TARGET_RUNTIME_DLLS:trace-utils>
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  FILES $<TARGET_RUNTIME_DLLS:7zip>
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
