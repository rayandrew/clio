set(app-src-files main.cpp)

add_executable(trace-utils ${app-src-files})
target_link_libraries(trace-utils
  trace-utils::lib
  argparse::argparse
  fmt::fmt
  spdlog::spdlog)
