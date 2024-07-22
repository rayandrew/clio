###############################################################################
# General Configuration
###############################################################################

# load package helper for generating cmake CONFIG packages
include(CMakePackageConfigHelpers)

# where to install files for "find_package"
set(trace-utils-pkg "share/cmake/trace-utils")

# set version fo shared library
set(trace-utils-vers "${TRACE_UTILS_VERSION_MAJOR}.${TRACE_UTILS_VERSION_MINOR}")

# configure config.h
configure_file("config.hpp.in" "config.hpp" @ONLY)

set(DEST_DIR "${CMAKE_INSTALL_PREFIX}")

###############################################################################
# Library
###############################################################################

set(lib-src-files
  logger.cpp
  utils.cpp
  stats.cpp
  characteristic.cpp
  trace/replayer.cpp
  trace/msft.cpp
  trace/tencent.cpp
  trace/alibaba.cpp)

add_library(trace-utils-lib ${lib-src-files})
add_library(trace-utils::lib ALIAS trace-utils-lib)
target_link_libraries(trace-utils-lib
  PUBLIC
  PRIVATE
      TBB::tbb
      spdlog::spdlog
      fmt::fmt 
      csv2::csv2
      indicators::indicators
      archive::archive
      scope_guard::scope_guard
      function2::function2
      mp-units::mp-units
      magic_enum::magic_enum
      cpptrace::cpptrace
      tsl::ordered_map
      # arrow_shared
      # parquet_shared
)

if(Filesystem_FOUND)
  if(TARGET std::filesystem)
    target_link_libraries(trace-utils-lib PRIVATE std::filesystem)
  endif()
else()  
  target_link_libraries(trace-utils-lib PRIVATE ghcFilesystem::ghc_filesystem)
endif()

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_libraries(trace-utils-lib coverage_config)
endif()

target_include_directories(trace-utils-lib PUBLIC $<INSTALL_INTERFACE:include> ${PYTHON_INCLUDE_DIRS})
target_include_directories(trace-utils-lib BEFORE PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
target_include_directories(trace-utils-lib BEFORE PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
target_include_directories(trace-utils-lib BEFORE PRIVATE
    "${libarchive_SOURCE_DIR}/libarchive")
set_target_properties(trace-utils-lib
  PROPERTIES VERSION ${TRACE_UTILS_VERSION}
  SOVERSION ${TRACE_UTILS_VERSION_MAJOR})

###############################################################################
# Install
###############################################################################

set(TRACE_UTILS_PRIVATE_LIBS "-ltrace-utils-lib")

configure_file("trace-utils.pc.in" "trace-utils.pc" @ONLY)

# installation stuff (packaging and install commands)
write_basic_package_version_file(
  "trace-utils-config-version.cmake"
  VERSION ${TRACE_UTILS_VERSION}
  COMPATIBILITY AnyNewerVersion)

# generate our config file for find_package()
set(INCLUDE_INSTALL_DIR include/)
set(LIB_INSTALL_DIR lib/)
configure_package_config_file(trace-utils-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/trace-utils-config.cmake
  INSTALL_DESTINATION ${LIB_INSTALL_DIR}/trace-utils/cmake
  PATH_VARS INCLUDE_INSTALL_DIR)

if (DEFINED USE_SKBUILD)
  set(RUNTIME_DESTINATION "trace_utils/bin")
  set(LIB_DESTINATION "trace_utils/lib")
  set(INCLUDE_DESTINATION "trace_utils/include")
else()
  set(RUNTIME_DESTINATION "lib")
  set(LIB_DESTINATION "lib")
  set(INCLUDE_DESTINATION "include")
endif()

# "make install" rules
install(TARGETS trace-utils-lib indicators ${LIBARCHIVE_TARGET}
  EXPORT trace-utils-lib-targets
  ARCHIVE DESTINATION ${LIB_DESTINATION}
  LIBRARY DESTINATION ${LIB_DESTINATION}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION})
install(
  FILES $<TARGET_RUNTIME_DLLS:trace-utils-lib>
  DESTINATION ${LIB_DESTINATION})
install(EXPORT trace-utils-lib-targets
  DESTINATION ${trace-utils-pkg}
  FILE "trace-utils-targets.cmake")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/trace-utils-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/trace-utils-config-version.cmake"
  DESTINATION ${trace-utils-pkg})
install(DIRECTORY ../include/trace-utils
  DESTINATION ${INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ../include/trace-utils
  DESTINATION ${INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/trace-utils.pc"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/pkgconfig/")
