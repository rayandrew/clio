cmake_minimum_required(VERSION 3.8...3.18)

set(TRACE_UTILS_VERSION_MAJOR 0)
set(TRACE_UTILS_VERSION_MINOR 1)
set(TRACE_UTILS_VERSION_PATCH 0)
set(TRACE_UTILS_VERSION "${TRACE_UTILS_VERSION_MAJOR}.${TRACE_UTILS_VERSION_MINOR}.${TRACE_UTILS_VERSION_PATCH}")
set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS_INIT} -fPIC")

project(trace_utils VERSION ${TRACE_UTILS_VERSION} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
enable_testing()

# string(REPLACE "." ";" VERSION_LIST ${TRACE_UTILS_VERSION})
# list(GET VERSION_LIST 0 TRACE_UTILS_VERSION_MAJOR)
# list(GET VERSION_LIST 1 TRACE_UTILS_VERSION_MINOR)
# list(GET VERSION_LIST 2 TRACE_UTILS_VERSION_PATCH)

###############################################################################
# General Configuration
###############################################################################

option(NO_CUDA "Disable CUDA support" OFF)

add_definitions(-Wextra -Wall -Wpedantic -DBIT7Z_AUTO_FORMAT)
add_library(coverage_config INTERFACE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# add our cmake module directory to the path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# link shared lib with full rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# setup cache variables for ccmake
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo
        CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

set (CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")
set (BUILD_SHARED_LIBS "ON" CACHE BOOL "Build a shared library")

if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options (coverage_config INTERFACE
        -O0        # no optimization
        -g         # generate debug info
        --coverage # sets all required flags
    )
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
        target_link_options (coverage_config INTERFACE --coverage)
    else()
        target_link_libraries (coverage_config INTERFACE --coverage)
    endif()
endif()

###############################################################################
# Dependencies
###############################################################################

include(cmake/CPM.cmake)
include(cmake/cleanup.cmake)

find_package(Filesystem)

set(FMT_INSTALL ON)
CPMAddPackage("gh:fmtlib/fmt#11.0.0")

set(SPDLOG_FMT_EXTERNAL ON)
set(SPDLOG_BUILD_SHARED ON)
set(SPDLOG_INSTALL ON)
CPMAddPackage("gh:gabime/spdlog@1.14.1")

if (NOT Filesystem_FOUND)
  set(GHC_FILESYSTEM_WITH_INSTALL ON)
  CPMAddPackage("gh:gulrak/filesystem@1.5.14")
  add_compile_definitions(USE_GHC_FILESYSTEM)
  set(GLOB_USE_GHC_FILESYSTEM ON)
endif()

CPMAddPackage("gh:p-ranav/glob#d025092c0e1eb1a8b226d3a799fd32680d2fd13f")

CPMAddPackage("gh:p-ranav/argparse#a2b4d27989177466d4ceceba05e8278f9da498f2")
CPMAddPackage("gh:p-ranav/csv2#4f3c41db6457465e94b92b91fc560b911c16a16a")
CPMAddPackage("gh:p-ranav/indicators#222382c3a6abbce32503792c59826063660ddb56")
CPMAddPackage("gh:oneapi-src/oneTBB@2021.13.0")

CPMAddPackage("gh:Neargye/scope_guard#05494fbbc3ec01a93b31c2b1c830b552f5ba08f4")

# CPMAddPackage(
#   NAME               7zip
#   GIT_TAG            a7a1d4a241492e81f659a920f7379c193593ebc6
#   GITHUB_REPOSITORY  ip7z/7zip
#   DOWNLOAD_ONLY      True
# )

# if(7zip_ADDED)
#   file(COPY "cmake/7zip/CMakeLists.txt" DESTINATION "${7zip_SOURCE_DIR}")
#   file(COPY "cmake/7zip/7zip-config.cmake.in" DESTINATION "${7zip_SOURCE_DIR}")
#   add_subdirectory("${7zip_SOURCE_DIR}")
# endif()
# CPMAddPackage("gh:rikyoz/bit7z@4.0.7")
# cleanup_interface_sources(bit7z64)

set(ENABLE_TAR OFF)
set(ENABLE_CPIO OFF)
set(ENABLE_CAT OFF)
set(ENABLE_UNZIP OFF)
set(ENABLE_ICONV OFF)
set(ENABLE_TEST OFF)
set(ENABLE_ACL OFF)
set(ENABLE_XATTR OFF)
set(ENABLE_INSTALL ON)
CPMAddPackage("gh:libarchive/libarchive@3.7.4")

if(BUILD_SHARED_LIBS)
  cleanup_interface_include_directories(archive)
  add_library(archive::archive ALIAS archive)
  set(LIBARCHIVE_TARGET archive)
else()
  cleanup_interface_include_directories(archive_static)
  add_library(archive::archive ALIAS archive_static)
  set(LIBARCHIVE_TARGET archive_static)
endif()

find_package(PkgConfig REQUIRED)

find_package(Python COMPONENTS Interpreter Development)
# set(PYBIND11_FINDPYTHON ON)
# find_package(pybind11 CONFIG REQUIRED)

# find_package(libarchive)

add_subdirectory(src)
add_subdirectory(app)
