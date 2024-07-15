cmake_minimum_required(VERSION 3.8...3.18)

set(TRACE_UTILS_VERSION_MAJOR 0)
set(TRACE_UTILS_VERSION_MINOR 1)
set(TRACE_UTILS_VERSION_PATCH 0)
set(TRACE_UTILS_VERSION "${TRACE_UTILS_VERSION_MAJOR}.${TRACE_UTILS_VERSION_MINOR}.${TRACE_UTILS_VERSION_PATCH}")
set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS_INIT} -fPIC")

project(trace_utils VERSION ${TRACE_UTILS_VERSION} LANGUAGES C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
enable_testing()

# string(REPLACE "." ";" VERSION_LIST ${TRACE_UTILS_VERSION})
# list(GET VERSION_LIST 0 TRACE_UTILS_VERSION_MAJOR)
# list(GET VERSION_LIST 1 TRACE_UTILS_VERSION_MINOR)
# list(GET VERSION_LIST 2 TRACE_UTILS_VERSION_PATCH)

###############################################################################
# General Configuration
###############################################################################

# option(NO_CUDA "Disable CUDA support" OFF)

add_definitions(-Wextra -Wall -Wpedantic)
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

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
 add_definitions(-Wno-format -Wno-format-security -Wno-overflow)
endif()

set(CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")
set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build a shared library")

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
# include(cmake/Arrow.cmake)
include(cmake/cleanup.cmake)

find_package(PkgConfig REQUIRED)

find_package(Filesystem)

set(FMT_INSTALL ON)
CPMAddPackage("gh:fmtlib/fmt#10.2.1")

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
CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")
CPMAddPackage("gh:p-ranav/csv2#4f3c41db6457465e94b92b91fc560b911c16a16a")
CPMAddPackage("gh:p-ranav/indicators#222382c3a6abbce32503792c59826063660ddb56")
CPMAddPackage("gh:oneapi-src/oneTBB@2021.13.0")
CPMAddPackage("gh:Neargye/scope_guard#05494fbbc3ec01a93b31c2b1c830b552f5ba08f4")
CPMAddPackage("gh:Naios/function2#43fc0ca473ecb081918709bd7d524d84c2ff8dce")
CPMAddPackage("gh:Neargye/magic_enum#664ee62c12570948b0e025d15b42d641fba8d54a")

CPMAddPackage(
  NAME                NaturalSort
  GIT_TAG             5c71f466f29d44c8d976dc1ba077eae9b6ce672a
  GITHUB_REPOSITORY   scopeInfinity/NaturalSort
  DOWNLOAD_ONLY       True
)
if(NaturalSort_ADDED)
  add_library(natsort INTERFACE)
  add_library(natsort::natsort ALIAS natsort)
  target_include_directories(natsort INTERFACE ${NaturalSort_SOURCE_DIR})
endif()

CPMAddPackage("gh:gsl-lite/gsl-lite@0.41.0")

CPMAddPackage(
  NAME                mp-units
  VERSION             2.1.0
  GITHUB_REPOSITORY   mpusz/mp-units
  DOWNLOAD_ONLY       True
)
if(mp-units_ADDED)
  set(MP_UNITS_API_CONTRACTS NONE)
  add_subdirectory(${mp-units_SOURCE_DIR}/src)
endif()



# set(ENV{TARGET} x86_64-linux-gnu)
# set(libunwind_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/libunwind)
CPMAddPackage(
  NAME                libunwind
  # GIT_TAG             3c47821d681777e3cff33edb25c804d93102e1c6
  GIT_TAG             v1.8.1
  GITHUB_REPOSITORY   libunwind/libunwind
  DOWNLOAD_ONLY       True
)
if(NOT EXISTS "${libunwind_BINARY_DIR}/include" AND libunwind_ADDED)

  # set(libunwind_INCLUDE_DIR ${libunwind_SOURCE_DIR}/include)
  # set(libunwind_SRC_DIR ${libunwind_SOURCE_DIR}/src)

  # set(libunwind_la_SOURCES_os_linux
  #   ${libunwind_SRC_DIR}/os-linux.c
  # )

  # set(libunwind_la_SOURCES_os_linux_local
  #   # Nothing when we don't want to support CXX exceptions
  # )

  # set(libunwind_la_SOURCES_os                 ${libunwind_la_SOURCES_os_linux})
  # set(libunwind_la_SOURCES_os_local           ${libunwind_la_SOURCES_os_linux_local})
  # set(libunwind_la_SOURCES_x86_64_os          ${libunwind_SRC_DIR}/x86_64/Gos-linux.c)
  # set(libunwind_la_SOURCES_x86_64_os_local    ${libunwind_SRC_DIR}/x86_64/Los-linux.c)

  # set(libunwind_la_SOURCES_local_unwind
  #   # Nothing when we don't want to support CXX exceptions
  # )

  # set(libunwind_la_SOURCES_local_nounwind
  #   ${libunwind_la_SOURCES_os_local}
  #   ${libunwind_SRC_DIR}/mi/backtrace.c
  #   ${libunwind_SRC_DIR}/mi/dyn-cancel.c
  #   ${libunwind_SRC_DIR}/mi/dyn-info-list.c
  #   ${libunwind_SRC_DIR}/mi/dyn-register.c
  #   ${libunwind_SRC_DIR}/mi/Ldyn-extract.c
  #   ${libunwind_SRC_DIR}/mi/Lfind_dynamic_proc_info.c
  #   ${libunwind_SRC_DIR}/mi/Lget_accessors.c
  #   ${libunwind_SRC_DIR}/mi/Lget_proc_info_by_ip.c
  #   ${libunwind_SRC_DIR}/mi/Lget_proc_name.c
  #   ${libunwind_SRC_DIR}/mi/Lput_dynamic_unwind_info.c
  #   ${libunwind_SRC_DIR}/mi/Ldestroy_addr_space.c
  #   ${libunwind_SRC_DIR}/mi/Lget_reg.c
  #   ${libunwind_SRC_DIR}/mi/Lset_reg.c
  #   ${libunwind_SRC_DIR}/mi/Lget_fpreg.c
  #   ${libunwind_SRC_DIR}/mi/Lset_fpreg.c
  #   ${libunwind_SRC_DIR}/mi/Lset_caching_policy.c
  #   ${libunwind_SRC_DIR}/mi/Lset_cache_size.c
  #   ${libunwind_SRC_DIR}/mi/Lset_iterate_phdr_function.c
  #   ${libunwind_SRC_DIR}/mi/Lget_elf_filename.c
  # )
  
  # set(libunwind_la_SOURCES_common
  #   ${libunwind_la_SOURCES_os}
  #   ${libunwind_SRC_DIR}/mi/init.c
  #   ${libunwind_SRC_DIR}/mi/flush_cache.c
  #   ${libunwind_SRC_DIR}/mi/mempool.c
  #   ${libunwind_SRC_DIR}/mi/strerror.c
  # )
  
  # set(libunwind_la_SOURCES_x86_64_common
  #   ${libunwind_la_SOURCES_common}
  #   ${libunwind_SRC_DIR}/x86_64/is_fpreg.c
  #   ${libunwind_SRC_DIR}/x86_64/regname.c
  # )

  # set(libunwind_la_SOURCES_local
  #   ${libunwind_la_SOURCES_local_nounwind}
  #   ${libunwind_la_SOURCES_local_unwind}
  # )

  # # The list of files that go into libunwind:
  # set(libunwind_la_SOURCES_x86_64
  #   ${libunwind_la_SOURCES_x86_64_common}
  #   ${libunwind_la_SOURCES_x86_64_os_local}
  #   ${libunwind_la_SOURCES_local}
  #   ${libunwind_SRC_DIR}/x86_64/setcontext.S
  #   ${libunwind_SRC_DIR}/x86_64/Lapply_reg_state.c
  #   ${libunwind_SRC_DIR}/x86_64/Lreg_states_iterate.c
  #   ${libunwind_SRC_DIR}/x86_64/Lcreate_addr_space.c
  #   ${libunwind_SRC_DIR}/x86_64/Lget_save_loc.c
  #   ${libunwind_SRC_DIR}/x86_64/Lglobal.c
  #   ${libunwind_SRC_DIR}/x86_64/Linit.c
  #   ${libunwind_SRC_DIR}/x86_64/Linit_local.c
  #   ${libunwind_SRC_DIR}/x86_64/Linit_remote.c
  #   ${libunwind_SRC_DIR}/x86_64/Lget_proc_info.c
  #   ${libunwind_SRC_DIR}/x86_64/Lregs.c
  #   ${libunwind_SRC_DIR}/x86_64/Lresume.c
  #   ${libunwind_SRC_DIR}/x86_64/Lstash_frame.c
  #   ${libunwind_SRC_DIR}/x86_64/Lstep.c
  #   ${libunwind_SRC_DIR}/x86_64/Ltrace.c
  #   ${libunwind_SRC_DIR}/x86_64/getcontext.S
  # )

  # # The list of files that go into libunwind-x86_64:
  # set(libunwind_x86_64_la_SOURCES_x86_64
  #   ${libunwind_la_SOURCES_x86_64_common}
  #   ${libunwind_la_SOURCES_x86_64_os}
  #   ${libunwind_la_SOURCES_generic}
  #   ${libunwind_SRC_DIR}/x86_64/Gapply_reg_state.c
  #   ${libunwind_SRC_DIR}/x86_64/Greg_states_iterate.c
  #   ${libunwind_SRC_DIR}/x86_64/Gcreate_addr_space.c
  #   ${libunwind_SRC_DIR}/x86_64/Gget_save_loc.c
  #   ${libunwind_SRC_DIR}/x86_64/Gglobal.c
  #   ${libunwind_SRC_DIR}/x86_64/Ginit.c
  #   ${libunwind_SRC_DIR}/x86_64/Ginit_local.c
  #   ${libunwind_SRC_DIR}/x86_64/Ginit_remote.c
  #   ${libunwind_SRC_DIR}/x86_64/Gget_proc_info.c
  #   ${libunwind_SRC_DIR}/x86_64/Gregs.c
  #   ${libunwind_SRC_DIR}/x86_64/Gresume.c
  #   ${libunwind_SRC_DIR}/x86_64/Gstash_frame.c
  #   ${libunwind_SRC_DIR}/x86_64/Gstep.c
  #   ${libunwind_SRC_DIR}/x86_64/Gtrace.c
  # )

  # set(libunwind_elf64_la_SOURCES
  #   ${libunwind_SRC_DIR}/elf64.c
  # )

  # SET(libunwind_la_SOURCES                    ${libunwind_la_SOURCES_x86_64})
  # SET(libunwind_remote_la_SOURCES             ${libunwind_x86_64_la_SOURCES_x86_64})
  # SET(libunwind_elf_la_SOURCES                ${libunwind_elf64_la_SOURCES})
  # list(APPEND libunwind_setjmp_la_SOURCES
  #   ${libunwind_SRC_DIR}/x86_64/longjmp.S
  #   ${libunwind_SRC_DIR}/x86_64/siglongjmp.S)

  # set(libunwind_dwarf_common_la_SOURCES
  #   ${libunwind_SRC_DIR}/dwarf/global.c
  # )
  # set(libunwind_dwarf_local_la_SOURCES
  #   ${libunwind_SRC_DIR}/dwarf/Lexpr.c
  #   ${libunwind_SRC_DIR}/dwarf/Lfde.c
  #   ${libunwind_SRC_DIR}/dwarf/Lparser.c
  #   ${libunwind_SRC_DIR}/dwarf/Lpe.c
  #   ${libunwind_SRC_DIR}/dwarf/Lfind_proc_info-lsb.c
  #   ${libunwind_SRC_DIR}/dwarf/Lfind_unwind_table.c
  #   ${libunwind_SRC_DIR}/dwarf/Lget_proc_info_in_range.c
  # )

  # set(libunwind_dwarf_generic_la_SOURCES
  #   ${libunwind_SRC_DIR}/dwarf/Gexpr.c
  #   ${libunwind_SRC_DIR}/dwarf/Gfde.c
  #   ${libunwind_SRC_DIR}/dwarf/Gparser.c
  #   ${libunwind_SRC_DIR}/dwarf/Gpe.c
  #   ${libunwind_SRC_DIR}/dwarf/Gfind_proc_info-lsb.c
  #   ${libunwind_SRC_DIR}/dwarf/Gfind_unwind_table.c
  #   ${libunwind_SRC_DIR}/dwarf/Gget_proc_info_in_range.c
  # )


  # set(libunwind_coredump_la_SOURCES
  #   ${libunwind_SRC_DIR}/coredump/_UCD_accessors.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_create.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_destroy.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_access_mem.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_elf_map_image.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_find_proc_info.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_get_proc_name.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_get_elf_filename.c

  #   ${libunwind_SRC_DIR}/mi/init.c
  #   ${libunwind_SRC_DIR}/coredump/_UPT_elf.c
  #   ${libunwind_SRC_DIR}/coredump/_UPT_access_fpreg.c
  #   ${libunwind_SRC_DIR}/coredump/_UPT_get_dyn_info_list_addr.c
  #   ${libunwind_SRC_DIR}/coredump/_UPT_put_unwind_info.c
  #   ${libunwind_SRC_DIR}/coredump/_UPT_resume.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_access_reg_linux.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_get_threadinfo_prstatus.c
  #   ${libunwind_SRC_DIR}/coredump/_UCD_get_mapinfo_linux.c
  # )
  
  # add_library(libunwind
  #   ${libunwind_la_SOURCES}
  #   ${libunwind_remote_la_SOURCES}
  #   ${libunwind_elf_la_SOURCES}
  #   ${libunwind_setjmp_la_SOURCES}
  #   ${libunwind_coredump_la_SOURCES}
  #   ${libunwind_SRC_DIR}/mi/Gget_accessors.c)
  # add_library(libunwind::libunwind ALIAS libunwind)

  # set(TARGET_AMD64 1)
  # set(arch x86_64)
  # set(HAVE_ELF_H 1)
  # set(HAVE_ENDIAN_H 1)
  # set(PACKAGE_STRING "libunwind")
  # set(PACKAGE_BUGREPORT "")
  # configure_file(${libunwind_INCLUDE_DIR}/config.h.cmake.in
  #   ${libunwind_BINARY_DIR}/config.h)
  # configure_file(${libunwind_INCLUDE_DIR}/libunwind-common.h.in
  #   ${libunwind_BINARY_DIR}/libunwind-common.h)
  # configure_file(${libunwind_INCLUDE_DIR}/libunwind.h.in
  #   ${libunwind_BINARY_DIR}/libunwind.h)
  # configure_file(${libunwind_INCLUDE_DIR}/tdep/libunwind_i.h.in
  #   ${libunwind_BINARY_DIR}/tdep/libunwind_i.h)

  # target_include_directories(libunwind PUBLIC
  #   ${libunwind_BINARY_DIR}
  #   ${libunwind_BINARY_DIR}/include
  #   ${libunwind_BINARY_DIR}/tdep
  #   ${libunwind_INCLUDE_DIR}/tdep-x86_64
  #   ${libunwind_INCLUDE_DIR}
  #   ${libunwind_INCLUDE_DIR}/tdep
  #   ${libunwind_SRC_DIR}
  # )
  # target_compile_definitions(libunwind PUBLIC
  #   __x86_64__
  #   __amd64__
  #   __linux__
  #   USE_ELF64
  #   HAVE_ELF_H
  #   HAVE_ENDIAN_H
  #   _GNU_SOURCE
  #   HAVE_CONFIG_H
  #   HAVE_LINK_H)
  # target_link_libraries(libunwind
  #   -static-libgcc -static-libstdc++)
  # target_compile_options(libunwind PUBLIC
  #   -Xlinker
  #   -nostartfiles)
  # # set_target_properties(libunwind
  # #   PROPERTIES VERSION ${TRACE_UTILS_VERSION}
  # #   SOVERSION ${TRACE_UTILS_VERSION_MAJOR})
  # set(LIBUNWIND_INCLUDE_DIRS ${libunwind_INCLUDE_DIR})

  execute_process (
    COMMAND bash -c "cd ${libunwind_SOURCE_DIR} && autoreconf -i && ./configure --prefix=${libunwind_BINARY_DIR} && make &&make install"
  )
endif()


list(APPEND CMAKE_PREFIX_PATH "${libunwind_BINARY_DIR}/lib/pkgconfig")
set(ENV{PKG_CONFIG_PATH} "${libunwind_BINARY_DIR}/lib/pkgconfig")
pkg_check_modules(LIBUNWIND REQUIRED libunwind)
set(CPPTRACE_UNWIND_WITH_LIBUNWIND ON)
CPMAddPackage("gh:jeremy-rifkin/cpptrace#90de25f1dfe637b7929454644e39d0436606c999")

# find_package(OpenMP REQUIRED)

# Arrow + Parquet
# CPMAddPackage(
#   NAME                arrow
#   GIT_TAG             8e5c4e6acafb7ef6deeaff929de6a80d6815845d
#   GITHUB_REPOSITORY   apache/arrow
#   DOWNLOAD_ONLY       False
# )
# if(arrow_ADDED)
#   set(arrow_SOURCE_DIR "${arrow_SOURCE_DIR}/cpp")
#   set(ARROW_PACKAGE_PREFIX "${arrow_SOURCE_DIR}/cpp")
#   set(ARROW_SIMD_LEVEL "NONE" CACHE STRING "" FORCE)
#   set(ARROW_OPTIONAL_INSTALL OFF)
#   set(Thrift_SOURCE "BUNDLED")
#   set(BOOST_SOURCE "BUNDLED")
#   set(xsimd_SOURCE "BUNDLED")
#   set(RapidJSON_SOURCE "BUNDLED")
#   set(ARROW_BUILD_TESTS OFF)
#   set(ARROW_BUILD_SHARED ON)
#   set(ARROW_PARQUET ON)
#   set(ARROW_CSV ON)
#   set(ARROW_FILESYSTEM ON)
#   set(ARROW_WITH_UTF8PROC OFF)
#   set(PARQUET_MINIMAL_DEPENDENCY OFF)
#   add_subdirectory(${arrow_SOURCE_DIR})
# endif()

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

find_package(Python COMPONENTS Interpreter Development)
# set(PYBIND11_FINDPYTHON ON)
# find_package(pybind11 CONFIG REQUIRED)

# find_package(libarchive)

add_subdirectory(src)
add_subdirectory(app)
