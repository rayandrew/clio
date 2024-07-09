# CC BY-SA 4.0
# https://stackoverflow.com/a/73948866/2418586

function(build_arrow)
    set(one_value_args)
    set(multi_value_args)

    cmake_parse_arguments(ARG
            "${options}"
            "${one_value_args}"
            "${multi_value_args}"
            ${ARGN})
    if (ARG_UNPARSED_ARGUMENTS)
        message(SEND_ERROR "Error: unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif ()

    # If Arrow needs to be built, the default location will be within the build tree.
    set(ARROW_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-prefix")

    set(ARROW_SHARED_LIBRARY_DIR "${ARROW_PREFIX}/lib")

    set(ARROW_SHARED_LIB_FILENAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}arrow${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(ARROW_SHARED_LIB "${ARROW_SHARED_LIBRARY_DIR}/${ARROW_SHARED_LIB_FILENAME}")
    set(PARQUET_SHARED_LIB_FILENAME
            "${CMAKE_SHARED_LIBRARY_PREFIX}parquet${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(PARQUET_SHARED_LIB "${ARROW_SHARED_LIBRARY_DIR}/${PARQUET_SHARED_LIB_FILENAME}")

    set(ARROW_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/arrow_ep-build")
    set(ARROW_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${ARROW_PREFIX}"
      "-DCMAKE_INSTALL_LIBDIR=lib" "-Dxsimd_SOURCE=BUNDLED"
      "-DBOOST_SOURCE=BUNDLED" "-DThrift_SOURCE=BUNDLED"
      "-DRapidJSON_SOURCE=BUNDLED"
      "-DARROW_BUILD_STATIC=OFF" "-DARROW_PARQUET=ON"
      "-DARROW_WITH_UTF8PROC=OFF" "-DARROW_WITH_RE2=OFF"
      "-DARROW_FILESYSTEM=ON" "-DARROW_CSV=ON" "-DARROW_PYTHON=ON")
    set(ARROW_INCLUDE_DIR "${ARROW_PREFIX}/include")

    set(ARROW_BUILD_BYPRODUCTS "${ARROW_SHARED_LIB}" "${PARQUET_SHARED_LIB}")

    include(ExternalProject)

    externalproject_add(arrow_ep
      # URL https://github.com/apache/arrow/archive/refs/tags/apache-arrow-9.0.0.tar.gz
      URL https://github.com/apache/arrow/archive/8e5c4e6acafb7ef6deeaff929de6a80d6815845d.tar.gz
      SOURCE_SUBDIR cpp
      BINARY_DIR "${ARROW_BINARY_DIR}"
      CMAKE_ARGS "${ARROW_CMAKE_ARGS}"
      BUILD_BYPRODUCTS "${ARROW_BUILD_BYPRODUCTS}")

    set(ARROW_LIBRARY_TARGET arrow_shared)
    set(PARQUET_LIBRARY_TARGET parquet_shared)

    file(MAKE_DIRECTORY "${ARROW_INCLUDE_DIR}")
    add_library(${ARROW_LIBRARY_TARGET} SHARED IMPORTED)
    add_library(${PARQUET_LIBRARY_TARGET} SHARED IMPORTED)
    set_target_properties(${ARROW_LIBRARY_TARGET}
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
            IMPORTED_LOCATION ${ARROW_SHARED_LIB})
    set_target_properties(${PARQUET_LIBRARY_TARGET}
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${ARROW_INCLUDE_DIR}
            IMPORTED_LOCATION ${PARQUET_SHARED_LIB})

    add_dependencies(${ARROW_LIBRARY_TARGET} arrow_ep)
endfunction()
