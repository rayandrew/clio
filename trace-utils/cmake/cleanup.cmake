# CC BY-SA 4.0
# https://stackoverflow.com/a/78685677

macro(cleanup_interface_include_directories TARGET)
  get_target_property(original_interface_incs ${TARGET} INTERFACE_INCLUDE_DIRECTORIES)
  set(final_interface_incs "")
  foreach(inc ${original_interface_incs})
    if("${inc}" MATCHES "^(${CMAKE_CURRENT_SOURCE_DIR}|${CMAKE_CURRENT_BINARY_DIR})")
      continue()
    endif()
    list(APPEND final_interface_incs ${inc})
  endforeach()
  set_target_properties(${TARGET} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${final_interface_incs}")
  unset(final_interface_incs)
endmacro()
