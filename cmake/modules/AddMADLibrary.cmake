macro(add_mad_library _name _source_files _header_files _dep_mad_comp _include_dir)

  # Create the MADNESS library and object library
  add_library(MAD${_name}-obj OBJECT ${${_source_files}} ${${_header_files}})
  add_library(MAD${_name} $<TARGET_OBJECTS:MAD${_name}-obj>)
  if(BUILD_SHARED_LIBS)
    set_target_properties(MAD${_name}-obj PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  endif()
  set_target_properties(MAD${_name} PROPERTIES PUBLIC_HEADER "${${_header_files}}")
  
  # Pass the private MAD${_name} compile flags to MAD${_name}-obj  
  target_compile_definitions(MAD${_name}-obj PRIVATE 
      $<TARGET_PROPERTY:MAD${_name},COMPILE_DEFINITIONS>)
  target_include_directories(MAD${_name}-obj PRIVATE 
      $<TARGET_PROPERTY:MAD${_name},INCLUDE_DIRECTORIES>)
  target_compile_options(MAD${_name}-obj PRIVATE 
      $<TARGET_PROPERTY:MAD${_name},COMPILE_OPTIONS>)
  
  # Add target dependencies
  add_library(${_name} ALIAS MAD${_name})
  add_dependencies(libraries MAD${_name})
  
  # Add library to the list of installed components
  install(TARGETS MAD${_name} EXPORT madness
      COMPONENT ${_name}
      PUBLIC_HEADER DESTINATION "${MADNESS_INSTALL_INCLUDEDIR}/${_include_dir}"
      LIBRARY DESTINATION "${MADNESS_INSTALL_LIBDIR}"
      ARCHIVE DESTINATION "${MADNESS_INSTALL_LIBDIR}"
      INCLUDES DESTINATION "${MADNESS_INSTALL_INCLUDEDIR}")
  
  # Create a target to install the component
  add_custom_target(install-${_name}
      COMMAND ${CMAKE_COMMAND} -DCOMPONENT=${_name} -P ${CMAKE_BINARY_DIR}/cmake_install.cmake
      COMMENT "Installing ${_name} library components")
  add_dependencies(install-${_name} MAD${_name})
  add_dependencies(install-libraries install-${_name})

  foreach(_dep ${_dep_mad_comp})
    if(TARGET install-${_dep})
      add_dependencies(install-${_name} install-${_dep})
    endif()
    if(TARGET ${_dep})
      target_compile_definitions(MAD${_name} PUBLIC 
          $<TARGET_PROPERTY:${_dep},INTERFACE_COMPILE_DEFINITIONS>)
      target_include_directories(MAD${_name} PUBLIC 
          $<TARGET_PROPERTY:${_dep},INTERFACE_INCLUDE_DIRECTORIES>)
      target_compile_options(MAD${_name} PUBLIC 
          $<TARGET_PROPERTY:${_dep},INTERFACE_COMPILE_OPTIONS>)
      target_link_libraries(MAD${_name} PUBLIC ${_dep})
    endif()
  endforeach()
  
  # Add compile and linker flags to library
  if(CXX11_COMPILE_FLAG)
    target_compile_options(MAD${_name} INTERFACE $<INSTALL_INTERFACE:${CXX11_COMPILE_FLAG}>)
  endif()
  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(MAD${_name} PUBLIC "-Wl,-no_pie")
  endif()
  if(GPERFTOOLS_FOUND)
    target_include_directories(MAD${_name} PUBLIC ${GPERFTOOLS_INCLUDE_DIRS})
    target_link_libraries(MAD${_name} PUBLIC ${GPERFTOOLS_LIBRARIES})
  endif()
  if(LIBUNWIND_FOUND)
    target_include_directories(MAD${_name} PUBLIC ${LIBUNWIND_INCLUDE_DIRS})
    target_link_libraries(MAD${_name} PUBLIC ${LIBUNWIND_LIBRARIES})
  endif()
  target_link_libraries(MAD${_name} PUBLIC ${CMAKE_THREAD_LIBS_INIT})

endmacro()