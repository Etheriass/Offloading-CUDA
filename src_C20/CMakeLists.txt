#------------------------------------------------------------------------------
# Include Header Directories
#------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)


#------------------------------------------------------------------------------
# Build Helloworld DB
#------------------------------------------------------------------------------
add_executable(hello_world ${CMAKE_CURRENT_SOURCE_DIR}/hello_world.cc)
# add_dependencies(hello_world database_lib)
# target_link_libraries(hello_world database_lib)


#------------------------------------------------------------------------------
# Add libraries + executables to CMake install
#------------------------------------------------------------------------------
install(
        TARGETS
        hello_world
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

#-----------------------------------------------------------------------------
# Add header file(s) to CMake Install
#-----------------------------------------------------------------------------
# install(
#         FILES
#         ${CMAKE_SOURCE_DIR}/include/database_lib.h
#         DESTINATION
#         ${CMAKE_INSTALL_PREFIX}/include
#         COMPONENT
#         headers)
