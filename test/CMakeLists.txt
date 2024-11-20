add_executable(test_main test_main.cc)

add_test(NAME test_main COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test/test_main.cc)
set_property(TEST test_main PROPERTY ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
