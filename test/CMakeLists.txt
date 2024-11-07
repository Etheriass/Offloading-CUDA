add_test(NAME hello_world_test COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/hello_world_test)
set_property(TEST hello_world_test PROPERTY ENVIRONMENT
        "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
