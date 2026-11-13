cmake_minimum_required(VERSION 3.16)
project(vectlink VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VECTLINK_SOURCES
    src/integrity/crc32c.c
    src/protocol/header.c
    src/transport/ring_buffer.c
)

if(WIN32)
    list(APPEND VECTLINK_SOURCES src/transport/shm_windows.c)
else()
    list(APPEND VECTLINK_SOURCES src/transport/shm_posix.c)
endif()

add_library(vectlink STATIC ${VECTLINK_SOURCES})

target_include_directories(vectlink
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(UNIX AND NOT APPLE)
    target_link_libraries(vectlink PUBLIC rt)
endif()

if(MSVC)
    target_compile_options(vectlink PRIVATE /W4 /WX)
else()
    target_compile_options(vectlink PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

function(vectlink_add_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE vectlink)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Werror -pedantic)
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

enable_testing()
vectlink_add_test(test_protocol tests/test_protocol.c)
vectlink_add_test(test_ring_buffer tests/test_ring_buffer.c)
vectlink_add_test(test_cross_process tests/test_cross_process.c)

add_executable(agent_sender examples/agent_sender.c)
target_link_libraries(agent_sender PRIVATE vectlink)
if(MSVC)
    target_compile_options(agent_sender PRIVATE /W4 /WX)
else()
    target_compile_options(agent_sender PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

add_executable(agent_receiver examples/agent_receiver.c)
target_link_libraries(agent_receiver PRIVATE vectlink)
if(MSVC)
    target_compile_options(agent_receiver PRIVATE /W4 /WX)
else()
    target_compile_options(agent_receiver PRIVATE -Wall -Wextra -Werror -pedantic)
endif()
