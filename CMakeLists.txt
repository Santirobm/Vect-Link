cmake_minimum_required(VERSION 3.16)
project(vectlink VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(VTLK_ENABLE_NATIVE_SIMD "Enable native SIMD (AVX2) optimizations" ON)

set(VECTLINK_SOURCES
    src/integrity/crc32c.c
    src/protocol/header.c
    src/transport/ring_buffer.c
    src/codec/fp16.c
    src/codec/int4.c
    src/codec/codec.c
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
    find_package(Threads REQUIRED)
    target_link_libraries(vectlink PUBLIC rt m Threads::Threads)
elseif(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(vectlink PUBLIC Threads::Threads)
endif()

if(MSVC)
    target_compile_options(vectlink PRIVATE /W4 /WX)
    if(VTLK_ENABLE_NATIVE_SIMD)
        target_compile_options(vectlink PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(vectlink PRIVATE -Wall -Wextra -Werror -pedantic)
    if(VTLK_ENABLE_NATIVE_SIMD)
        target_compile_options(vectlink PRIVATE -march=native)
    endif()
endif()

function(vectlink_add_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE vectlink)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /WX /UNDEBUG)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Werror -pedantic -UNDEBUG)
    endif()
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

enable_testing()
vectlink_add_test(test_protocol tests/test_protocol.c)
vectlink_add_test(test_ring_buffer tests/test_ring_buffer.c)
vectlink_add_test(test_cross_process tests/test_cross_process.c)
vectlink_add_test(test_codec tests/test_codec.c)

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

add_executable(benchmark_codec benchmarks/benchmark_codec.c)
target_link_libraries(benchmark_codec PRIVATE vectlink)
if(MSVC)
    target_compile_options(benchmark_codec PRIVATE /O2 /W4 /WX)
    if(VTLK_ENABLE_NATIVE_SIMD)
        target_compile_options(benchmark_codec PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(benchmark_codec PRIVATE -O3 -Wall -Wextra -Werror -pedantic)
    if(VTLK_ENABLE_NATIVE_SIMD)
        target_compile_options(benchmark_codec PRIVATE -march=native)
    endif()
endif()

add_executable(benchmark_transport benchmarks/benchmark_transport.c)
target_include_directories(benchmark_transport PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(benchmark_transport PRIVATE vectlink)
if(WIN32)
    target_link_libraries(benchmark_transport PRIVATE ws2_32)
else()
    target_link_libraries(benchmark_transport PRIVATE pthread)
endif()
if(MSVC)
    target_compile_options(benchmark_transport PRIVATE /O2 /W4 /WX)
    if(VTLK_ENABLE_NATIVE_SIMD)
        target_compile_options(benchmark_transport PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(benchmark_transport PRIVATE -O3 -Wall -Wextra -Werror -pedantic)
    if(VTLK_ENABLE_NATIVE_SIMD)
        target_compile_options(benchmark_transport PRIVATE -march=native)
    endif()
endif()
