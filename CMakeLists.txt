cmake_minimum_required(VERSION 3.16)
project(vectlink VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(VTLK_DISABLE_AVX2 "Disable AVX2 runtime dispatch (simulate legacy CPU)" OFF)
option(VTLK_BUILD_LLAMA_BRIDGE "Build llama.cpp latent bridge example" OFF)

set(VECTLINK_SOURCES
    src/integrity/crc32c.c
    src/protocol/header.c
    src/transport/ring_buffer.c
    src/codec/fp16.c
    src/codec/int4.c
    src/codec/codec.c
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|X86_64|i686|x86")
    list(APPEND VECTLINK_SOURCES src/codec/int4_avx2.c src/integrity/crc32c_sse42.c)
    if(NOT VTLK_DISABLE_AVX2)
        if(MSVC)
            set_source_files_properties(src/codec/int4_avx2.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties(src/codec/int4_avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2")
        endif()
    endif()
    if(NOT MSVC)
        set_source_files_properties(src/integrity/crc32c_sse42.c PROPERTIES COMPILE_OPTIONS "-msse4.2")
    endif()
endif()

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
else()
    target_compile_options(vectlink PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

if(VTLK_DISABLE_AVX2)
    target_compile_definitions(vectlink PRIVATE VTLK_DISABLE_AVX2=1)
endif()

add_library(vectlink_shared SHARED ${VECTLINK_SOURCES})
set_target_properties(vectlink_shared PROPERTIES OUTPUT_NAME "vectlink")
if(WIN32)
    set_target_properties(vectlink_shared PROPERTIES PREFIX "")
endif()
target_include_directories(vectlink_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(UNIX AND NOT APPLE)
    target_link_libraries(vectlink_shared PUBLIC rt m Threads::Threads)
elseif(UNIX)
    target_link_libraries(vectlink_shared PUBLIC Threads::Threads)
endif()
if(MSVC)
    target_compile_options(vectlink_shared PRIVATE /W4 /WX)
else()
    target_compile_options(vectlink_shared PRIVATE -Wall -Wextra -Werror -pedantic -fvisibility=hidden)
endif()
target_compile_definitions(vectlink_shared PRIVATE VTLK_BUILDING_SHARED=1)
if(VTLK_DISABLE_AVX2)
    target_compile_definitions(vectlink_shared PRIVATE VTLK_DISABLE_AVX2=1)
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
target_include_directories(test_protocol PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
vectlink_add_test(test_ring_buffer tests/test_ring_buffer.c)
vectlink_add_test(test_cross_process tests/test_cross_process.c)
vectlink_add_test(test_codec tests/test_codec.c)
target_include_directories(test_codec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

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
else()
    target_compile_options(benchmark_codec PRIVATE -O3 -Wall -Wextra -Werror -pedantic)
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
else()
    target_compile_options(benchmark_transport PRIVATE -O3 -Wall -Wextra -Werror -pedantic)
endif()

if(VTLK_BUILD_LLAMA_BRIDGE)
    set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp repository root containing include/ and build/")
    if(NOT LLAMA_CPP_DIR)
        message(FATAL_ERROR "VTLK_BUILD_LLAMA_BRIDGE=ON requires -DLLAMA_CPP_DIR=<path to llama.cpp>")
    endif()

    find_path(LLAMA_INCLUDE_DIR
        NAMES llama.h
        PATHS "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}"
        NO_DEFAULT_PATH
    )
    find_path(GGML_INCLUDE_DIR
        NAMES ggml.h
        PATHS "${LLAMA_CPP_DIR}/ggml/include" "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}"
        NO_DEFAULT_PATH
    )
    find_library(LLAMA_LIBRARY
        NAMES llama libllama
        PATHS
            "${LLAMA_CPP_DIR}/build/src"
            "${LLAMA_CPP_DIR}/build/src/Release"
            "${LLAMA_CPP_DIR}/build/bin"
            "${LLAMA_CPP_DIR}/build/bin/Release"
            "${LLAMA_CPP_DIR}/build/lib"
            "${LLAMA_CPP_DIR}/build/lib/Release"
            "${LLAMA_CPP_DIR}/build"
        NO_DEFAULT_PATH
    )

    if(NOT LLAMA_INCLUDE_DIR)
        message(FATAL_ERROR "Could not find llama.h under LLAMA_CPP_DIR=${LLAMA_CPP_DIR}")
    endif()
    if(NOT LLAMA_LIBRARY)
        message(FATAL_ERROR "Could not find llama library under ${LLAMA_CPP_DIR}/build")
    endif()

    set(VTLK_LLAMA_INCLUDES "${LLAMA_INCLUDE_DIR}")
    if(GGML_INCLUDE_DIR)
        list(APPEND VTLK_LLAMA_INCLUDES "${GGML_INCLUDE_DIR}")
    endif()

    foreach(llama_target llama_agent_a llama_agent_b benchmark_llama_handoff benchmark_llama_multiprocess)
        add_executable(${llama_target} examples/${llama_target}.c)
        # llama.h and ggml.h use C11 features (_Noreturn, typedef redefs, anonymous unions)
        set_target_properties(${llama_target} PROPERTIES
            C_STANDARD 11
            C_STANDARD_REQUIRED ON
            C_EXTENSIONS OFF
        )
        target_include_directories(${llama_target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
            ${VTLK_LLAMA_INCLUDES}
        )
        target_link_libraries(${llama_target} PRIVATE vectlink "${LLAMA_LIBRARY}")
        if(NOT WIN32)
            target_link_libraries(${llama_target} PRIVATE pthread m)
        endif()
        if(MSVC)
            target_compile_options(${llama_target} PRIVATE /W4 /WX)
        else()
            target_compile_options(${llama_target} PRIVATE -Wall -Wextra -Werror)
        endif()
    endforeach()
endif()
