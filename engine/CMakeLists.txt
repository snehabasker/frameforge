cmake_minimum_required(VERSION 3.20)
project(frameforge_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FRAMEFORGE_BUILD_TESTS "Build GoogleTest unit tests" ON)
option(FRAMEFORGE_BUILD_BENCH "Build Google Benchmark microbenchmarks" OFF)
option(FRAMEFORGE_ENABLE_CUDA_EP "Enable ONNX Runtime CUDA execution provider" OFF)
option(FRAMEFORGE_ENABLE_ASAN "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(FRAMEFORGE_ENABLE_TSAN "Build with ThreadSanitizer" OFF)

if(FRAMEFORGE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

if(FRAMEFORGE_ENABLE_TSAN)
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# M0: placeholder target so the build is green end to end before any real
# sources exist. Milestone M4 replaces this with the real library + CLI.
add_library(frameforge_placeholder INTERFACE)
target_include_directories(frameforge_placeholder INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(frameforge_version_probe src/version_probe.cpp)
target_link_libraries(frameforge_version_probe PRIVATE frameforge_placeholder)

if(FRAMEFORGE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(FRAMEFORGE_BUILD_BENCH)
  add_subdirectory(bench)
endif()

# baselines/fsr1_easu lives outside engine/ per the spec's repo layout
# (section 2), but shares this build tree (and, when tests are enabled,
# the gtest_main target that engine/tests/CMakeLists.txt just fetched --
# hence this add_subdirectory comes AFTER the tests one above).
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../baselines/fsr1_easu
  ${CMAKE_CURRENT_BINARY_DIR}/fsr1_easu
)
