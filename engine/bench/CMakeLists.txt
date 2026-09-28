include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# Google Benchmark's own sysinfo.cc hits an unhandled-enum -Werror=switch
# on this GCC 16 / MinGW combination (an upstream gap, not our code) --
# disable -Werror for its own sources only; our warnings stay strict.
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.zip
  URL_HASH SHA256=8a63c9c6adf9e7ce8d0d81f251c47de83efb5e077e147d109fa2045daac8368b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(frameforge_bench queue_bench.cpp)
target_link_libraries(frameforge_bench PRIVATE frameforge_core benchmark::benchmark)
