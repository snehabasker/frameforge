# Populated in M5 with Google Benchmark microbenchmarks for the queue
# implementations. Left as a stub so FRAMEFORGE_BUILD_BENCH=ON has a target
# to configure against from M0 onward.
add_executable(frameforge_bench_placeholder placeholder_bench.cpp)
target_link_libraries(frameforge_bench_placeholder PRIVATE frameforge_placeholder)
