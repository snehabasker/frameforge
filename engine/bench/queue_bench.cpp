// Benchmarks BlockingQueue vs SpscRingBuffer under a single-producer/
// single-consumer workload (CLAUDE.md 7.3: "Benchmark both and keep the
// better one as default, with the numbers in RESULTS.md"). Numbers from
// running this are copied into docs/RESULTS.md by hand from the actual
// benchmark output (not invented) -- see that file for the real run.
#include <thread>

#include <benchmark/benchmark.h>

#include "frameforge/queue.hpp"

namespace {

void BM_BlockingQueue_SPSC(benchmark::State& state) {
  const int n_items = static_cast<int>(state.range(0));
  for (auto _ : state) {
    frameforge::BlockingQueue<int> q(64);
    std::thread producer([&] {
      for (int i = 0; i < n_items; ++i) q.Push(i);
      q.Close();
    });
    int received = 0;
    while (auto v = q.Pop()) {
      benchmark::DoNotOptimize(v);
      ++received;
    }
    producer.join();
    benchmark::DoNotOptimize(received);
  }
  state.SetItemsProcessed(state.iterations() * n_items);
}
BENCHMARK(BM_BlockingQueue_SPSC)->Arg(1000)->Arg(100000)->Unit(benchmark::kMillisecond);

void BM_SpscRingBuffer_SPSC(benchmark::State& state) {
  const int n_items = static_cast<int>(state.range(0));
  for (auto _ : state) {
    frameforge::SpscRingBuffer<int> rb(64);
    std::thread producer([&] {
      int i = 0;
      while (i < n_items) {
        if (rb.TryPush(i)) ++i;
      }
    });
    int received = 0;
    while (received < n_items) {
      if (auto v = rb.TryPop()) {
        benchmark::DoNotOptimize(v);
        ++received;
      }
    }
    producer.join();
    benchmark::DoNotOptimize(received);
  }
  state.SetItemsProcessed(state.iterations() * n_items);
}
BENCHMARK(BM_SpscRingBuffer_SPSC)->Arg(1000)->Arg(100000)->Unit(benchmark::kMillisecond);

}  // namespace

BENCHMARK_MAIN();
