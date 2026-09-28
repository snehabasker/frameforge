// Two bounded queue implementations (CLAUDE.md 7.3):
//   - BlockingQueue<T>: mutex + condition_variable, general purpose.
//   - SpscRingBuffer<T>: lock-free single-producer/single-consumer ring
//     buffer using std::atomic with explicit memory orders.
// Both are benchmarked in engine/bench/queue_bench.cpp; the pipeline
// (pipeline.hpp) uses whichever wins (CLAUDE.md 7.3: "keep the better one
// as default, with the numbers in RESULTS.md").
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <queue>
#include <vector>

namespace frameforge {

// A simple bounded blocking queue. Push blocks while full; Pop blocks while
// empty. `Close()` wakes every blocked thread and makes further Pop() calls
// return std::nullopt once drained -- this is the clean-shutdown mechanism
// (CLAUDE.md 7.3: "no exceptions crossing thread boundaries silently. Clean
// shutdown on error or SIGINT").
template <typename T>
class BlockingQueue {
 public:
  explicit BlockingQueue(size_t capacity) : capacity_(capacity) {}

  // Returns false if the queue was closed before the item could be pushed.
  bool Push(T item) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
    if (closed_) return false;
    queue_.push(std::move(item));
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Returns std::nullopt once the queue is closed AND drained.
  std::optional<T> Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
    if (queue_.empty()) {
      return std::nullopt;  // closed_ and drained
    }
    T item = std::move(queue_.front());
    queue_.pop();
    lock.unlock();
    not_full_.notify_one();
    return item;
  }

  void Close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  size_t SizeApprox() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  size_t capacity_;
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::queue<T> queue_;
  bool closed_ = false;
};

// Lock-free single-producer/single-consumer bounded ring buffer.
//
// Memory-order reasoning (only two threads ever touch this: exactly one
// producer, exactly one consumer):
//   - `head_` (next write slot) is written ONLY by the producer, read by
//     both. The producer publishes a new item with a release store to
//     head_ AFTER writing the slot's data, so a consumer that observes the
//     new head_ (via an acquire load) is guaranteed to see that slot's
//     data too (release-acquire pairing on `head_`).
//   - `tail_` (next read slot) is written ONLY by the consumer, read by
//     both, with the same release-on-write/acquire-on-read pairing so the
//     producer can safely reuse a slot once it observes the consumer has
//     moved past it.
// Capacity is fixed at construction and rounded up to a power of two so
// the wrap-around index math is a cheap mask instead of a modulo.
template <typename T>
class SpscRingBuffer {
 public:
  explicit SpscRingBuffer(size_t capacity) : mask_(RoundUpPow2(capacity) - 1),
                                              buffer_(RoundUpPow2(capacity)) {}

  // Returns false if full (non-blocking by design -- callers needing to
  // block should pair this with their own backoff/condvar, or use
  // BlockingQueue instead).
  bool TryPush(T item) {
    size_t head = head_.load(std::memory_order_relaxed);
    size_t tail = tail_.load(std::memory_order_acquire);
    if (head - tail >= buffer_.size()) {
      return false;  // full
    }
    buffer_[head & mask_] = std::move(item);
    head_.store(head + 1, std::memory_order_release);
    return true;
  }

  // Returns std::nullopt if empty.
  std::optional<T> TryPop() {
    size_t tail = tail_.load(std::memory_order_relaxed);
    size_t head = head_.load(std::memory_order_acquire);
    if (tail == head) {
      return std::nullopt;  // empty
    }
    T item = std::move(buffer_[tail & mask_]);
    tail_.store(tail + 1, std::memory_order_release);
    return item;
  }

  size_t Capacity() const { return buffer_.size(); }

 private:
  static size_t RoundUpPow2(size_t v) {
    size_t p = 1;
    while (p < v) p <<= 1;
    return p == 0 ? 1 : p;
  }

  size_t mask_;
  std::vector<T> buffer_;
  alignas(64) std::atomic<size_t> head_{0};  // producer-owned
  alignas(64) std::atomic<size_t> tail_{0};  // consumer-owned, separate cache line
};

}  // namespace frameforge
