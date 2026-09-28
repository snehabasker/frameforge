// Queue correctness under contention (CLAUDE.md 7.4), including
// multi-producer/multi-consumer stress for BlockingQueue and the intended
// single-producer/single-consumer stress for SpscRingBuffer. Run this
// binary under ThreadSanitizer (the cpp-tsan CI job, or locally with
// -DFRAMEFORGE_ENABLE_TSAN=ON) to catch any data race, not just the
// functional correctness checked here.
#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "frameforge/queue.hpp"

// ---------- BlockingQueue ----------

TEST(BlockingQueue, SingleThreadFifoOrder) {
  frameforge::BlockingQueue<int> q(4);
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(q.Push(i));
  for (int i = 0; i < 4; ++i) {
    auto v = q.Pop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, i);
  }
}

TEST(BlockingQueue, CloseWakesBlockedPopAndReturnsNullopt) {
  frameforge::BlockingQueue<int> q(1);
  std::thread popper([&] {
    auto v = q.Pop();
    EXPECT_FALSE(v.has_value());
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  q.Close();
  popper.join();
}

TEST(BlockingQueue, CloseWakesBlockedPushAndReturnsFalse) {
  frameforge::BlockingQueue<int> q(1);
  ASSERT_TRUE(q.Push(1));  // fill it
  std::atomic<bool> push_returned{false};
  std::thread pusher([&] {
    bool ok = q.Push(2);  // blocks: queue full
    EXPECT_FALSE(ok);
    push_returned = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  q.Close();
  pusher.join();
  EXPECT_TRUE(push_returned.load());
}

TEST(BlockingQueue, ContentionStressPreservesAllItems) {
  constexpr int kProducers = 4;
  constexpr int kConsumers = 4;
  constexpr int kItemsPerProducer = 2000;
  frameforge::BlockingQueue<int> q(16);

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < kItemsPerProducer; ++i) {
        q.Push(p * kItemsPerProducer + i);
      }
    });
  }

  std::atomic<long long> sum{0};
  std::atomic<int> count{0};
  std::vector<std::thread> consumers;
  for (int c = 0; c < kConsumers; ++c) {
    consumers.emplace_back([&] {
      while (auto v = q.Pop()) {
        sum += *v;
        ++count;
      }
    });
  }

  for (auto& t : producers) t.join();
  q.Close();
  for (auto& t : consumers) t.join();

  long long expected_sum = 0;
  for (int i = 0; i < kProducers * kItemsPerProducer; ++i) expected_sum += i;

  EXPECT_EQ(count.load(), kProducers * kItemsPerProducer);
  EXPECT_EQ(sum.load(), expected_sum);
}

// ---------- SpscRingBuffer ----------

TEST(SpscRingBuffer, CapacityRoundsUpToPowerOfTwo) {
  frameforge::SpscRingBuffer<int> rb(5);
  EXPECT_EQ(rb.Capacity(), 8u);
}

TEST(SpscRingBuffer, FifoOrderSingleThread) {
  frameforge::SpscRingBuffer<int> rb(4);
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(rb.TryPush(i));
  EXPECT_FALSE(rb.TryPush(99));  // full
  for (int i = 0; i < 4; ++i) {
    auto v = rb.TryPop();
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, i);
  }
  EXPECT_FALSE(rb.TryPop().has_value());  // empty
}

TEST(SpscRingBuffer, ProducerConsumerStressPreservesOrderAndCount) {
  constexpr int kItems = 200000;
  frameforge::SpscRingBuffer<int> rb(256);

  std::thread producer([&] {
    int i = 0;
    while (i < kItems) {
      if (rb.TryPush(i)) ++i;
    }
  });

  std::vector<int> received;
  received.reserve(kItems);
  std::thread consumer([&] {
    while (static_cast<int>(received.size()) < kItems) {
      if (auto v = rb.TryPop()) {
        received.push_back(*v);
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(static_cast<int>(received.size()), kItems);
  for (int i = 0; i < kItems; ++i) {
    ASSERT_EQ(received[i], i) << "order violated at index " << i;
  }
}
