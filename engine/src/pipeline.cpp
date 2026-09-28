#include "frameforge/pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <optional>
#include <thread>
#include <vector>

#include "frameforge/preprocess.hpp"
#include "frameforge/png_io.hpp"
#include "frameforge/queue.hpp"

namespace fs = std::filesystem;

namespace frameforge {

namespace {

std::vector<fs::path> ListFrames(const fs::path& dir) {
  std::vector<fs::path> frames;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".exr") frames.push_back(entry.path());
  }
  std::sort(frames.begin(), frames.end());
  return frames;
}

// Minimal blocking wrapper around the lock-free SpscRingBuffer (the
// benchmark winner, engine/bench/queue_bench.cpp) for pipeline use: a
// short yield-spin while full/empty. Real per-item work here (EXR decode,
// ONNX inference, PNG encode) is milliseconds, so a brief spin before
// yielding costs nothing relative to that -- unlike queue_bench.cpp's
// microbenchmark, where avoiding syscalls is the whole point.
template <typename T>
class BlockingSpsc {
 public:
  explicit BlockingSpsc(size_t capacity) : ring_(capacity) {}

  void Push(T item) {
    while (!ring_.TryPush(std::move(item))) {
      std::this_thread::yield();
    }
  }
  void Close() { closed_.store(true, std::memory_order_release); }

  // Returns std::nullopt only once Close() has been called AND the ring is
  // drained.
  std::optional<T> Pop() {
    while (true) {
      if (auto v = ring_.TryPop()) return v;
      if (closed_.load(std::memory_order_acquire)) {
        auto v = ring_.TryPop();
        if (v) return v;
        return std::nullopt;
      }
      std::this_thread::yield();
    }
  }

 private:
  SpscRingBuffer<T> ring_;
  std::atomic<bool> closed_{false};
};

struct ReadyFrame {
  size_t index;
  Tensor lr_rgb;
  Tensor lr_depth;   // already normalized
  Tensor lr_mv;      // zeroed for frame 0
  bool use_depth;
  bool use_temporal;
};

struct InferredFrame {
  size_t index;
  Tensor output;
};

}  // namespace

void RunSequencePipelined(const std::string& seq_dir, const std::string& out_dir,
                           OrtModelRunner& runner, const ChannelMap& channel_map) {
  fs::create_directories(out_dir);
  auto frame_paths = ListFrames(seq_dir);
  if (frame_paths.empty()) {
    throw std::runtime_error("no frame_*.exr files found in " + seq_dir);
  }
  const size_t n = frame_paths.size();

  BlockingSpsc<ReadyFrame> reader_to_inference(4);
  BlockingSpsc<InferredFrame> inference_to_writer(4);
  std::exception_ptr reader_exception, inference_exception, writer_exception;

  // Reader: decodes every EXR + tonemaps + per-sequence-normalizes depth.
  // This whole phase runs before the first item is pushed, because
  // per-sequence depth normalization (matching
  // frameforge.colorspace.normalize_depth) needs every frame's depth
  // range up front -- a real data dependency, not a missed optimization;
  // see docs/ARCHITECTURE.md for the full reasoning. Inference and writer
  // still genuinely overlap with EACH OTHER below.
  std::thread reader([&] {
    try {
      std::vector<RawFramePasses> raw(n);
      for (size_t i = 0; i < n; ++i) raw[i] = LoadFramePasses(frame_paths[i].string(), channel_map);

      std::vector<Tensor> depths;
      depths.reserve(n);
      for (auto& f : raw) depths.push_back(f.depth);
      NormalizeDepthInPlace(depths);

      for (size_t i = 0; i < n; ++i) {
        ReadyFrame rf;
        rf.index = i;
        rf.lr_rgb = ApplyTonemap(raw[i].combined);
        rf.lr_depth = depths[i];
        rf.lr_mv = raw[i].vector_to_previous;
        if (i == 0) std::fill(rf.lr_mv.data(), rf.lr_mv.data() + rf.lr_mv.size(), 0.0f);
        rf.use_depth = runner.spec().use_depth;
        rf.use_temporal = runner.spec().use_temporal;
        reader_to_inference.Push(std::move(rf));
      }
    } catch (...) {
      reader_exception = std::current_exception();
    }
    reader_to_inference.Close();
  });

  // Inference: strictly sequential across frames (the recurrent
  // dependency), but overlaps with the writer thread below.
  std::thread inference([&] {
    try {
      Tensor prev_hr_output;
      bool have_prev = false;
      while (auto rf = reader_to_inference.Pop()) {
        Tensor* depth_ptr = rf->use_depth ? &rf->lr_depth : nullptr;
        Tensor* mv_ptr = nullptr;
        Tensor* prev_ptr = nullptr;
        if (rf->use_temporal) {
          mv_ptr = &rf->lr_mv;
          if (!have_prev) prev_hr_output = BilinearUpsample2x(rf->lr_rgb);
          prev_ptr = &prev_hr_output;
        }
        Tensor output = runner.Run(rf->lr_rgb, depth_ptr, mv_ptr, prev_ptr);
        prev_hr_output = output;
        have_prev = true;
        inference_to_writer.Push(InferredFrame{rf->index, std::move(output)});
      }
    } catch (...) {
      inference_exception = std::current_exception();
    }
    inference_to_writer.Close();
  });

  std::thread writer([&] {
    try {
      while (auto inf = inference_to_writer.Pop()) {
        char name_buf[64];
        std::snprintf(name_buf, sizeof(name_buf), "frame_%04zu.png", inf->index);
        WritePng((fs::path(out_dir) / name_buf).string(), inf->output);
      }
    } catch (...) {
      writer_exception = std::current_exception();
    }
  });

  reader.join();
  inference.join();
  writer.join();

  // Surface the FIRST exception hit, in stage order -- CLAUDE.md 7.3: "no
  // exceptions crossing thread boundaries silently."
  if (reader_exception) std::rethrow_exception(reader_exception);
  if (inference_exception) std::rethrow_exception(inference_exception);
  if (writer_exception) std::rethrow_exception(writer_exception);
}

void RunSequencesConcurrently(const std::vector<std::string>& seq_dirs,
                               const std::string& out_root, const std::string& onnx_path,
                               int num_threads, ExecutionProvider ep,
                               const ChannelMap& channel_map) {
  std::atomic<size_t> next_index{0};
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> worker_exceptions(static_cast<size_t>(num_threads));

  for (int w = 0; w < num_threads; ++w) {
    workers.emplace_back([&, w] {
      try {
        ModelIoSpec spec;
        OrtModelRunner runner(onnx_path, spec, ep, 1);
        while (true) {
          size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
          if (i >= seq_dirs.size()) break;
          fs::path seq_path(seq_dirs[i]);
          fs::path out_dir = fs::path(out_root) / seq_path.filename();
          RunSequencePipelined(seq_dirs[i], out_dir.string(), runner, channel_map);
        }
      } catch (...) {
        worker_exceptions[static_cast<size_t>(w)] = std::current_exception();
      }
    });
  }
  for (auto& t : workers) t.join();
  for (auto& e : worker_exceptions) {
    if (e) std::rethrow_exception(e);
  }
}

}  // namespace frameforge
