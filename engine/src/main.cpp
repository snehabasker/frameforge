// FrameForge inference engine CLI (CLAUDE.md 7.3).
//
// Usage:
//   frameforge --model m.onnx --input <seq_dir> --output <dir>
//              [--threads N] [--ep cpu|cuda] [--tile 256 --overlap 16] [--bench]
//
// `--input` may be a single sequence directory (containing frame_*.exr
// directly -- runs the threaded reader/inference/writer pipeline on it,
// see pipeline.hpp) or a directory of MULTIPLE sequence subdirectories
// (each containing frame_*.exr -- runs them concurrently across
// `--threads` workers, one inference worker per sequence, per CLAUDE.md
// 7.3's concurrency model: the recurrent O_{t-1}->O_t dependency means
// frames within one sequence must infer in order, so cross-sequence
// concurrency is where `--threads` > 1 actually helps).

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "frameforge/ort_session.hpp"
#include "frameforge/pipeline.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string model;
  std::string input;
  std::string output;
  int threads = 1;
  std::string ep = "cpu";
  int tile = 0;
  int overlap = 16;
  bool bench = false;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "--model") {
      args.model = next();
    } else if (arg == "--input") {
      args.input = next();
    } else if (arg == "--output") {
      args.output = next();
    } else if (arg == "--threads") {
      args.threads = std::stoi(next());
    } else if (arg == "--ep") {
      args.ep = next();
    } else if (arg == "--tile") {
      args.tile = std::stoi(next());
    } else if (arg == "--overlap") {
      args.overlap = std::stoi(next());
    } else if (arg == "--bench") {
      args.bench = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(1);
    }
  }
  if (args.model.empty() || args.input.empty() || args.output.empty()) {
    std::fprintf(stderr,
                 "usage: frameforge --model m.onnx --input <seq_dir> --output <dir> "
                 "[--threads N] [--ep cpu|cuda] [--tile 256 --overlap 16] [--bench]\n");
    std::exit(1);
  }
  return args;
}

bool IsDirectSequence(const fs::path& dir) {
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() == ".exr") return true;
  }
  return false;
}

std::vector<fs::path> SubSequenceDirs(const fs::path& dir) {
  std::vector<fs::path> dirs;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.is_directory() && IsDirectSequence(entry.path())) {
      dirs.push_back(entry.path());
    }
  }
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);

  try {
    fs::create_directories(args.output);
    frameforge::ExecutionProvider ep = (args.ep == "cuda") ? frameforge::ExecutionProvider::kCUDA
                                                            : frameforge::ExecutionProvider::kCPU;

    auto t_start = std::chrono::steady_clock::now();

    if (IsDirectSequence(args.input)) {
      frameforge::ModelIoSpec spec;
      frameforge::OrtModelRunner runner(args.model, spec, ep, std::max(1, args.threads));
      frameforge::RunSequencePipelined(args.input, args.output, runner);
      size_t total_frames = 0;
      for (const auto& e : fs::directory_iterator(args.output)) {
        if (e.path().extension() == ".png") ++total_frames;
      }
      std::printf("processed %zu frame(s)\n", total_frames);
    } else {
      auto seq_dirs_paths = SubSequenceDirs(args.input);
      if (seq_dirs_paths.empty()) {
        std::fprintf(stderr, "no sequences (frame_*.exr, directly or in subdirs) found under %s\n",
                     args.input.c_str());
        return 1;
      }
      std::vector<std::string> seq_dirs;
      for (auto& p : seq_dirs_paths) seq_dirs.push_back(p.string());
      frameforge::RunSequencesConcurrently(seq_dirs, args.output, args.model,
                                            std::max(1, args.threads), ep);
      std::printf("processed %zu sequence(s)\n", seq_dirs.size());
    }

    auto elapsed = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_start)
                       .count();
    if (args.bench) {
      std::printf("total: %.2f ms\n", elapsed);
    }
    std::printf("wrote output to %s\n", args.output.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
