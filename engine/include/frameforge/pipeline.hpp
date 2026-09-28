// Threaded reader -> inference -> writer pipeline (CLAUDE.md 7.3).
//
// Concurrency model (see docs/ARCHITECTURE.md for the full reasoning):
// the recurrent O_{t-1} -> O_t dependency means frames WITHIN one sequence
// must be inferred strictly in order -- there is no parallelism to extract
// there. Parallelism instead comes from two places: (1) overlapping the
// three pipeline STAGES within a sequence (reading frame t+1's EXR while
// inferring frame t while writing frame t-1's PNG), which
// RunSequencePipelined does; and (2) running independent SEQUENCES
// concurrently, one inference worker per sequence, which
// RunSequencesConcurrently does with a `--threads`-sized worker pool.
//
// Queue choice: SpscRingBuffer, the clear winner in
// engine/bench/queue_bench.cpp (~5.6x the throughput of BlockingQueue at
// 100k items -- see docs/RESULTS.md), per CLAUDE.md 7.3's "keep the better
// one as default" instruction.
#pragma once

#include <string>
#include <vector>

#include "frameforge/exr_io.hpp"
#include "frameforge/ort_session.hpp"
#include "frameforge/tensor.hpp"

namespace frameforge {

// Runs one sequence's reader/inference/writer stages on three threads
// connected by SpscRingBuffer queues. Produces byte-identical output to
// the sequential frameforge::RunSequence (same math, just overlapped) --
// see tests/pipeline_test.cpp's equivalence check.
void RunSequencePipelined(const std::string& seq_dir, const std::string& out_dir,
                           OrtModelRunner& runner, const ChannelMap& channel_map = ChannelMap{});

// Processes multiple sequences concurrently using `num_threads` workers,
// each running RunSequencePipelined on one sequence at a time. Each worker
// gets its own OrtModelRunner (FrameForge models are <300K params, so
// duplicating one per thread is cheap) rather than sharing a session
// across threads, sidestepping any question of ONNX Runtime's
// concurrent-Run() thread-safety guarantees for this project's purposes.
void RunSequencesConcurrently(const std::vector<std::string>& seq_dirs,
                               const std::string& out_root, const std::string& onnx_path,
                               int num_threads, ExecutionProvider ep,
                               const ChannelMap& channel_map = ChannelMap{});

}  // namespace frameforge
