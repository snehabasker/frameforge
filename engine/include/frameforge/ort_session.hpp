// RAII wrapper around the ONNX Runtime C++ API for FrameForgeNet inference.
//
// The `#define _stdcall` below neutralizes a real MinGW/GCC incompatibility
// in onnxruntime_c_api.h: the header unconditionally uses `_stdcall` (an
// MSVC-only spelling GCC doesn't define) under `#ifdef _WIN32`, regardless
// of compiler. x64 has one calling convention, so this is a no-op fix, not
// a behavior change -- confirmed by a real compile+link+run round trip
// against the official prebuilt onnxruntime.dll (see docs/DECISIONS.md).
// Harmless (and unnecessary but not wrong) on non-Windows.
#pragma once

#if defined(_WIN32) && !defined(_MSC_VER) && !defined(_stdcall)
#define _stdcall
#endif

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

#include "frameforge/tensor.hpp"

namespace frameforge {

enum class ExecutionProvider { kCPU, kCUDA };

// Which of FrameForgeNet's optional inputs a given exported model needs
// (mirrors python/frameforge/models/student.py's VARIANT_CONFIGS).
// `use_depth`/`use_temporal` are overwritten by OrtModelRunner's
// constructor from the loaded model's actual input names -- pass whatever
// here, only `scale` is taken at face value (ONNX doesn't expose it
// directly; FrameForge is always x2 per spec, so the default is correct
// unless a future variant changes it).
struct ModelIoSpec {
  bool use_depth = true;
  bool use_temporal = true;
  int scale = 2;
};

class OrtModelRunner {
 public:
  OrtModelRunner(const std::string& onnx_path, const ModelIoSpec& spec,
                 ExecutionProvider ep = ExecutionProvider::kCPU, int intra_op_threads = 1);

  // Runs one recurrent step. `prev_hr_output` may be null (as a Tensor with
  // n()==0) only if `spec.use_temporal` is false; when temporal, the caller
  // must supply a concrete tensor (bilinear-upsampled lr_rgb for the first
  // frame) -- see export_onnx.py's docstring for why the ONNX graph can't
  // do that substitution itself.
  Tensor Run(const Tensor& lr_rgb, const Tensor* lr_depth, const Tensor* lr_mv,
             const Tensor* prev_hr_output);

  const ModelIoSpec& spec() const { return spec_; }

 private:
  Ort::Value MakeInputTensor(const Tensor& t);

  ModelIoSpec spec_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  std::vector<std::string> input_names_;
  std::vector<const char*> input_names_c_;
};

}  // namespace frameforge
