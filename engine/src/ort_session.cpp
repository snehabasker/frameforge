#include "frameforge/ort_session.hpp"

#include <algorithm>
#include <stdexcept>

namespace frameforge {

namespace {

std::wstring ToWide(const std::string& s) {
  return std::wstring(s.begin(), s.end());  // ASCII-safe paths only, sufficient here
}

}  // namespace

OrtModelRunner::OrtModelRunner(const std::string& onnx_path, const ModelIoSpec& spec,
                                ExecutionProvider ep, int intra_op_threads)
    : spec_(spec),
      env_(ORT_LOGGING_LEVEL_WARNING, "frameforge"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  session_options_.SetIntraOpNumThreads(intra_op_threads);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (ep == ExecutionProvider::kCUDA) {
    // Optional, behind a flag (CLAUDE.md 7.1); requires the CUDA EP shared
    // libs to be present alongside onnxruntime.dll/.so at runtime. Falls
    // through to a clear exception if the provider can't be appended
    // rather than silently running on CPU.
    OrtCUDAProviderOptions cuda_options{};
    session_options_.AppendExecutionProvider_CUDA(cuda_options);
  }

#ifdef _WIN32
  session_ = std::make_unique<Ort::Session>(env_, ToWide(onnx_path).c_str(), session_options_);
#else
  session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options_);
#endif

  Ort::AllocatorWithDefaultOptions allocator;
  size_t n_inputs = session_->GetInputCount();
  input_names_.reserve(n_inputs);
  for (size_t i = 0; i < n_inputs; ++i) {
    auto name = session_->GetInputNameAllocated(i, allocator);
    input_names_.emplace_back(name.get());
  }
  input_names_c_.reserve(n_inputs);
  for (auto& n : input_names_) input_names_c_.push_back(n.c_str());

  // Auto-detect use_depth/use_temporal from the model's actual input names
  // rather than trusting the caller to get it right -- the exported graph
  // itself is the ground truth (export_onnx.py names inputs exactly
  // "lr_depth"/"lr_mv"/"prev_hr_output" when present).
  auto has_input = [&](const char* name) {
    return std::find(input_names_.begin(), input_names_.end(), name) != input_names_.end();
  };
  spec_.use_depth = has_input("lr_depth");
  spec_.use_temporal = has_input("lr_mv");
}

Ort::Value OrtModelRunner::MakeInputTensor(const Tensor& t) {
  auto shape = t.shape();
  return Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(t.data()), t.size(),
                                          shape.data(), shape.size());
}

Tensor OrtModelRunner::Run(const Tensor& lr_rgb, const Tensor* lr_depth, const Tensor* lr_mv,
                            const Tensor* prev_hr_output) {
  std::vector<Ort::Value> inputs;
  inputs.push_back(MakeInputTensor(lr_rgb));

  if (spec_.use_depth) {
    if (lr_depth == nullptr) {
      throw std::invalid_argument("model requires lr_depth but none was given");
    }
    inputs.push_back(MakeInputTensor(*lr_depth));
  }
  if (spec_.use_temporal) {
    if (lr_mv == nullptr || prev_hr_output == nullptr) {
      throw std::invalid_argument("model requires lr_mv and prev_hr_output but got null");
    }
    inputs.push_back(MakeInputTensor(*lr_mv));
    inputs.push_back(MakeInputTensor(*prev_hr_output));
  }

  if (inputs.size() != input_names_c_.size()) {
    throw std::runtime_error("input count mismatch: model expects " +
                              std::to_string(input_names_c_.size()) + ", spec provided " +
                              std::to_string(inputs.size()) +
                              " (check ModelIoSpec matches the exported variant)");
  }

  const char* output_names[] = {"output"};
  auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_c_.data(), inputs.data(),
                                inputs.size(), output_names, 1);

  auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  Tensor result(static_cast<int>(out_shape[0]), static_cast<int>(out_shape[1]),
                static_cast<int>(out_shape[2]), static_cast<int>(out_shape[3]));
  const float* out_data = outputs[0].GetTensorData<float>();
  std::copy(out_data, out_data + result.size(), result.data());
  return result;
}

}  // namespace frameforge
