/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/tools/benchmark/benchmark_tflite_model.h"

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <vector>

#include "absl/base/attributes.h"
#include "ruy/profiler/profiler.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/cpu.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/profiling/platform_profiler.h"
#include "tensorflow/lite/profiling/profile_summary_formatter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/profiling_listener.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/tools/logging_reporter.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

// Version with Weak linker attribute doing nothing: if someone links this
// library with another definition of this function (presumably to actually
// register custom ops), that version will be used instead.
void ABSL_ATTRIBUTE_WEAK
RegisterSelectedOps(::tflite::MutableOpResolver* resolver) {}

using tflite::impl::SubgraphKey;

namespace tflite {
namespace benchmark {
namespace {

// Backward compat with previous approach to enabling op profiling.
#if defined(TFLITE_PROFILING_ENABLED)
constexpr int kOpProfilingEnabledDefault = true;
#else
constexpr int kOpProfilingEnabledDefault = false;
#endif

// Dumps platform-wide tracing files via a platform-based profiler that's built
// upon platform tracing tools, like ATrace on Android etc.
class PlatformProfilingListener : public BenchmarkListener {
 public:
  explicit PlatformProfilingListener(Interpreter* interpreter) {
    TFLITE_TOOLS_CHECK(interpreter);
    platform_profiler_ = profiling::CreatePlatformProfiler();
    interpreter->SetProfiler(platform_profiler_.get());
  }

 private:
  std::unique_ptr<tflite::Profiler> platform_profiler_;
};

// Dumps ruy profiling events if the ruy profiler is enabled.
class RuyProfileListener : public BenchmarkListener {
 public:
  void OnBenchmarkStart(const BenchmarkParams& params) override;

  void OnBenchmarkEnd(const BenchmarkResults& results) override;

 private:
  std::unique_ptr<ruy::profiler::ScopeProfile> ruy_profile_;
};

void RuyProfileListener::OnBenchmarkStart(const BenchmarkParams& params) {
  ruy_profile_.reset(new ruy::profiler::ScopeProfile);
}

void RuyProfileListener::OnBenchmarkEnd(const BenchmarkResults& results) {
  ruy_profile_ = nullptr;
}

int GetNumElements(const TfLiteIntArray* dim_array) {
  int num_elements = 1;
  for (size_t i = 0; i < dim_array->size; i++) {
    num_elements *= dim_array->data[i];
  }
  return num_elements;
}

void FillRandomString(tflite::DynamicBuffer* buffer,
                      const TfLiteIntArray* dim_array,
                      const std::function<std::string()>& random_func) {
  int num_elements = GetNumElements(dim_array);
  for (int i = 0; i < num_elements; ++i) {
    auto str = random_func();
    buffer->AddString(str.data(), str.length());
  }
}

std::shared_ptr<profiling::ProfileSummaryFormatter>
CreateProfileSummaryFormatter(bool format_as_csv) {
  return format_as_csv
             ? std::make_shared<profiling::ProfileSummaryCSVFormatter>()
             : std::make_shared<profiling::ProfileSummaryDefaultFormatter>();
}

// https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
inline bool FileExists(const std::string& name) {
  struct stat buffer;
  return stat(name.c_str(), &buffer) == 0;
}

}  // namespace

BenchmarkParams BenchmarkTfLiteModel::DefaultParams() {
  BenchmarkParams default_params = BenchmarkModel::DefaultParams();
  default_params.AddParam("allow_fp16", BenchmarkParam::Create<bool>(false));
  default_params.AddParam("require_full_delegation",
                          BenchmarkParam::Create<bool>(false));
  default_params.AddParam(
      "enable_op_profiling",
      BenchmarkParam::Create<bool>(kOpProfilingEnabledDefault));
  default_params.AddParam("max_profiling_buffer_entries",
                          BenchmarkParam::Create<int32_t>(1024));
  default_params.AddParam("profiling_output_csv_file",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("enable_platform_tracing",
                          BenchmarkParam::Create<bool>(false));

  for (const auto& delegate_provider :
       tools::GetRegisteredDelegateProviders()) {
    default_params.Merge(delegate_provider->DefaultParams());
  }

  return default_params;
}

BenchmarkTfLiteModel::BenchmarkTfLiteModel(BenchmarkParams params)
    : BenchmarkModel(std::move(params)),
      random_engine_(std::random_device()()) {
  AddListener(&log_output_);
}

void BenchmarkTfLiteModel::CleanUp() {
  // set this flag in case we had a abrupt shutdown
  kill_app_ = true;
  for (int i = 0; i < runtime_config_.model_information.size(); i++) {
    // Free up any pre-allocated tensor data during PrepareInputData.
    runtime_config_.model_information[i].input_tensor_data.clear();
  }
}

BenchmarkTfLiteModel::~BenchmarkTfLiteModel() { CleanUp(); }

std::vector<Flag> BenchmarkTfLiteModel::GetFlags() {
  std::vector<Flag> flags = BenchmarkModel::GetFlags();
  std::vector<Flag> specific_flags = {
      CreateFlag<bool>("allow_fp16", &params_, "allow fp16"),
      CreateFlag<bool>("require_full_delegation", &params_,
                       "require delegate to run the entire graph"),
      CreateFlag<bool>("enable_op_profiling", &params_, "enable op profiling"),
      CreateFlag<int32_t>("max_profiling_buffer_entries", &params_,
                          "max profiling buffer entries"),
      CreateFlag<std::string>(
          "profiling_output_csv_file", &params_,
          "File path to export profile data as CSV, if not set "
          "prints to stdout."),
      CreateFlag<bool>("enable_platform_tracing", &params_,
                       "enable platform-wide tracing, only meaningful when "
                       "--enable_op_profiling is set to true.")};

  flags.insert(flags.end(), specific_flags.begin(), specific_flags.end());

  for (const auto& delegate_provider :
       tools::GetRegisteredDelegateProviders()) {
    auto delegate_flags = delegate_provider->CreateFlags(&params_);
    flags.insert(flags.end(), delegate_flags.begin(), delegate_flags.end());
  }

  return flags;
}

void BenchmarkTfLiteModel::LogParams() {
  BenchmarkModel::LogParams();
  TFLITE_LOG(INFO) << "Allow fp16 : [" << params_.Get<bool>("allow_fp16")
                   << "]";
  TFLITE_LOG(INFO) << "Require full delegation : ["
                   << params_.Get<bool>("require_full_delegation") << "]";
  TFLITE_LOG(INFO) << "Enable op profiling: ["
                   << params_.Get<bool>("enable_op_profiling") << "]";
  TFLITE_LOG(INFO) << "Max profiling buffer entries: ["
                   << params_.Get<int32_t>("max_profiling_buffer_entries")
                   << "]";
  TFLITE_LOG(INFO) << "CSV File to export profiling data to: ["
                   << params_.Get<std::string>("profiling_output_csv_file")
                   << "]";
  TFLITE_LOG(INFO) << "Enable platform-wide tracing: ["
                   << params_.Get<bool>("enable_platform_tracing") << "]";

  for (const auto& delegate_provider :
       tools::GetRegisteredDelegateProviders()) {
    delegate_provider->LogParams(params_);
  }
}

TfLiteStatus BenchmarkTfLiteModel::ValidateParams() {
  if (params_.Get<std::string>("json_path").empty()) {
    TFLITE_LOG(ERROR)
        << "Please specify the name of the config file with --json_path";
    return kTfLiteError;
  }

  return kTfLiteOk;
}

uint64_t BenchmarkTfLiteModel::ComputeInputBytes() {
  TFLITE_TOOLS_CHECK(interpreter_);
  uint64_t total_input_bytes = 0;
  
  for (int i = 0; i < runtime_config_.model_information.size(); ++i) {
    int subgraph_index = 
        interpreter_->GetSubgraphIdx(i, kTfLiteCPU);
    for (int input : interpreter_->inputs(subgraph_index)) {
      auto* t = interpreter_->tensor(subgraph_index, input);
      total_input_bytes += t->bytes;
    }
  }
  return total_input_bytes;
}

int64_t BenchmarkTfLiteModel::MayGetModelFileSize() {
  int64_t total_mem_size = 0;
  auto& model_information = runtime_config_.model_information;
  for (int i = 0; i < model_information.size(); ++i) {
    std::ifstream in_file(model_information[i].config.model_fname,
                          std::ios::binary | std::ios::ate);
    total_mem_size += in_file.tellg();
  }

  return total_mem_size;
}

util::InputTensorData BenchmarkTfLiteModel::LoadInputTensorData(
    const TfLiteTensor& t, const std::string& input_file_path) {
  std::ifstream value_file(input_file_path, std::ios::binary);
  if (!value_file.good()) {
    TFLITE_LOG(FATAL) << "Failed to read the input_layer_value_file:"
                      << input_file_path;
  }
  util::InputTensorData t_data;
  if (t.type == kTfLiteString) {
    t_data.data = util::VoidUniquePtr(
        static_cast<void*>(new tflite::DynamicBuffer()),
        [](void* ptr) { delete static_cast<DynamicBuffer*>(ptr); });
    std::string line;
    size_t num_line = 0;
    // Read the line with the delimiter '\0'.
    while (std::getline(value_file, line, '\0')) {
      num_line++;
      static_cast<DynamicBuffer*>(t_data.data.get())
          ->AddString(line.data(), line.length());
    }
    int num_elements = GetNumElements(t.dims);
    if (num_line != num_elements) {
      TFLITE_LOG(FATAL) << "The number of string in the input_layer_value_file("
                        << input_file_path << ") is " << num_line
                        << ". It should be " << num_elements << ".";
    }
  } else {
    value_file.seekg(0, std::ios_base::end);
    if (value_file.tellg() != t.bytes) {
      TFLITE_LOG(FATAL) << "The size of " << input_file_path << " is "
                        << value_file.tellg() << " bytes. It should be "
                        << t.bytes << " bytes.";
    }
    t_data.bytes = t.bytes;
    t_data.data =
        util::VoidUniquePtr(static_cast<void*>(new char[t.bytes]),
                      [](void* ptr) { delete[] static_cast<char*>(ptr); });
    value_file.clear();
    value_file.seekg(0, std::ios_base::beg);
    value_file.read(static_cast<char*>(t_data.data.get()), t.bytes);
  }
  return t_data;
}

util::InputTensorData
BenchmarkTfLiteModel::CreateRandomTensorData(const TfLiteTensor& t,
                                             const util::InputLayerInfo* layer_info) {
  bool has_value_range = false;
  int low_range = 0;
  int high_range = 0;
  if (layer_info) {
    has_value_range = layer_info->has_value_range;
    low_range = layer_info->low;
    high_range = layer_info->high;
  }
  int num_elements = GetNumElements(t.dims);
  switch (t.type) {
    case kTfLiteFloat32: {
      return CreateInputTensorData<float>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
    }
    case kTfLiteFloat16: {
      // TODO(b/138843274): Remove this preprocessor guard when bug is fixed.
#if TFLITE_ENABLE_FP16_CPU_BENCHMARKS
#if __GNUC__ && \
    (__clang__ || __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE)
      // __fp16 is available on Clang or when __ARM_FP16_FORMAT_* is defined.
      return CreateInputTensorData<__fp16>(
          num_elements, std::uniform_real_distribution<float>(-0.5f, 0.5f));
#else
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t->name
                        << " of type FLOAT16 on this platform.";
#endif
#else
      // You need to build with -DTFLITE_ENABLE_FP16_CPU_BENCHMARKS=1 using a
      // compiler that supports __fp16 type. Note: when using Clang and *not*
      // linking with compiler-rt, a definition of __gnu_h2f_ieee and
      // __gnu_f2h_ieee must be supplied.
      TFLITE_LOG(FATAL) << "Populating the tensor " << t.name
                        << " of type FLOAT16 is disabled.";
#endif  // TFLITE_ENABLE_FP16_CPU_BENCHMARKS
      break;
    }
    case kTfLiteFloat64: {
      return CreateInputTensorData<double>(
          num_elements, std::uniform_real_distribution<double>(-0.5, 0.5));
    }
    case kTfLiteInt64: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      return CreateInputTensorData<int64_t>(
          num_elements, std::uniform_int_distribution<int64_t>(low, high));
    }
    case kTfLiteInt32: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      return CreateInputTensorData<int32_t>(
          num_elements, std::uniform_int_distribution<int32_t>(low, high));
    }
    case kTfLiteInt16: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 99;
      return CreateInputTensorData<int16_t>(
          num_elements, std::uniform_int_distribution<int16_t>(low, high));
    }
    case kTfLiteUInt8: {
      int low = has_value_range ? low_range : 0;
      int high = has_value_range ? high_range : 254;
      // std::uniform_int_distribution is specified not to support char types.
      return CreateInputTensorData<uint8_t>(
          num_elements, std::uniform_int_distribution<uint32_t>(low, high));
    }
    case kTfLiteInt8: {
      int low = has_value_range ? low_range : -127;
      int high = has_value_range ? high_range : 127;
      // std::uniform_int_distribution is specified not to support char types.
      return CreateInputTensorData<int8_t>(
          num_elements, std::uniform_int_distribution<int32_t>(low, high));
    }
    case kTfLiteString: {
      // TODO(haoliang): No need to cache string tensors right now.
      break;
    }
    default: {
      TFLITE_LOG(FATAL) << "Don't know how to populate tensor " << t.name
                        << " of type " << t.type;
    }
  }
  return util::InputTensorData();
}

TfLiteStatus BenchmarkTfLiteModel::PrepareInputData() {
  CleanUp();

  for (int i = 0; i < runtime_config_.model_information.size(); ++i) {
    // Note the corresponding relation between 'interpreter_inputs' and 'inputs_'
    // (i.e. the specified input layer info) has been checked in
    // BenchmarkTfLiteModel::Init() before calling this function. So, we simply
    // use the corresponding input layer info to initializethe input data value
    // properly.
    auto subgraph_index = 
        interpreter_->GetSubgraphIdx(i, kTfLiteCPU);
    auto interpreter_inputs = interpreter_->inputs(subgraph_index);
    auto& input_layer_infos = runtime_config_.model_information[i].input_layer_infos;
    auto& input_tensor_data = runtime_config_.model_information[i].input_tensor_data;
    for (int j = 0; j < interpreter_inputs.size(); ++j) {
      int tensor_index = interpreter_inputs[j];
      const TfLiteTensor* t = interpreter_->tensor(subgraph_index, tensor_index);
      const util::InputLayerInfo* input_layer_info = nullptr;
      // Note that when input layer parameters (i.e. --input_layer,
      // --input_layer_shape) are not specified, inputs_ is empty.
      if (!input_layer_infos.empty()) input_layer_info = &input_layer_infos[j];

      util::InputTensorData t_data;
      if (input_layer_info && !input_layer_info->input_file_path.empty()) {
        t_data = LoadInputTensorData(*t, input_layer_info->input_file_path);
      } else {
        t_data = CreateRandomTensorData(*t, input_layer_info);
      }
      input_tensor_data.push_back(std::move(t_data));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::ResetInputsAndOutputs() {
  for (int model_id = 0; model_id < runtime_config_.model_information.size(); ++model_id) {
    auto& input_layer_infos = runtime_config_.model_information[model_id].input_layer_infos;
    auto& input_tensor_data = runtime_config_.model_information[model_id].input_tensor_data;

    // TODO: #73 share tensors across different subgraphs from same model
    for (int device_id = 0; device_id < kTfLiteNumDevices; ++device_id) {
      TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_id);

      // reset inputs for all subgraphs that start with op 0
      // TODO: may need to do this for all subgraphs that require external inputs
      for (int subgraph_index : interpreter_->GetSubgraphIdx(model_id,
                                                             device_flag, 0)) {
        auto interpreter_inputs = interpreter_->inputs(subgraph_index);
        // Set the values of the input tensors from inputs_data_.
        for (int j = 0; j < interpreter_inputs.size(); ++j) {
          int i = interpreter_inputs[j];
          TfLiteTensor* t = interpreter_->tensor(subgraph_index, i);
          if (t->type == kTfLiteString) {
            if (input_tensor_data[j].data) {
              static_cast<DynamicBuffer*>(input_tensor_data[j].data.get())
                  ->WriteToTensor(t, /*new_shape=*/nullptr);
            } else {
              tflite::DynamicBuffer buffer;
              FillRandomString(&buffer, t->dims, []() {
                return "we're have some friends over saturday to hang out in the "
                      "yard";
              });
              buffer.WriteToTensor(t, /*new_shape=*/nullptr);
            }
          } else {
            std::memcpy(t->data.raw, input_tensor_data[j].data.get(),
                        input_tensor_data[j].bytes);
          }
        }
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::InitInterpreter() {
  auto resolver = GetOpResolver();
  const int32_t num_threads = params_.Get<int32_t>("num_threads");
  const bool use_caching = params_.Get<bool>("use_caching");

  (&interpreter_)->reset(
      new Interpreter(LoggingReporter::DefaultLoggingReporter(),
                      runtime_config_.planner_type));
  interpreter_->SetWindowSize(runtime_config_.schedule_window_size);
  interpreter_->SetProfileSmoothingConstant(
      runtime_config_.profile_smoothing_factor);
  if (runtime_config_.allow_work_steal) {
    interpreter_->AllowWorkSteal();
  }

  // Set log file path and write log headers
  TF_LITE_ENSURE_STATUS(interpreter_->PrepareLogging(runtime_config_.log_path));
  const tflite::impl::TfLiteCPUMaskFlags cpu_mask = 
      static_cast<tflite::impl::TfLiteCPUMaskFlags>(runtime_config_.cpu_masks);
  auto cpu_mask_set = tflite::impl::TfLiteCPUMaskGetSet(cpu_mask);
  TF_LITE_ENSURE_STATUS(SetCPUThreadAffinity(cpu_mask_set));

  TFLITE_LOG(INFO) << "Set affinity to "
      << tflite::impl::TfLiteCPUMaskGetName(cpu_mask)
      << " cores";

  for (int i = 0; i < kTfLiteNumDevices; i++) {
    const TfLiteDeviceFlags device_id = static_cast<TfLiteDeviceFlags>(i);
    // Skip as workers are not always available
    if (!interpreter_->GetWorker(device_id))
      continue;
    // Use global mask only if worker_mask is invalid
    tflite::impl::TfLiteCPUMaskFlags worker_mask =
        runtime_config_.worker_cpu_masks[i] == tflite::impl::kTfLiteNumCpuMasks ?
        cpu_mask : runtime_config_.worker_cpu_masks[i];
    const tflite::impl::CpuSet worker_mask_set = tflite::impl::TfLiteCPUMaskGetSet(worker_mask);
    TF_LITE_ENSURE_STATUS(interpreter_->SetWorkerThreadAffinity(worker_mask_set, device_id));
    TFLITE_LOG(INFO) << "Set affinity of "
                     << TfLiteDeviceGetName(device_id)
                     << " to "
                     << tflite::impl::TfLiteCPUMaskGetName(worker_mask)
                     << " cores";
  }

  auto& model_information = runtime_config_.model_information;
  for (int i = 0; i < model_information.size(); ++i) {
    std::string model_name = model_information[i].config.model_fname;
    TF_LITE_ENSURE_STATUS(LoadModel(model_name));
    int model_id = tflite::InterpreterBuilder::RegisterModel(
        *models_[i], model_information[i].config, *resolver, &interpreter_, num_threads);

    if (model_id == -1)
      return kTfLiteError;
    model_name_to_id_[model_name] = model_id;
  }

  if (interpreter_->NeedProfile()) {
    Json::Value model_name_profile;

    // load data from the given model profile file
    // if there is no such file, then `model_name_profile` will be empty
    if (FileExists(runtime_config_.model_profile)) {
      std::ifstream model_profile_file(runtime_config_.model_profile,
                                       std::ifstream::binary);
      model_profile_file >> model_name_profile;
    }

    // convert the model name strings to integer ids for the interpreter
    auto model_id_profile = ConvertModelNameToId(model_name_profile);
    interpreter_->Profile(params_.Get<int32_t>("profile_warmup_runs"),
                          params_.Get<int32_t>("profile_num_runs"),
                          model_id_profile);

    // update the profile file to include all new profile results from this run
    if (!runtime_config_.model_profile.empty()) {
      ConvertModelIdToName(model_id_profile, model_name_profile);
      std::ofstream out_file(runtime_config_.model_profile, std::ios::out);
      out_file << model_name_profile;
    }
  }

  TFLITE_LOG(INFO) <<  interpreter_->subgraphs_size()
                  << " subgraph loaded to the interpreter";

  if (!interpreter_) {
    TFLITE_LOG(ERROR) << "Failed to initialize the interpreter";
    return kTfLiteError;
  }
  // Manually enable caching behavior in TF Lite interpreter.
  if (use_caching) {
    external_context_.reset(new tflite::ExternalCpuBackendContext());
    std::unique_ptr<tflite::CpuBackendContext> cpu_backend_context(
        new tflite::CpuBackendContext());
    cpu_backend_context->SetUseCaching(true);
    cpu_backend_context->SetMaxNumThreads(num_threads);
    external_context_->set_internal_backend_context(
        std::move(cpu_backend_context));
    interpreter_->SetExternalContext(kTfLiteCpuBackendContext,
                                     external_context_.get());
  }

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::Init() {
  TF_LITE_ENSURE_STATUS(
      util::ParseJsonFile(params_.Get<std::string>("json_path"),
                          &runtime_config_)
  );
  TF_LITE_ENSURE_STATUS(InitInterpreter());

  // Install profilers if necessary right after interpreter is created so that
  // any memory allocations inside the TFLite runtime could be recorded if the
  // installed profiler profile memory usage information.
  profiling_listener_ = MayCreateProfilingListener();
  if (profiling_listener_) AddListener(profiling_listener_.get());

  interpreter_->SetAllowFp16PrecisionForFp32(params_.Get<bool>("allow_fp16"));

  for (int model_id = 0; model_id < runtime_config_.model_information.size(); model_id++) {
    // TODO: #73 share tensors across different subgraphs from same model
    for (int i = 0; i < kTfLiteNumDevices; i++) {
      const TfLiteDeviceFlags device_id = static_cast<TfLiteDeviceFlags>(i);
      auto subgraph_index = interpreter_->GetSubgraphIdx(model_id, device_id);
      if (subgraph_index < 0) {
        continue;
      }
      auto interpreter_inputs = interpreter_->inputs(subgraph_index);
      auto& input_layer_infos = runtime_config_.model_information[model_id].input_layer_infos;

      if (!input_layer_infos.empty()) {
        TFLITE_TOOLS_CHECK_EQ(input_layer_infos.size(), interpreter_inputs.size())
            << "Inputs mismatch: Model inputs #:" << input_layer_infos.size()
            << " expected: " << interpreter_inputs.size();
      }

      // Check if the tensor names match, and log a warning if it doesn't.
      // TODO(ycling): Consider to make this an error again when the new converter
      // create tensors with consistent naming.
      for (int j = 0; j < input_layer_infos.size(); ++j) {
        const util::InputLayerInfo& input = input_layer_infos[j];
        int tensor_index = interpreter_inputs[j];
        TfLiteTensor* t = interpreter_->tensor(subgraph_index, tensor_index);
        if (input.name != t->name) {
          TFLITE_LOG(WARN) << "Tensor # " << tensor_index << " is named " << t->name
                           << " but flags call it " << input.name;
        }
      }

      // Resize all non-string tensors.
      for (int j = 0; j < input_layer_infos.size(); ++j) {
        const util::InputLayerInfo& input = input_layer_infos[j];
        int tensor_index = interpreter_inputs[j];
        TfLiteTensor* t = interpreter_->tensor(subgraph_index, tensor_index);
        if (t->type != kTfLiteString) {
          interpreter_->ResizeInputTensor(subgraph_index, tensor_index, input.shape);
        }
      }
    }
  }

  ruy_profiling_listener_.reset(new RuyProfileListener());
  AddListener(ruy_profiling_listener_.get());

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::LoadModel(std::string graph) {
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(graph.c_str());

  if (!model) {
    TFLITE_LOG(ERROR) << "Failed to mmap model " << graph;
    return kTfLiteError;
  }
  TFLITE_LOG(INFO) << "Loaded model " << graph;
  models_.emplace_back(std::move(model));

  return kTfLiteOk;
}

std::unique_ptr<tflite::OpResolver> BenchmarkTfLiteModel::GetOpResolver()
    const {
  auto resolver = new tflite::ops::builtin::BuiltinOpResolver();
  RegisterSelectedOps(resolver);
  return std::unique_ptr<tflite::OpResolver>(resolver);
}

std::unique_ptr<BenchmarkListener>
BenchmarkTfLiteModel::MayCreateProfilingListener() const {
  if (!params_.Get<bool>("enable_op_profiling")) return nullptr;

  if (params_.Get<bool>("enable_platform_tracing")) {
    return std::unique_ptr<BenchmarkListener>(
        new PlatformProfilingListener(interpreter_.get()));
  }

  return std::unique_ptr<BenchmarkListener>(new ProfilingListener(
      interpreter_.get(), params_.Get<int32_t>("max_profiling_buffer_entries"),
      params_.Get<std::string>("profiling_output_csv_file"),
      CreateProfileSummaryFormatter(
          !params_.Get<std::string>("profiling_output_csv_file").empty())));
}

Interpreter::ModelDeviceToLatency
BenchmarkTfLiteModel::ConvertModelNameToId(const Json::Value name_profile) {
  Interpreter::ModelDeviceToLatency id_profile;
  for (auto name_profile_it = name_profile.begin();
       name_profile_it != name_profile.end(); ++name_profile_it) {
    std::string model_name = name_profile_it.key().asString();

    // check the integer id of this model name
    auto name_to_id_it = model_name_to_id_.find(model_name);
    if (name_to_id_it == model_name_to_id_.end()) {
      // we're not interested in this model for this run
      continue;
    }
    int model_id = name_to_id_it->second;

    const Json::Value idx_profile = *name_profile_it;
    for (auto idx_profile_it = idx_profile.begin();
         idx_profile_it != idx_profile.end(); ++idx_profile_it) {
      std::string idx = idx_profile_it.key().asString();

      // parse the key to retrieve start/end indices
      // e.g., "25/50" --> delim_pos = 2
      auto delim_pos = idx.find("/");
      std::string start_idx = idx.substr(0, delim_pos);
      std::string end_idx = idx.substr(delim_pos + 1, idx.length() - delim_pos - 1);
      
      const Json::Value device_profile = *idx_profile_it;
      for (auto device_profile_it = device_profile.begin();
           device_profile_it != device_profile.end();
           ++device_profile_it) {
        int device_id = device_profile_it.key().asInt();
        TfLiteDeviceFlags device_flag = static_cast<TfLiteDeviceFlags>(device_id);
        int64_t profiled_latency = (*device_profile_it).asInt64();

        if (profiled_latency <= 0) {
          // jsoncpp treats missing values (null) as zero,
          // so they will be filtered out here
          continue;
        }

        SubgraphKey key(model_id, device_flag,
                        std::stoi(start_idx), std::stoi(end_idx));
        id_profile[key] = profiled_latency;
      }
    }
  }
  return id_profile;
}

void BenchmarkTfLiteModel::ConvertModelIdToName(const Interpreter::ModelDeviceToLatency id_profile,
                                                Json::Value& name_profile) {
  for (auto& pair : id_profile) {
    SubgraphKey key = pair.first;
    int model_id = key.model_id;
    std::string start_idx = std::to_string(key.start_idx);
    std::string end_idx = std::to_string(key.end_idx);
    int64_t profiled_latency = pair.second;

    // check the string name of this model id
    std::string model_name;
    for (auto& name_id_pair : model_name_to_id_) {
      if (name_id_pair.second == key.model_id) {
        model_name = name_id_pair.first;
        break;
      }
    }

    if (model_name.empty()) {
      TFLITE_LOG(WARN) << "Cannot find model #" << model_id
                       << " in model_name_to_id_. Will ignore.";
      continue;
    }

    // copy all entries in id_profile --> name_profile
    // as an ad-hoc method, we simply concat the start/end indices to form
    // the level-two key in the final json value
    name_profile[model_name][start_idx + "/" + end_idx][key.device_flag] = profiled_latency;
  }
}

TfLiteStatus BenchmarkTfLiteModel::RunImpl(int i) { return interpreter_->Invoke(i); }
TfLiteStatus BenchmarkTfLiteModel::RunAll() {
  int num_iters = 3;
  for (int i = 0; i < num_iters; ++i) {
    for (int j = 0; j < models_.size(); ++j) {
      interpreter_->InvokeModelAsync(j);
    }
  }
  interpreter_->GetPlanner()->Wait();
  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::RunPeriodic() {
  // initialize values in case this isn't our first run
  kill_app_ = false;

  // spawn a child thread to do our work, since we're going to sleep
  // Note: spawning a separate thread is technically unnecessary if we only
  // have a single thread that generate requests, but we may have multiple
  // threads doing that in the future so we might as well make the code easily
  // adaptable to such situtations.
  GeneratePeriodicRequests();

  // wait for 60 seconds until we stop the benchmark
  // we could set a command line arg for this value as well
  std::this_thread::sleep_for(
      std::chrono::milliseconds(runtime_config_.running_time_ms));
  kill_app_ = true;

  interpreter_->GetPlanner()->Wait();
  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::RunStream() {
  int run_duration_us = runtime_config_.running_time_ms * 1000;
  int num_frames = 0;
  int64_t start = profiling::time::NowMicros();
  while(true) {
    interpreter_->InvokeModelsSync();
    int64_t current = profiling::time::NowMicros();
    num_frames++;
    if (current - start >= run_duration_us)
      break;
  }
  int64_t end = profiling::time::NowMicros();
  TFLITE_LOG(INFO) << "# processed frames: " << num_frames;
  TFLITE_LOG(INFO) << "Time taken (us): " << (end - start);
  TFLITE_LOG(INFO) << "Measured FPS: "
                   << (num_frames / (float)(end - start)) * 1000000;

  return kTfLiteOk;
}

void BenchmarkTfLiteModel::GeneratePeriodicRequests() {
  for (auto& m : interpreter_->GetModelConfig()) {
    int model_id = m.first;
    ModelConfig& model_config = m.second;
    int batch_size = model_config.batch_size,
        period_ms = model_config.period_ms;

    std::thread t([this, batch_size, model_id, period_ms]() {
      std::vector<Job> requests(batch_size, Job(model_id));
      while (true) {
        // measure the time it took to generate requests
        int64_t start = profiling::time::NowMicros();
        interpreter_->InvokeModelsAsync(requests);
        int64_t end = profiling::time::NowMicros();
        int duration_ms = (end - start) / 1000;

        // sleep until we reach period_ms
        if (duration_ms < period_ms) {
          std::this_thread::sleep_for(
              std::chrono::milliseconds(period_ms - duration_ms));
        }

        if (kill_app_) return;
      }
    });

    t.detach();
  }
}

}  // namespace benchmark
}  // namespace tflite
