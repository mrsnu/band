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
#include "absl/strings/numbers.h"
#include "ruy/profiler/profiler.h"  // from @ruy
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/cpu/cpu.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/profiling/platform_profiler.h"
#include "tensorflow/lite/profiling/profile_summary_formatter.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/benchmark/benchmark_utils.h"
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

std::vector<std::string> Split(const std::string& str, const char delim) {
  std::vector<std::string> results;
  if (!util::SplitAndParse(str, delim, &results)) {
    results.clear();
  }
  return results;
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

int FindLayerInfoIndex(std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info,
                       const std::string& input_name,
                       const string& names_string) {
  for (int i = 0; i < info->size(); ++i) {
    if (info->at(i).name == input_name) {
      return i;
    }
  }
  TFLITE_LOG(FATAL) << "Cannot find the corresponding input_layer name("
                    << input_name << ") in --input_layer as " << names_string;
  return -1;
}

TfLiteStatus PopulateInputValueRanges(
    const std::string& names_string, const std::string& value_ranges_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  std::vector<std::string> value_ranges = Split(value_ranges_string, ':');
  for (const auto& val : value_ranges) {
    std::vector<std::string> name_range = Split(val, ',');
    if (name_range.size() != 3) {
      TFLITE_LOG(ERROR) << "Wrong input value range item specified: " << val;
      return kTfLiteError;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx = FindLayerInfoIndex(info, name_range[0], names_string);

    // Parse the range value.
    int low, high;
    bool has_low = absl::SimpleAtoi(name_range[1], &low);
    bool has_high = absl::SimpleAtoi(name_range[2], &high);
    if (!has_low || !has_high || low > high) {
      TFLITE_LOG(ERROR)
          << "Wrong low and high value of the input value range specified: "
          << val;
      return kTfLiteError;
    }
    info->at(layer_info_idx).has_value_range = true;
    info->at(layer_info_idx).low = low;
    info->at(layer_info_idx).high = high;
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateInputValueFiles(
    const std::string& names_string, const std::string& value_files_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  std::vector<std::string> value_files = Split(value_files_string, ',');
  for (const auto& val : value_files) {
    std::vector<std::string> name_file = Split(val, ':');
    if (name_file.size() != 2) {
      TFLITE_LOG(ERROR) << "Wrong input value file item specified: " << val;
      return kTfLiteError;
    }

    // Ensure the specific input layer name exists.
    int layer_info_idx = FindLayerInfoIndex(info, name_file[0], names_string);
    if (info->at(layer_info_idx).has_value_range) {
      TFLITE_LOG(WARN)
          << "The input_name:" << info->at(layer_info_idx).name
          << " appears both in input_layer_value_files and "
             "input_layer_value_range. The input_layer_value_range of the "
             "input_name will be ignored.";
    }
    info->at(layer_info_idx).input_file_path = name_file[1];
  }
  return kTfLiteOk;
}

TfLiteStatus PopulateInputLayerInfo(
    const std::string& names_string, const std::string& shapes_string,
    const std::string& value_ranges_string,
    const std::string& value_files_string,
    std::vector<BenchmarkTfLiteModel::InputLayerInfo>* info) {
  info->clear();
  std::vector<std::string> names = Split(names_string, ',');
  std::vector<std::string> shapes = Split(shapes_string, ':');

  if (names.size() != shapes.size()) {
    TFLITE_LOG(ERROR) << "The number of items in"
                      << " --input_layer_shape (" << shapes_string << ", with "
                      << shapes.size() << " items)"
                      << " must match the number of items in"
                      << " --input_layer (" << names_string << ", with "
                      << names.size() << " items)."
                      << " For example --input_layer=input1,input2"
                      << " --input_layer_shape=1,224,224,4:1,20";
    return kTfLiteError;
  }

  for (int i = 0; i < names.size(); ++i) {
    info->push_back(BenchmarkTfLiteModel::InputLayerInfo());
    BenchmarkTfLiteModel::InputLayerInfo& input = info->back();

    input.name = names[i];

    TFLITE_TOOLS_CHECK(util::SplitAndParse(shapes[i], ',', &input.shape))
        << "Incorrect size string specified: " << shapes[i];
    for (int dim : input.shape) {
      if (dim == -1) {
        TFLITE_LOG(ERROR)
            << "Any unknown sizes in the shapes (-1's) must be replaced"
            << " with the size you want to benchmark with.";
        return kTfLiteError;
      }
    }
  }

  // Populate input value range if it's specified.
  TF_LITE_ENSURE_STATUS(
      PopulateInputValueRanges(names_string, value_ranges_string, info));

  // Populate input value files if it's specified.
  TF_LITE_ENSURE_STATUS(
      PopulateInputValueFiles(names_string, value_files_string, info));

  return kTfLiteOk;
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
  default_params.AddParam("input_layer",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_shape",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_value_range",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("input_layer_value_files",
                          BenchmarkParam::Create<std::string>(""));
  default_params.AddParam("use_legacy_nnapi",
                          BenchmarkParam::Create<bool>(false));
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
  // Free up any pre-allocated tensor data during PrepareInputData.
  inputs_data_.clear();
}

BenchmarkTfLiteModel::~BenchmarkTfLiteModel() { CleanUp(); }

std::vector<Flag> BenchmarkTfLiteModel::GetFlags() {
  std::vector<Flag> flags = BenchmarkModel::GetFlags();
  std::vector<Flag> specific_flags = {
      CreateFlag<std::string>("input_layer", &params_, "input layer names"),
      CreateFlag<std::string>("input_layer_shape", &params_,
                              "input layer shape"),
      CreateFlag<std::string>(
          "input_layer_value_range", &params_,
          "A map-like string representing value range for *integer* input "
          "layers. Each item is separated by ':', and the item value consists "
          "of input layer name and integer-only range values (both low and "
          "high are inclusive) separated by ',', e.g. input1,1,2:input2,0,254"),
      CreateFlag<std::string>(
          "input_layer_value_files", &params_,
          "A map-like string representing value file. Each item is separated "
          "by ',', and the item value consists "
          "of input layer name and value file path separated by ':', e.g. "
          "input1:file_path1,input2:file_path2. If the input_name appears both "
          "in input_layer_value_range and input_layer_value_files, "
          "input_layer_value_range of the input_name will be ignored. The file "
          "format is binary and it should be array format or null separated "
          "strings format."),
      CreateFlag<bool>("use_legacy_nnapi", &params_, "use legacy nnapi api"),
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
  TFLITE_LOG(INFO) << "Input layers: ["
                   << params_.Get<std::string>("input_layer") << "]";
  TFLITE_LOG(INFO) << "Input shapes: ["
                   << params_.Get<std::string>("input_layer_shape") << "]";
  TFLITE_LOG(INFO) << "Input value ranges: ["
                   << params_.Get<std::string>("input_layer_value_range")
                   << "]";
  TFLITE_LOG(INFO) << "Input layer values files: ["
                   << params_.Get<std::string>("input_layer_value_files")
                   << "]";
#if defined(__ANDROID__)
  TFLITE_LOG(INFO) << "Use legacy nnapi : ["
                   << params_.Get<bool>("use_legacy_nnapi") << "]";
#endif
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

  return PopulateInputLayerInfo(
      params_.Get<std::string>("input_layer"),
      params_.Get<std::string>("input_layer_shape"),
      params_.Get<std::string>("input_layer_value_range"),
      params_.Get<std::string>("input_layer_value_files"), &inputs_);
}

uint64_t BenchmarkTfLiteModel::ComputeInputBytes() {
  TFLITE_TOOLS_CHECK(interpreter_);
  uint64_t total_input_bytes = 0;
  for (int input : interpreter_->inputs()) {
    auto* t = interpreter_->tensor(input);
    total_input_bytes += t->bytes;
  }
  return total_input_bytes;
}

int64_t BenchmarkTfLiteModel::MayGetModelFileSize() {
  int64_t total_mem_size = 0;

  for (int i = 0; i < model_configs_.size(); ++i) {
    std::ifstream in_file(model_configs_[i].model_fname,
                          std::ios::binary | std::ios::ate);
    total_mem_size += in_file.tellg();
  }

  return total_mem_size;
}

BenchmarkTfLiteModel::InputTensorData BenchmarkTfLiteModel::LoadInputTensorData(
    const TfLiteTensor& t, const std::string& input_file_path) {
  std::ifstream value_file(input_file_path, std::ios::binary);
  if (!value_file.good()) {
    TFLITE_LOG(FATAL) << "Failed to read the input_layer_value_file:"
                      << input_file_path;
  }
  InputTensorData t_data;
  if (t.type == kTfLiteString) {
    t_data.data = VoidUniquePtr(
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
        VoidUniquePtr(static_cast<void*>(new char[t.bytes]),
                      [](void* ptr) { delete[] static_cast<char*>(ptr); });
    value_file.clear();
    value_file.seekg(0, std::ios_base::beg);
    value_file.read(static_cast<char*>(t_data.data.get()), t.bytes);
  }
  return t_data;
}

BenchmarkTfLiteModel::InputTensorData
BenchmarkTfLiteModel::CreateRandomTensorData(const TfLiteTensor& t,
                                             const InputLayerInfo* layer_info) {
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
  return InputTensorData();
}

TfLiteStatus BenchmarkTfLiteModel::PrepareInputData() {
  CleanUp();

  // Note the corresponding relation between 'interpreter_inputs' and 'inputs_'
  // (i.e. the specified input layer info) has been checked in
  // BenchmarkTfLiteModel::Init() before calling this function. So, we simply
  // use the corresponding input layer info to initializethe input data value
  // properly.
  auto interpreter_inputs = interpreter_->inputs();
  for (int i = 0; i < interpreter_inputs.size(); ++i) {
    int tensor_index = interpreter_inputs[i];
    const TfLiteTensor& t = *(interpreter_->tensor(tensor_index));
    const InputLayerInfo* input_layer_info = nullptr;
    // Note that when input layer parameters (i.e. --input_layer,
    // --input_layer_shape) are not specified, inputs_ is empty.
    if (!inputs_.empty()) input_layer_info = &inputs_[i];

    InputTensorData t_data;
    if (input_layer_info && !input_layer_info->input_file_path.empty()) {
      t_data = LoadInputTensorData(t, input_layer_info->input_file_path);
    } else {
      t_data = CreateRandomTensorData(t, input_layer_info);
    }
    inputs_data_.push_back(std::move(t_data));
  }
  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::ResetInputsAndOutputs() {
  auto interpreter_inputs = interpreter_->inputs();
  // Set the values of the input tensors from inputs_data_.
  for (int j = 0; j < interpreter_inputs.size(); ++j) {
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type == kTfLiteString) {
      if (inputs_data_[j].data) {
        static_cast<DynamicBuffer*>(inputs_data_[j].data.get())
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
      std::memcpy(t->data.raw, inputs_data_[j].data.get(),
                  inputs_data_[j].bytes);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus BenchmarkTfLiteModel::InitInterpreter() {
  auto resolver = GetOpResolver();
  const int32_t num_threads = params_.Get<int32_t>("num_threads");
  const bool use_caching = params_.Get<bool>("use_caching");
  auto cpuMask = tflite::impl::GetCPUThreadAffinityMask(
      static_cast<tflite::impl::TFLiteCPUMasks>(runtime_config_.cpu_masks));

  (&interpreter_)->reset(
      new Interpreter(LoggingReporter::DefaultLoggingReporter(),
                      runtime_config_.planner_type));
  interpreter_->SetWindowSize(runtime_config_.schedule_window_size);
  if (runtime_config_.allow_work_steal) {
    interpreter_->AllowWorkSteal();
  }

  // Set log file path and write log headers
  TF_LITE_ENSURE_STATUS(interpreter_->PrepareLogging(runtime_config_.log_path));

  // Set worker threads and current thread affinity
  TF_LITE_ENSURE_STATUS(interpreter_->SetWorkerThreadAffinity(cpuMask));
  TF_LITE_ENSURE_STATUS(SetCPUThreadAffinity(cpuMask));

  TFLITE_LOG(INFO) << "Set affinity to "
      << tflite::impl::GetCPUThreadAffinityMaskString(
             static_cast<tflite::impl::TFLiteCPUMasks>(runtime_config_.cpu_masks))
      << " cores";
  
  for (int i = 0; i < model_configs_.size(); ++i) {
    std::string model_name = model_configs_[i].model_fname;
    TF_LITE_ENSURE_STATUS(LoadModel(model_name));
    int model_id = tflite::InterpreterBuilder::RegisterModel(
        *models_[i], model_configs_[i], *resolver, &interpreter_, num_threads);

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
  TF_LITE_ENSURE_STATUS(ParseJsonFile());
  TF_LITE_ENSURE_STATUS(InitInterpreter());

  // Install profilers if necessary right after interpreter is created so that
  // any memory allocations inside the TFLite runtime could be recorded if the
  // installed profiler profile memory usage information.
  profiling_listener_ = MayCreateProfilingListener();
  if (profiling_listener_) AddListener(profiling_listener_.get());

  interpreter_->UseNNAPI(params_.Get<bool>("use_legacy_nnapi"));
  interpreter_->SetAllowFp16PrecisionForFp32(params_.Get<bool>("allow_fp16"));

  auto interpreter_inputs = interpreter_->inputs();

  if (!inputs_.empty()) {
    TFLITE_TOOLS_CHECK_EQ(inputs_.size(), interpreter_inputs.size())
        << "Inputs mismatch: Model inputs #:" << inputs_.size()
        << " expected: " << interpreter_inputs.size();
  }

  // Check if the tensor names match, and log a warning if it doesn't.
  // TODO(ycling): Consider to make this an error again when the new converter
  // create tensors with consistent naming.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (input.name != t->name) {
      TFLITE_LOG(WARN) << "Tensor # " << i << " is named " << t->name
                       << " but flags call it " << input.name;
    }
  }

  // Resize all non-string tensors.
  for (int j = 0; j < inputs_.size(); ++j) {
    const InputLayerInfo& input = inputs_[j];
    int i = interpreter_inputs[j];
    TfLiteTensor* t = interpreter_->tensor(i);
    if (t->type != kTfLiteString) {
      interpreter_->ResizeInputTensor(i, input.shape);
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

TfLiteStatus BenchmarkTfLiteModel::ParseJsonFile() {
  std::string json_fname = params_.Get<std::string>("json_path");
  std::ifstream config(json_fname, std::ifstream::binary);

  Json::Value root;
  config >> root;

  if (!root.isObject()) {
    TFLITE_LOG(ERROR) << "Please validate the json config file.";
    return kTfLiteError;
  }

  // Set Runtime Configurations
  // Optional
  if (!root["cpu_masks"].isNull())
    runtime_config_.cpu_masks = root["cpu_masks"].asInt();
  if (!root["running_time_ms"].isNull())
    runtime_config_.running_time_ms = root["running_time_ms"].asInt();
  if (!root["model_profile"].isNull())
    runtime_config_.model_profile = root["model_profile"].asString();
  if (!root["allow_work_steal"].isNull())
    runtime_config_.allow_work_steal = root["allow_work_steal"].asBool();
  if (!root["schedule_window_size"].isNull()) {
    runtime_config_.schedule_window_size = root["schedule_window_size"].asInt();
    if (runtime_config_.schedule_window_size <= 0) {
      TFLITE_LOG(ERROR) << "Make sure `schedule_window_size` > 0.";
      return kTfLiteError;
    }
  }

  // Required
  if (root["log_path"].isNull() ||
      root["planner"].isNull() ||
      root["execution_mode"].isNull() ||
      root["models"].isNull()) {
    TFLITE_LOG(ERROR) << "Please check if arguments `execution_mode`, "
                      << "`log_path`, `planner` and `models`"
                      << " are given in the config file.";
    return kTfLiteError;
  }

  runtime_config_.log_path = root["log_path"].asString();
  runtime_config_.execution_mode = root["execution_mode"].asString();

  int planner_id = root["planner"].asInt();
  if (planner_id < kFixedDevice || planner_id >= kNumPlannerTypes) {
    TFLITE_LOG(ERROR) << "Wrong `planner` argument is given.";
    return kTfLiteError;
  }
  runtime_config_.planner_type = static_cast<TfLitePlannerType>(planner_id);

  // Set Model Configurations
  for (int i = 0; i < root["models"].size(); ++i) {
    Interpreter::ModelConfig model;
    Json::Value model_json_value = root["models"][i];
    if (model_json_value["graph"].isNull() ||
        model_json_value["period_ms"].isNull()) {
      TFLITE_LOG(ERROR) << "Please check if arguments `graph` and `period_ms`"
                           " are given in the model configs.";
      return kTfLiteError;
    }
    model.model_fname = model_json_value["graph"].asString();
    model.period_ms = model_json_value["period_ms"].asInt();

    // Set `batch_size`.
    // If no `batch_size` is given, the default batch size will be set to 1.
    if (!model_json_value["batch_size"].isNull())
      model.batch_size = model_json_value["batch_size"].asInt();

    // Set `device`.
    // Fixes to the device if specified in case of `FixedDevicePlanner`.
    if (!model_json_value["device"].isNull())
      model.device = model_json_value["device"].asInt();

    model_configs_.push_back(model);
  }

  if (model_configs_.size() == 0) {
    TFLITE_LOG(ERROR) << "Please specify at list one model "
                      << "in `models` argument.";
    return kTfLiteError;
  }

  TFLITE_LOG(INFO) << root;

  return kTfLiteOk;
}

Interpreter::ModelDeviceToLatency
BenchmarkTfLiteModel::ConvertModelNameToId(const Json::Value name_profile) {
  Interpreter::ModelDeviceToLatency id_profile;
  for (auto profile_it = name_profile.begin();
       profile_it != name_profile.end(); ++profile_it) {
    std::string model_name = profile_it.key().asString();

    // check the integer id of this model name
    auto name_to_id_it = model_name_to_id_.find(model_name);
    if (name_to_id_it == model_name_to_id_.end()) {
      // we're not interested in this model for this run
      continue;
    }
    int model_id = name_to_id_it->second;

    const Json::Value inner = *profile_it;
    for (auto inner_it = inner.begin(); inner_it != inner.end(); ++inner_it) {
      int device_id = inner_it.key().asInt();
      int64_t profiled_latency = (*inner_it).asInt64();

      if (profiled_latency <= 0) {
        // jsoncpp treats missing values (null) as zero,
        // so they will be filtered out here
        continue;
      }

      // copy all entries in name_profile --> id_profile for this model
      id_profile[{model_id, device_id}] = profiled_latency;
    }
  }
  return id_profile;
}

void BenchmarkTfLiteModel::ConvertModelIdToName(const Interpreter::ModelDeviceToLatency id_profile,
                                                Json::Value& name_profile) {
  for (auto& pair : id_profile) {
    int model_id = pair.first.first;
    int device_id = pair.first.second;
    int64_t profiled_latency = pair.second;

    // check the string name of this model id
    std::string model_name;
    for (auto& name_id_pair : model_name_to_id_) {
      if (name_id_pair.second == model_id) {
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
    name_profile[model_name][device_id] = profiled_latency;
  }
}

TfLiteStatus BenchmarkTfLiteModel::RunImpl() { return interpreter_->Invoke(); }
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
                   << ((float)num_frames / ((end - start) / 1000000));

  return kTfLiteOk;
}

void BenchmarkTfLiteModel::GeneratePeriodicRequests() {
  for (auto& m : interpreter_->GetModelConfig()) {
    int model_id = m.first;
    Interpreter::ModelConfig& model_config = m.second;
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
