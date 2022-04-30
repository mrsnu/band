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
#include "tensorflow/lite/c/c_api.h"

#include <memory>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/config.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/version.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace {
class CallbackErrorReporter : public tflite::ErrorReporter {
 public:
  using ErrorCallback = void (*)(void* user_data, const char* format,
                                 va_list args);

  CallbackErrorReporter(ErrorCallback callback, void* user_data)
      : callback_(callback), user_data_(user_data) {}

  int Report(const char* format, va_list args) override {
    callback_(user_data_, format, args);
    return 0;
  }

 private:
  ErrorCallback callback_;
  void* user_data_;
};
}  // namespace

// LINT.IfChange

const char* TfLiteVersion() { return TFLITE_VERSION_STRING; }

TfLiteModel* TfLiteModelCreate(const void* model_data, size_t model_size) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      static_cast<const char*>(model_data), model_size);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model), nullptr, tflite::MutableOpResolver()} : nullptr;
}

TfLiteModel* TfLiteModelCreateFromFile(const char* model_path) {
  auto model = tflite::FlatBufferModel::VerifyAndBuildFromFile(model_path);
  std::shared_ptr<const tflite::FlatBufferModel> shared_model(model.release());
  return shared_model ? new TfLiteModel{std::move(shared_model), model_path, tflite::MutableOpResolver()} : nullptr;
}

void TfLiteModelDelete(TfLiteModel* model) { delete model; }

TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate() {
  return new TfLiteInterpreterOptions{};
}

void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions* options) {
  delete options;
}

void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data) {
  options->error_reporter = reporter;
  options->error_reporter_user_data = user_data;
}

void TfLiteInterpreterOptionsSetOnInvokeEnd(
    TfLiteInterpreterOptions* options,
    void (*on_end_invoke)(void* user_data, int job_id, TfLiteStatus status),
    void* user_data) {
  options->on_end_invoke = on_end_invoke;
  options->on_invoke_user_data = user_data;
}

TfLiteStatus TfLiteInterpreterOptionsSetConfigPath(
    TfLiteInterpreterOptions* options,
    const char* config_path) {
  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;
  if (options && options->error_reporter != nullptr) {
    optional_error_reporter.reset(
        new CallbackErrorReporter(options->error_reporter,
                                  options->error_reporter_user_data));
  }
  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();

  if (tflite::ParseRuntimeConfigFromJson(config_path, options->config, error_reporter) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Parsing runtime_config json file failed.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus TfLiteInterpreterOptionsSetConfigFile(
    TfLiteInterpreterOptions* options,
    const void* config_data, size_t config_size) {
  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;
  if (options && options->error_reporter != nullptr) {
    optional_error_reporter.reset(
        new CallbackErrorReporter(options->error_reporter,
                                  options->error_reporter_user_data));
  }
  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();

  if (tflite::ParseRuntimeConfigFromJson(config_data, config_size, options->config, error_reporter) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Parsing runtime_config json file failed.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteInterpreterOptions* optional_options) {
  std::unique_ptr<tflite::ErrorReporter> optional_error_reporter;
  if (optional_options && optional_options->error_reporter != nullptr) {
    optional_error_reporter.reset(
        new CallbackErrorReporter(optional_options->error_reporter,
                                  optional_options->error_reporter_user_data));
  }
  tflite::ErrorReporter* error_reporter = optional_error_reporter
                                              ? optional_error_reporter.get()
                                              : tflite::DefaultErrorReporter();

  std::unique_ptr<tflite::Interpreter> interpreter =
      std::make_unique<tflite::Interpreter>(error_reporter,
                                            optional_options->config);
  if (optional_options->on_end_invoke) {
    auto user_data_invoke = std::bind(
        optional_options->on_end_invoke, optional_options->on_invoke_user_data,
        std::placeholders::_1, std::placeholders::_2);
    interpreter->SetEndInvokeFunction(user_data_invoke);
  }
  return new TfLiteInterpreter{std::move(optional_error_reporter),
                               std::move(interpreter)};
}

void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter) {
  delete interpreter;
}

int32_t TfLiteInterpreterRegisterModel(TfLiteInterpreter* interpreter, TfLiteModel* model) {
  if (interpreter == nullptr || model == nullptr) return 0;

  tflite::ModelConfig model_config;

  // TODO(b/111881878): Allow use of C API without pulling in all builtin ops.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddAll(model->op_resolver);

  int model_id =
      tflite::InterpreterBuilder::RegisterModel(*model->impl, &model_config, resolver, &interpreter->impl);

  if (model_id == -1) {
    TF_LITE_REPORT_ERROR(interpreter->impl->GetErrorReporter(),
                         "Internal error: Cannot register model: %s",
                         model->model_path);
  }

  return model_id;
}

void TfLiteInterpreterInvokeSync(TfLiteInterpreter* interpreter, int32_t model_id, TfLiteTensor** inputs, TfLiteTensor** outputs) {
  if (inputs && outputs) {
    std::vector<TfLiteTensor*> input_tensors(inputs, inputs + sizeof(inputs) / sizeof(TfLiteTensor*));
    std::vector<TfLiteTensor*> output_tensors(outputs, outputs + sizeof(outputs) / sizeof(TfLiteTensor*));
    interpreter->impl->InvokeModelSync(model_id, input_tensors, output_tensors);
  } else {
    interpreter->impl->InvokeModelSync(model_id);
  }
}

int32_t TfLiteInterpreterInvokeAsync(TfLiteInterpreter* interpreter, int32_t model_id, TfLiteTensor** inputs) {
  if (inputs) {
    std::vector<TfLiteTensor*> input_tensors(inputs, inputs + sizeof(inputs) / sizeof(TfLiteTensor*));
    return interpreter->impl->InvokeModelAsync(model_id, input_tensors);
  } else {
    return interpreter->impl->InvokeModelAsync(model_id);
  }
}

TfLiteStatus TfLiteInterpreterWait(TfLiteInterpreter* interpreter, int job_id, TfLiteTensor** outputs) {
  interpreter->impl->GetPlanner()->Wait({job_id});
  if (outputs) {
    std::vector<TfLiteTensor*> output_tensors(outputs, outputs + sizeof(outputs) / sizeof(TfLiteTensor*));
    return interpreter->impl->GetOutputTensors(job_id, output_tensors);
  }
  return kTfLiteOk;
}

int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter, int32_t model_id) {
  if (interpreter == nullptr) return 0;
  size_t subgraph_index = interpreter->impl->GetSubgraphIdx(model_id, kTfLiteCPU);
  return interpreter->impl->inputs(subgraph_index).size();
}

int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter, int32_t model_id) {
  if (interpreter == nullptr) return 0;
  size_t subgraph_index = interpreter->impl->GetSubgraphIdx(model_id, kTfLiteCPU);
  return interpreter->impl->outputs(subgraph_index).size();
}

TfLiteTensor* TfLiteInterpreterAllocateInputTensor(
    const TfLiteInterpreter* interpreter, int32_t model_id, int32_t input_index) {
  if (interpreter == nullptr) return nullptr;
  size_t subgraph_index = interpreter->impl->GetSubgraphIdx(model_id, kTfLiteCPU);

  return TfLiteTensorCreateLike(
      interpreter->impl->tensor(subgraph_index, interpreter->impl->inputs(subgraph_index)[input_index]));
}

TfLiteTensor* TfLiteInterpreterAllocateOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t model_id, int32_t output_index) {
  if (interpreter == nullptr) return nullptr;
  size_t subgraph_index = interpreter->impl->GetSubgraphIdx(model_id, kTfLiteCPU);

  return TfLiteTensorCreateLike(
      interpreter->impl->tensor(subgraph_index, interpreter->impl->outputs(subgraph_index)[output_index]));
}

void TfLiteTensorDeallocate(TfLiteTensor* tensor) {
  TfLiteTensorDelete(tensor);
}

TfLiteType TfLiteTensorType(const TfLiteTensor* tensor) { return tensor->type; }

int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor) {
  return tensor->dims->size;
}

int32_t TfLiteTensorDim(const TfLiteTensor* tensor, int32_t dim_index) {
  return tensor->dims->data[dim_index];
}

size_t TfLiteTensorByteSize(const TfLiteTensor* tensor) {
  return tensor->bytes;
}

void* TfLiteTensorData(const TfLiteTensor* tensor) {
  return static_cast<void*>(tensor->data.raw);
}

const char* TfLiteTensorName(const TfLiteTensor* tensor) {
  return tensor->name;
}

TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor) {
  return tensor->params;
}

TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor* tensor,
                                        const void* input_data,
                                        size_t input_data_size) {
  if (tensor->bytes != input_data_size) {
    return kTfLiteError;
  }
  memcpy(tensor->data.raw, input_data, input_data_size);
  return kTfLiteOk;
}

TfLiteStatus TfLiteTensorCopyToBuffer(const TfLiteTensor* tensor,
                                      void* output_data,
                                      size_t output_data_size) {
  if (tensor->bytes != output_data_size) {
    return kTfLiteError;
  }
  memcpy(output_data, tensor->data.raw, output_data_size);
  return kTfLiteOk;
}

// LINT.ThenChange(//tensorflow/lite/experimental/examples/unity/TensorFlowLitePlugin/Assets/TensorFlowLite/SDK/Scripts/Interpreter.cs)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
