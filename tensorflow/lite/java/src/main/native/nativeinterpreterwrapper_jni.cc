/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Temporal usage for debugging
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , "libtflite", __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG   , "libtflite", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , "libtflite", __VA_ARGS__)
#include <android/log.h>

#include <dlfcn.h>
#include <jni.h>
#include <stdio.h>
#include <time.h>

#include <vector>

#include "tensorflow/lite/config.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/tflite_api_dispatcher/tflite_api_dispatcher.h"
#include "tensorflow/lite/java/src/main/native/jni_utils.h"
#include "tensorflow/lite/util.h"

namespace tflite {
// This is to be provided at link-time by a library.
extern std::unique_ptr<OpResolver> CreateOpResolver();
}  // namespace tflite

using tflite::jni::BufferErrorReporter;
using tflite::jni::ThrowException;

namespace {

tflite_api_dispatcher::Interpreter* convertLongToInterpreter(JNIEnv* env,
                                                             jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to Interpreter.");
    return nullptr;
  }
  return reinterpret_cast<tflite_api_dispatcher::Interpreter*>(handle);
}

tflite_api_dispatcher::TfLiteModel* convertLongToModel(JNIEnv* env,
                                                       jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to model.");
    return nullptr;
  }
  return reinterpret_cast<tflite_api_dispatcher::TfLiteModel*>(handle);
}

BufferErrorReporter* convertLongToErrorReporter(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to ErrorReporter.");
    return nullptr;
  }
  return reinterpret_cast<BufferErrorReporter*>(handle);
}

std::vector<int> convertJIntArrayToVector(JNIEnv* env, jintArray inputs) {
  int size = static_cast<int>(env->GetArrayLength(inputs));
  std::vector<int> outputs(size, 0);
  jint* ptr = env->GetIntArrayElements(inputs, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, kIllegalArgumentException,
                   "Array has empty dimensions.");
    return {};
  }
  for (int i = 0; i < size; ++i) {
    outputs[i] = ptr[i];
  }
  env->ReleaseIntArrayElements(inputs, ptr, JNI_ABORT);
  return outputs;
}

jintArray convertVectorToJIntArray(JNIEnv* env, std::vector<int> inputs) {
  jintArray outputs = env->NewIntArray(inputs.size());
  jint* ptr = env->GetIntArrayElements(outputs, nullptr);
  for (int i = 0; i < inputs.size(); i++) {
    ptr[i] = inputs[i];
  }
  env->ReleaseIntArrayElements(outputs, ptr, 0);
  return outputs;
}

tflite::Tensors convertJLongArrayToTensors(JNIEnv* env, jlongArray handles) {
  std::vector<TfLiteTensor*> tensors;
  jlong* tensor_handles = env->GetLongArrayElements(handles, NULL);
  for (size_t i = 0; i < env->GetArrayLength(handles); i++) {
    tensors.push_back(
        tflite::jni::GetTensorFromHandle(env, tensor_handles[i]));
  }
  env->ReleaseLongArrayElements(handles, tensor_handles, 0);
  return tensors;
}

int getDataType(TfLiteType data_type) {
  switch (data_type) {
    case kTfLiteFloat32:
      return 1;
    case kTfLiteInt32:
      return 2;
    case kTfLiteUInt8:
      return 3;
    case kTfLiteInt64:
      return 4;
    case kTfLiteString:
      return 5;
    default:
      return -1;
  }
}

void printDims(char* buffer, int max_size, int* dims, int num_dims) {
  if (max_size <= 0) return;
  buffer[0] = '?';
  int size = 1;
  for (int i = 1; i < num_dims; ++i) {
    if (max_size > size) {
      int written_size =
          snprintf(buffer + size, max_size - size, ",%d", dims[i]);
      if (written_size < 0) return;
      size += written_size;
    }
  }
}

// Checks whether there is any difference between dimensions of a tensor and a
// given dimensions. Returns true if there is difference, else false.
bool AreDimsDifferent(JNIEnv* env, TfLiteTensor* tensor, jintArray dims) {
  int num_dims = static_cast<int>(env->GetArrayLength(dims));
  jint* ptr = env->GetIntArrayElements(dims, nullptr);
  if (ptr == nullptr) {
    ThrowException(env, kIllegalArgumentException,
                   "Empty dimensions of input array.");
    return true;
  }
  bool is_different = false;
  if (tensor->dims->size != num_dims) {
    is_different = true;
  } else {
    for (int i = 0; i < num_dims; ++i) {
      if (ptr[i] != tensor->dims->data[i]) {
        is_different = true;
        break;
      }
    }
  }
  env->ReleaseIntArrayElements(dims, ptr, JNI_ABORT);
  return is_different;
}

// TODO(yichengfan): evaluate the benefit to use tflite verifier.
bool VerifyModel(const void* buf, size_t len) {
  flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf), len);
  return tflite::VerifyModelBuffer(verifier);
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputNames(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle,
                                                                jint model_id) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    ThrowException(env, kUnsupportedOperationException,
                   "Internal error: Can not find java/lang/String class to get "
                   "input names.");
    return nullptr;
  }
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);
  size_t size = interpreter->inputs(subgraph_index).size();
  jobjectArray names = static_cast<jobjectArray>(
      env->NewObjectArray(size, string_class, env->NewStringUTF("")));
  for (int i = 0; i < size; ++i) {
    env->SetObjectArrayElement(names, i,
                               env->NewStringUTF(interpreter->GetInputName(subgraph_index, i)));
  }
  return names;
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allocateInputTensor(
    JNIEnv* env, jclass clazz, jlong handle, jint model_id, jint input_index) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);

  TfLiteTensor* input = TfLiteTensorCreateLike(
      interpreter->tensor(subgraph_index, interpreter->inputs(subgraph_index)[input_index]));

  return reinterpret_cast<jlong>(
      new tflite::jni::TensorHandle(input));
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getInputCount(JNIEnv* env,
                                                                jclass clazz,
                                                                jlong handle,
                                                                jint model_id) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);
  return static_cast<jint>(interpreter->inputs(subgraph_index).size());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allocateOutputTensor(
    JNIEnv* env, jclass clazz, jlong handle, jint model_id, jint output_index) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);

  TfLiteTensor* output = TfLiteTensorCreateLike(
      interpreter->tensor(subgraph_index, interpreter->outputs(subgraph_index)[output_index]));

  return reinterpret_cast<jlong>(
      new tflite::jni::TensorHandle(output));
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputCount(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle,
                                                                 jint model_id) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return 0;
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);
  return static_cast<jint>(interpreter->outputs(subgraph_index).size());
}

JNIEXPORT jobjectArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputNames(JNIEnv* env,
                                                                 jclass clazz,
                                                                 jlong handle,
                                                                 jint model_id) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return nullptr;
  jclass string_class = env->FindClass("java/lang/String");
  if (string_class == nullptr) {
    ThrowException(env, kUnsupportedOperationException,
                   "Internal error: Can not find java/lang/String class to get "
                   "output names.");
    return nullptr;
  }
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);
  size_t size = interpreter->outputs(subgraph_index).size();
  jobjectArray names = static_cast<jobjectArray>(
      env->NewObjectArray(size, string_class, env->NewStringUTF("")));
  for (int i = 0; i < size; ++i) {
    env->SetObjectArrayElement(
        names, i, env->NewStringUTF(interpreter->GetOutputName(subgraph_index, i)));
  }
  return names;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowFp16PrecisionForFp32(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetAllowFp16PrecisionForFp32(static_cast<bool>(allow));
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_allowBufferHandleOutput(
    JNIEnv* env, jclass clazz, jlong handle, jboolean allow) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetAllowBufferHandleOutput(allow);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_numThreads(JNIEnv* env,
                                                             jclass clazz,
                                                             jlong handle,
                                                             jint num_threads) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return;
  interpreter->SetNumThreads(static_cast<int>(num_threads));
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createErrorReporter(
    JNIEnv* env, jclass clazz, jint size) {
  BufferErrorReporter* error_reporter =
      new BufferErrorReporter(env, static_cast<int>(size));
  return reinterpret_cast<jlong>(error_reporter);
}

// Verifies whether the model is a flatbuffer file.
class JNIFlatBufferVerifier : public tflite_api_dispatcher::TfLiteVerifier {
 public:
  bool Verify(const char* data, int length,
              tflite::ErrorReporter* reporter) override {
    if (!VerifyModel(data, length)) {
      reporter->Report("The model is not a valid Flatbuffer file");
      return false;
    }
    return true;
  }
};

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModel(
    JNIEnv* env, jclass clazz, jstring model_file, jlong error_handle) {
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  const char* path = env->GetStringUTFChars(model_file, nullptr);

  std::unique_ptr<tflite_api_dispatcher::TfLiteVerifier> verifier;
  verifier.reset(new JNIFlatBufferVerifier());

  auto model = tflite_api_dispatcher::TfLiteModel::VerifyAndBuildFromFile(
      path, verifier.get(), error_reporter);
  if (!model) {
    ThrowException(env, kIllegalArgumentException,
                   "Contents of %s does not encode a valid "
                   "TensorFlow Lite model: %s",
                   path, error_reporter->CachedLastErrorMessage());
    env->ReleaseStringUTFChars(model_file, path);
    return 0;
  }
  env->ReleaseStringUTFChars(model_file, path);
  return reinterpret_cast<jlong>(model.release());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createModelWithBuffer(
    JNIEnv* env, jclass /*clazz*/, jobject model_buffer, jlong error_handle) {
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer));
  jlong capacity = env->GetDirectBufferCapacity(model_buffer);
  if (!VerifyModel(buf, capacity)) {
    ThrowException(env, kIllegalArgumentException,
                   "ByteBuffer is not a valid flatbuffer model");
    return 0;
  }

  auto model = tflite_api_dispatcher::TfLiteModel::BuildFromBuffer(
      buf, static_cast<size_t>(capacity), error_reporter);
  if (!model) {
    ThrowException(env, kIllegalArgumentException,
                   "ByteBuffer does not encode a valid model: %s",
                   error_reporter->CachedLastErrorMessage());
    return 0;
  }
  return reinterpret_cast<jlong>(model.release());
}

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_createInterpreter(
    JNIEnv* env, jclass clazz, jlong error_handle, jstring json_file) {
  LOGI("CreateInterpreter starts");
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;

  const char* path = env->GetStringUTFChars(json_file, nullptr);

  tflite::RuntimeConfig runtime_config;
  if (ParseRuntimeConfigFromJson(path, runtime_config) != kTfLiteOk) {
    ThrowException(env, kIllegalArgumentException,
                   "Parsing runtime_config json file failed");
    return 0;
  }

  for (int i = 0; i < runtime_config.planner_config.schedulers.size() ; i++) {
    LOGI("Parse done interpreter's planner : %d",
         runtime_config.planner_config.schedulers[i]);
  }
  auto interpreter(std::make_unique<tflite_api_dispatcher::Interpreter>(
      error_reporter, runtime_config));
  env->ReleaseStringUTFChars(json_file, path);

  LOGI("CreateInterpreter finishes");
  return reinterpret_cast<jlong>(interpreter.release());
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_registerModel(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong model_handle, 
    jlong error_handle, jstring model_name) {
  LOGI("RegisterModel starts");
  std::unique_ptr<tflite_api_dispatcher::Interpreter> interpreter(
      convertLongToInterpreter(env, interpreter_handle));
  if (interpreter == nullptr) return 0;

  tflite_api_dispatcher::TfLiteModel* model =
      convertLongToModel(env, model_handle);
  if (model == nullptr) return 0;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return 0;

  tflite::ModelConfig modelConfig;
  modelConfig.model_fname = env->GetStringUTFChars(model_name, nullptr);
  auto resolver = ::tflite::CreateOpResolver();
  int model_id =
      tflite_api_dispatcher::InterpreterBuilder::RegisterModel(
          *model, &modelConfig, *resolver.get(), &interpreter, 1);
  if (model_id == -1) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Cannot create interpreter: %s",
                   error_reporter->CachedLastErrorMessage());
  }
  interpreter.release();

  // TODO : NeedProfile / useCaching / MayCreateProfilingListener / interpreter_inputs

  LOGI("RegisterModel finishes. model_id = %d", model_id);
  return model_id;
}

JNIEXPORT jintArray JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_runAsync(
    JNIEnv* env, jclass clazz, jintArray model_ids, jobjectArray input_tensor_handles,
    jlong interpreter_handle, jlong error_handle, jlong slo) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return NULL;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return NULL;

  jsize num_models = env->GetArrayLength(model_ids);
  jint* model_ids_elements = env->GetIntArrayElements(model_ids, NULL);
  jsize num_model_inputs = env->GetArrayLength(input_tensor_handles);

  std::vector<tflite::Job> jobs;
  std::vector<tflite::Tensors> input_tensors(num_model_inputs);

  for (int i = 0; i < num_models; i++) {
    jobs.push_back(tflite::Job(model_ids_elements[i], slo));
    LOGI("RunAsync starts with model_id = %d", model_ids_elements[i]);

    if (input_tensors.size()) {
      jlongArray input_handles =
          (jlongArray)(env->GetObjectArrayElement(input_tensor_handles, i));
      input_tensors[i] = convertJLongArrayToTensors(env, input_handles);
    }
  }
  
  env->ReleaseIntArrayElements(model_ids, model_ids_elements, 0);

  std::vector<int> job_ids_vector =
      interpreter->InvokeModelsAsync(jobs, input_tensors);
  std::string job_ids_string;
  for (int job_id : job_ids_vector) {
    job_ids_string += std::to_string(job_id) + ",";
  }
  job_ids_string.pop_back();

  LOGI("RunAsync starts with job ids=%s", job_ids_string.c_str());
  LOGI("RunAsync finishes");
  return convertVectorToJIntArray(env, job_ids_vector);
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_wait(
    JNIEnv* env, jclass clazz, jintArray job_ids,
    jobjectArray output_tensor_handles, jlong interpreter_handle,
    jlong error_handle) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  std::vector<int> job_ids_vector = convertJIntArrayToVector(env, job_ids);
  if (job_ids_vector.size() == 0) return;
  std::string job_ids_string;
  for (int job_id : job_ids_vector) {
    job_ids_string += std::to_string(job_id) + ",";
  }
  job_ids_string.pop_back();

  LOGI("Wait starts with job ids=%s", job_ids_string.c_str());
    
  interpreter->GetPlanner()->Wait(job_ids_vector);

  jsize num_model_outputs = env->GetArrayLength(output_tensor_handles);
  if (num_model_outputs > 0) {
    std::vector<tflite::Tensors> output_tensors(num_model_outputs);

    for (int i = 0; i < num_model_outputs; i++) {
      jlongArray output_handles =
          (jlongArray)(env->GetObjectArrayElement(output_tensor_handles, i));
      output_tensors[i] = convertJLongArrayToTensors(env, output_handles);
      TfLiteStatus status = interpreter->GetOutputTensors(
          job_ids_vector[i], output_tensors[i]);
      
      if (status != kTfLiteOk) {
        ThrowException(env, kIllegalArgumentException,
                      "Internal error: Failed to copy %d-th output of job %d: %s",
                      i, job_ids_vector[i], error_reporter->CachedLastErrorMessage());
        return;
      }
    }
  }

  LOGI("Wait finishes");
}

JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_getOutputDataType(
    JNIEnv* env, jclass clazz, jlong handle, jint model_id, jint output_idx) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, handle);
  if (interpreter == nullptr) return -1;
  size_t subgraph_index = interpreter->GetSubgraphIdx(model_id, kTfLiteCPU);
  const int idx = static_cast<int>(output_idx);
  if (output_idx < 0 || output_idx >= interpreter->outputs(subgraph_index).size()) {
    ThrowException(env, kIllegalArgumentException,
                   "Failed to get %d-th output out of %d outputs", output_idx,
                   interpreter->outputs(subgraph_index).size());
    return -1;
  }
  TfLiteTensor* target = interpreter->tensor(subgraph_index, interpreter->outputs(subgraph_index)[idx]);
  int type = getDataType(target->type);
  return static_cast<jint>(type);
}

JNIEXPORT jboolean JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_resizeInput(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle,
    jint model_id, jint input_idx, jintArray dims, jboolean strict) {
  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return JNI_FALSE;
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return JNI_FALSE;
  bool is_changed_global = false;
  for (int device_id = 0; device_id < kTfLiteNumDevices; device_id++) {
    // Get all starting subgraphs.
    // TODO(#73): Remove duplicate memcopy for same model
    std::set<int> subgraph_indices = interpreter->GetSubgraphIdx(
        model_id, static_cast<TfLiteDeviceFlags>(device_id), 0);

    for (int subgraph_idx : subgraph_indices) {
      if (input_idx < 0 || input_idx >= interpreter->inputs(subgraph_idx).size()) {
        ThrowException(
            env, kIllegalArgumentException,
            "Input error: Can not resize %d-th input for a model having "
            "%d inputs.",
            input_idx, interpreter->inputs(subgraph_idx).size());
        return JNI_FALSE;
      }
      const int tensor_idx = interpreter->inputs(subgraph_idx)[input_idx];
      // check whether it is resizing with the same dimensions.
      TfLiteTensor* target = interpreter->tensor(subgraph_idx, tensor_idx);
      bool is_changed = AreDimsDifferent(env, target, dims);
      if (is_changed) {
        TfLiteStatus status;
        if (strict) {
          status = interpreter->ResizeInputTensorStrict(
              subgraph_idx, tensor_idx, convertJIntArrayToVector(env, dims));
        } else {
          status = interpreter->ResizeInputTensor(
              subgraph_idx, tensor_idx, convertJIntArrayToVector(env, dims));
        }
        if (status != kTfLiteOk) {
          ThrowException(env, kIllegalArgumentException,
                        "Internal error: Failed to resize %d-th input: %s",
                        input_idx, error_reporter->CachedLastErrorMessage());
          return JNI_FALSE;
        }
      }
      is_changed_global |= is_changed;
    }
  }
  return is_changed_global ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_NativeInterpreterWrapper_resetVariableTensors(
    JNIEnv* env, jclass clazz, jlong interpreter_handle, jlong error_handle, jint model_id) {
  tflite_api_dispatcher::Interpreter* interpreter =
      convertLongToInterpreter(env, interpreter_handle);
  if (interpreter == nullptr) return;

  BufferErrorReporter* error_reporter =
      convertLongToErrorReporter(env, error_handle);
  if (error_reporter == nullptr) return;

  for (int device_id = 0; device_id < kTfLiteNumDevices; device_id++) {
    // Get all starting subgraphs.
    // TODO(#73): Remove duplicate memcopy for same model
    std::set<int> subgraph_indices = interpreter->GetSubgraphIdx(
        model_id, static_cast<TfLiteDeviceFlags>(device_id), 0);

    for (int subgraph_idx : subgraph_indices) {
      TfLiteStatus status = interpreter->ResetVariableTensors(subgraph_idx);
      if (status != kTfLiteOk) {
        ThrowException(env, kIllegalArgumentException,
                      "Internal error: Failed to reset variable tensors: %s",
                      error_reporter->CachedLastErrorMessage());
      }
    }
  }
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_delete(
    JNIEnv* env, jclass clazz, jlong error_handle, jlong interpreter_handle) {
  if (interpreter_handle != 0) {
    delete convertLongToInterpreter(env, interpreter_handle);
  }
  if (error_handle != 0) {
    delete convertLongToErrorReporter(env, error_handle);
  }
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_deleteModel(
    JNIEnv* env, jclass clazz, jlong model_handle) {
  if (model_handle != 0) {
    delete convertLongToModel(env, model_handle);
  }
}

JNIEXPORT void JNICALL Java_org_tensorflow_lite_NativeInterpreterWrapper_deleteTensor(
    JNIEnv* env, jclass clazz, jlong tensor_handle) {
  if (tensor_handle != 0) {
    TfLiteTensor* t = tflite::jni::GetTensorFromHandle(env, tensor_handle);

    TfLiteTensorFree(t);
    free(t);
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif
