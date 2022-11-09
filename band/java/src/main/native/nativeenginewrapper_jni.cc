#include <jni.h>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/java/src/main/native/jni_utils.h"
#include "band/model.h"
#include "band/tensor.h"

using Band::Engine;
using Band::Model;
using Band::RuntimeConfig;
using Band::RuntimeConfigBuilder;
using Band::Tensor;
using Band::Tensors;
using Band::jni::BufferErrorReporter;
using Band::jni::convertLongToConfig;
using Band::jni::convertLongToEngine;
using Band::jni::convertLongToModel;
using Band::jni::convertLongToTensor;
using Band::jni::convertLongToJobId;
using Band::jni::convertLongListToTensors;

// TODO(widiba03304): error reporter should be adopted to check null for
// pointers

// private static native long createErrorReporter(int size);
JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_createErrorReporter(JNIEnv* env,
                                                            jclass clazz,
                                                            jint size) {
  BufferErrorReporter* error_reporter =
      new BufferErrorReporter(env, static_cast<int>(size));
  return reinterpret_cast<jlong>(error_reporter);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_deleteErrorReporter(JNIEnv* envv,
                                                            jclass clazz,
                                                            jlong error_reporter_handle) {
  delete reinterpret_cast<BufferErrorReporter*>(error_reporter_handle);
}

// private static native long createEngine(long configHandle);
JNIEXPORT jlong JNICALL 
Java_org_mrsnu_band_NativeEngineWrapper_createEngine(
    JNIEnv* env, jclass clazz, jlong config_handle) {
  RuntimeConfig* config = convertLongToConfig(env, config_handle);
  // Destroy unique_ptr, delete is a must.
  return reinterpret_cast<jlong>(Engine::Create(*config).release());
}

JNIEXPORT void JNICALL 
Java_org_mrsnu_band_NativeEngineWrapper_deleteEngine(
    JNIEnv* env, jclass clazz, jlong engine_handle) {
  delete reinterpret_cast<Engine*>(engine_handle);
}

// private static native long registerModel(long engineHandle, long
// modelHandle);
JNIEXPORT void JNICALL 
Java_org_mrsnu_band_NativeEngineWrapper_registerModel(
    JNIEnv* env, jclass clazz, jlong engine_handle, jlong model_handle) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  engine->RegisterModel(model);
}

// private static native long getNumInputTensors(long engineHandle, long
// modelHandle);
JNIEXPORT jint JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_getNumInputTensors(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong engine_handle,
                                                           jlong model_handle) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  return static_cast<jint>(engine->GetInputTensorIndices(model->GetId()).size());
}

// private static native long getNumOutputTensors(long engineHandle, long
// modelHandle);
JNIEXPORT jint JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_getNumOutputTensors(
    JNIEnv* env, jclass clazz, jlong engine_handle, jlong model_handle) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  return static_cast<jint>(engine->GetOutputTensorIndices(model->GetId()).size());
}

// private static native long createInputTensor(long engineHandle, long
// modelHandle, int index);
JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_createInputTensor(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong engine_handle,
                                                          jlong model_handle,
                                                          jint index) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  auto input_indices = engine->GetInputTensorIndices(model->GetId());
  Tensor* input_tensor = engine->CreateTensor(
      model->GetId(), input_indices[static_cast<int>(index)]);
  return reinterpret_cast<jlong>(input_tensor);
}

// private static native long createOutputTensor(long engineHandle, long
// modelHandle, int index);
JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_createOutputTensor(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong engine_handle,
                                                           jlong model_handle,
                                                           jint index) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  auto output_indices = engine->GetOutputTensorIndices(model->GetId());
  Tensor* output_tensor = engine->CreateTensor(
      model->GetId(), output_indices[static_cast<int>(index)]);
  return reinterpret_cast<jlong>(output_tensor);
}

// private static native long requestSync(long engineHandle, long modelHandle,
// List<Long> inputTensorHandles, List<Long> outputTensorHandles);
JNIEXPORT void JNICALL 
Java_org_mrsnu_band_NativeEngineWrapper_requestSync(
    JNIEnv* env, jclass clazz, jlong engine_handle, jlong model_handle,
    jobject input_tensor_handles, jobject output_tensor_handles) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  engine->InvokeSyncModel(model->GetId(),
                          convertLongListToTensors(env, input_tensor_handles),
                          convertLongListToTensors(env, output_tensor_handles));
}

// private static native long requestAsync(long engineHandle, long modelHandle,
// List<Long> inputTensorHandles);
JNIEXPORT jint JNICALL 
Java_org_mrsnu_band_NativeEngineWrapper_requestAsync(
    JNIEnv* env, jclass clazz, jlong engine_handle, jlong model_handle,
    jobject input_tensor_handles) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  Model* model = convertLongToModel(env, model_handle);
  auto job_id = engine->InvokeAsyncModel(
      model->GetId(), convertLongListToTensors(env, input_tensor_handles));
  return static_cast<jint>(job_id);
}

// private static native long wait(long engineHandle, long requestHandle,
// List<Long> outputTensorHandles);
JNIEXPORT void JNICALL 
Java_org_mrsnu_band_NativeEngineWrapper_wait(
    JNIEnv* env, jclass clazz, jlong engine_handle, jlong request_handle,
    jobject output_tensor_handles) {
  Engine* engine = convertLongToEngine(env, engine_handle);
  engine->Wait(static_cast<int>(request_handle),
               convertLongListToTensors(env, output_tensor_handles));
}