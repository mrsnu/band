#include <jni.h>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/java/src/main/native/jni_utils.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/tensor.h"

using Band::Engine;
using Band::Model;
using Band::RuntimeConfig;
using Band::RuntimeConfigBuilder;
using Band::Tensor;
using Band::Tensors;
using Band::jni::BufferErrorReporter;
using Band::jni::ConvertLongListToTensors;
using Band::jni::ConvertLongToConfig;
using Band::jni::ConvertLongToEngine;
using Band::jni::ConvertLongToJobId;
using Band::jni::ConvertLongToModel;
using Band::jni::ConvertLongToTensor;

namespace {

RuntimeConfig* ConvertJobjectToConfig(JNIEnv* env, jobject config) {
  JNI_DEFINE_CLS_AND_MTD(cfg, "org/mrsnu/band/Config", "getNativeHandle",
                         "()J");
  return ConvertLongToConfig(env, env->CallLongMethod(config, cfg_mtd));
}

Model* ConvertJobjectToModel(JNIEnv* env, jobject model) {
  JNI_DEFINE_CLS_AND_MTD(mdl, "org/mrsnu/band/Model", "getNativeHandle", "()J");
  return ConvertLongToModel(env, env->CallLongMethod(model, mdl_mtd));
}

Tensors ConvertJobjectToTensors(JNIEnv* env, jobject tensor_list) {
  JNI_DEFINE_CLS_AND_MTD(tnr, "org/mrsnu/band/Tensor", "getNativeHandle",
                         "()J");
  JNI_DEFINE_CLS(list, "java/util/List");
  JNI_DEFINE_MTD(list_size, list_cls, "size", "()I");
  JNI_DEFINE_MTD(list_get, list_cls, "get", "(I)Ljava/lang/Object;");

  jint size = env->CallIntMethod(tensor_list, list_size_mtd);
  Tensors tensors;
  for (int i = 0; i < size; i++) {
    jobject tensor = env->CallObjectMethod(tensor_list, list_get_mtd, i);
    jlong tensor_handle = env->CallLongMethod(tensor, tnr_mtd);
    tensors.push_back(ConvertLongToTensor(env, tensor_handle));
  }
  return tensors;
}

}  // anonymous namespace

extern "C" {

// private static native long createEngine(long configHandle);
JNIEXPORT jlong JNICALL Java_org_mrsnu_band_NativeEngineWrapper_createEngine(
    JNIEnv* env, jclass clazz, jobject config) {
  RuntimeConfig* native_config = ConvertJobjectToConfig(env, config);
  return reinterpret_cast<jlong>(Engine::Create(*native_config).release());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_deleteEngine(
    JNIEnv* env, jclass clazz, jlong engineHandle) {
  delete reinterpret_cast<Engine*>(engineHandle);
}

// private static native long registerModel(long engineHandle, long
// modelHandle);
JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_registerModel(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  engine->RegisterModel(native_model);
}

// private static native long getNumInputTensors(long engineHandle, long
// modelHandle);
JNIEXPORT jint JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_getNumInputTensors(JNIEnv* env,
                                                           jclass clazz,
                                                           jlong engineHandle,
                                                           jobject model) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  return static_cast<jint>(
      engine->GetInputTensorIndices(native_model->GetId()).size());
}

// private static native long getNumOutputTensors(long engineHandle, long
// modelHandle);
JNIEXPORT jint JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_getNumOutputTensors(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong engineHandle,
                                                            jobject model) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  return static_cast<jint>(
      engine->GetOutputTensorIndices(native_model->GetId()).size());
}

// private static native long createInputTensor(long engineHandle, long
// modelHandle, int index);
JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_createInputTensor(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model, jint index) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  auto input_indices = engine->GetInputTensorIndices(native_model->GetId());
  Tensor* input_tensor = engine->CreateTensor(
      native_model->GetId(), input_indices[static_cast<int>(index)]);
  return reinterpret_cast<jlong>(input_tensor);
}

// private static native long createOutputTensor(long engineHandle, long
// modelHandle, int index);
JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_createOutputTensor(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model, jint index) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  auto output_indices = engine->GetOutputTensorIndices(native_model->GetId());
  Tensor* output_tensor = engine->CreateTensor(
      native_model->GetId(), output_indices[static_cast<int>(index)]);
  return reinterpret_cast<jlong>(output_tensor);
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_requestSync(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model,
    jobject input_tensor_handles, jobject output_tensor_handles) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  Tensors input_tensors = ConvertJobjectToTensors(env, input_tensor_handles);
  Tensors output_tensors = ConvertJobjectToTensors(env, output_tensor_handles);
  float* input_raw = reinterpret_cast<float*>(input_tensors[0]->GetData());
  float* output_raw = reinterpret_cast<float*>(output_tensors[0]->GetData());
  engine->RequestSync(native_model->GetId(), BandGetDefaultRequestOption(),
                      input_tensors, output_tensors);
}

// private static native long requestAsync(long engineHandle, long modelHandle,
// List<Long> inputTensors);
JNIEXPORT jint JNICALL Java_org_mrsnu_band_NativeEngineWrapper_requestAsync(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model,
    jobject input_tensor_handles) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  auto job_id =
      engine->RequestAsync(native_model->GetId(), BandGetDefaultRequestOption(),
                           ConvertLongListToTensors(env, input_tensor_handles));
  return static_cast<jint>(job_id);
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_wait(
    JNIEnv* env, jclass clazz, jlong engineHandle, jlong request_handle,
    jobject output_tensor_handles) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  engine->Wait(static_cast<int>(request_handle),
               ConvertLongListToTensors(env, output_tensor_handles));
}

}  // extern "C"