#include <jni.h>

#include <algorithm>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/java/src/main/native/jni_utils.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/tensor.h"

using band::Engine;
using band::Model;
using band::ModelId;
using band::RuntimeConfig;
using band::RuntimeConfigBuilder;
using band::Tensor;
using band::Tensors;
using band::jni::BufferErrorReporter;
using band::jni::ConvertListToVectorOfPointer;
using band::jni::ConvertLongToConfig;
using band::jni::ConvertLongToEngine;
using band::jni::ConvertLongToJobId;
using band::jni::ConvertLongToModel;
using band::jni::ConvertLongToTensor;

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

jintArray ConvertNativeToIntArray(JNIEnv* env, jsize length, const int* array) {
  jintArray arr = env->NewIntArray(length);
  env->SetIntArrayRegion(arr, 0, length, reinterpret_cast<const int*>(array));
  return arr;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_mrsnu_band_NativeEngineWrapper_createEngine(
    JNIEnv* env, jclass clazz, jobject config) {
  RuntimeConfig* native_config = ConvertJobjectToConfig(env, config);
  return reinterpret_cast<jlong>(Engine::Create(*native_config).release());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_deleteEngine(
    JNIEnv* env, jclass clazz, jlong engineHandle) {
  delete reinterpret_cast<Engine*>(engineHandle);
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_registerModel(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model) {
  BAND_LOG_PROD(band::BAND_LOG_INFO, "Registering model: %p, %p", engineHandle,
                model);
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  BAND_LOG_PROD(band::BAND_LOG_INFO, "Registering model: %p, %p", engine,
                native_model);
  auto status = engine->RegisterModel(native_model);
  if (!status.ok()) {
    // TODO(widiba03304): refactor absl
    return;
  }
}

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
    jobject input_tensor_handles, jobject output_tensor_handles,
    jint target_worker, jboolean require_callback, jint slo_us,
    jfloat slo_scale) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);

  JNI_DEFINE_CLS_AND_MTD(tnr, "org/mrsnu/band/Tensor", "getNativeHandle",
                         "()J");
  Tensors input_tensors =
      ConvertListToVectorOfPointer<band::interface::ITensor>(
          env, input_tensor_handles, tnr_mtd);
  Tensors output_tensors =
      ConvertListToVectorOfPointer<band::interface::ITensor>(
          env, output_tensor_handles, tnr_mtd);

  float* input_raw = reinterpret_cast<float*>(input_tensors[0]->GetData());
  float* output_raw = reinterpret_cast<float*>(output_tensors[0]->GetData());
  auto status = engine->RequestSync(
      native_model->GetId(),
      {target_worker, static_cast<bool>(require_callback), slo_us, slo_scale},
      input_tensors, output_tensors);
  if (!status.ok()) {
    // TODO(widiba03304): refactor absl
    return;
  }
}

JNIEXPORT jint JNICALL Java_org_mrsnu_band_NativeEngineWrapper_requestAsync(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject model,
    jobject input_tensor_handles, jint target_worker, jboolean require_callback,
    jint slo_us, jfloat slo_scale) {
  JNI_DEFINE_CLS_AND_MTD(tnr, "org/mrsnu/band/Tensor", "getNativeHandle",
                         "()J");
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  Model* native_model = ConvertJobjectToModel(env, model);
  auto job_id =
      engine
          ->RequestAsync(native_model->GetId(),
                         band::RequestOption::GetDefaultOption(),
                         ConvertListToVectorOfPointer<band::interface::ITensor>(
                             env, input_tensor_handles, tnr_mtd))
          .value();
  return static_cast<jint>(job_id);
}

JNIEXPORT jintArray JNICALL
Java_org_mrsnu_band_NativeEngineWrapper_requestAsyncBatch(
    JNIEnv* env, jclass clazz, jlong engineHandle, jobject models,
    jobject inputTensorsList, jintArray targetWorkersList,
    jbooleanArray requireCallbacksList, jintArray sloUsList,
    jfloatArray sloScaleList) {
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  JNI_DEFINE_CLS_AND_MTD(mdl, "org/mrsnu/band/Model", "getNativeHandle", "()J");
  JNI_DEFINE_CLS_AND_MTD(tnr, "org/mrsnu/band/Tensor", "getNativeHandle",
                         "()J");
  JNI_DEFINE_CLS(list, "java/util/List");
  JNI_DEFINE_MTD(list_size, list_cls, "size", "()I");
  JNI_DEFINE_MTD(list_get, list_cls, "get", "(I)Ljava/lang/Object;");

  std::vector<Model*> model_list =
      ConvertListToVectorOfPointer<Model>(env, models, mdl_mtd);
  std::vector<ModelId> model_ids;
  for (Model* model : model_list) {
    model_ids.push_back(model->GetId());
  }
  std::vector<band::RequestOption> request_options;
  for (int i = 0; i < model_list.size(); i++) {
    int target_worker = env->GetIntArrayElements(targetWorkersList, nullptr)[i];
    bool require_callback =
        env->GetBooleanArrayElements(requireCallbacksList, nullptr)[i];
    int slo_us = env->GetIntArrayElements(sloUsList, nullptr)[i];
    float slo_scale = env->GetFloatArrayElements(sloScaleList, nullptr)[i];
    request_options.push_back(
        {target_worker, require_callback, slo_us, slo_scale});
  }

  std::vector<band::Tensors> input_lists;
  for (int i = 0; i < model_list.size(); i++) {
    jobject input_list =
        env->CallObjectMethod(inputTensorsList, list_get_mtd, i);
    input_lists.push_back(
        ConvertListToVectorOfPointer<band::interface::ITensor>(env, input_list,
                                                               tnr_mtd));
  }
  std::vector<int> ret =
      engine->RequestAsync(model_ids, request_options, input_lists).value();
  return ConvertNativeToIntArray(env, ret.size(), ret.data());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeEngineWrapper_wait(
    JNIEnv* env, jclass clazz, jlong engineHandle, jint jobId,
    jobject outputTensors) {
  JNI_DEFINE_CLS_AND_MTD(tnr, "org/mrsnu/band/Tensor", "getNativeHandle",
                         "()J");
  Engine* engine = ConvertLongToEngine(env, engineHandle);
  auto status = engine->Wait(
      jobId, ConvertListToVectorOfPointer<band::interface::ITensor>(
                 env, outputTensors, tnr_mtd));
  if (!status.ok()) {
    // TODO(widiba03304): refactor absl
    return;
  }
}

}  // extern "C"