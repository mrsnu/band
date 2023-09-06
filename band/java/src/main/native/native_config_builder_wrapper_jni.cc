// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>

#include "band/config_builder.h"
#include "band/java/src/main/native/jni_utils.h"
#include "band/logger.h"

using band::RuntimeConfigBuilder;
using band::jni::ConvertJstringToString;
using band::jni::ConvertLongToConfigBuilder;
using band::jni::JNIRuntimeConfig;

namespace {

template <typename T>
std::vector<T> ConvertIntArrayTo(JNIEnv* env, jintArray array) {
  size_t length = static_cast<size_t>(env->GetArrayLength(array));
  std::vector<T> ret;
  jint* array_ptr = env->GetIntArrayElements(array, /*isCopy=*/nullptr);
  for (int i = 0; i < length; i++) {
    // Note T should not be a pointer type.
    ret.push_back(static_cast<T>(array_ptr[i]));
  }
  return ret;
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_createConfigBuilder(
    JNIEnv* env, jclass clazz) {
  RuntimeConfigBuilder* b = new RuntimeConfigBuilder();
  return reinterpret_cast<jlong>(b);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_deleteConfigBuilder(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle) {
  delete reinterpret_cast<RuntimeConfigBuilder*>(configBuilderHandle);
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeConfigBuilderWrapper_addOnline(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jboolean online) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)->AddOnline(online);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addNumWarmups(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint numWarmups) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddNumWarmups(numWarmups);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addNumRuns(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint numRuns) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)->AddNumRuns(numRuns);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addProfileDataPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jstring profileDataPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddProfileDataPath(ConvertJstringToString(env, profileDataPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addSmoothingFactor(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jfloat smoothingFactor) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddSmoothingFactor(static_cast<float>(smoothingFactor));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addPlannerLogPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jstring plannerLogPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddPlannerLogPath(ConvertJstringToString(env, plannerLogPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addScheduleWindowSize(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jint scheduleWindowSize) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddScheduleWindowSize(static_cast<int>(scheduleWindowSize));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addSchedulers(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jintArray schedulers) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddSchedulers(ConvertIntArrayTo<band::SchedulerType>(env, schedulers));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addPlannerCPUMask(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint cpuMask) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddPlannerCPUMask(static_cast<band::CPUMaskFlag>(cpuMask));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addWorkers(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jintArray workers) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddWorkers(ConvertIntArrayTo<band::DeviceFlag>(env, workers));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addWorkerCPUMasks(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jintArray cpuMasks) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddWorkerCPUMasks(ConvertIntArrayTo<band::CPUMaskFlag>(env, cpuMasks));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addWorkerNumThreads(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jintArray numThreads) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddWorkerNumThreads(ConvertIntArrayTo<int>(env, numThreads));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addAllowWorkSteal(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jboolean allowWorkSteal) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddAllowWorkSteal(static_cast<bool>(allowWorkSteal));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addAvailabilityCheckIntervalMs(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jint availabilityCheckIntervalMs) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddAvailabilityCheckIntervalMs(
          static_cast<int>(availabilityCheckIntervalMs));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addMinimumSubgraphSize(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jint minimumSubgraphSize) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddMinimumSubgraphSize(static_cast<int>(minimumSubgraphSize));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addSubgraphPreparationType(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jint subgaphPreparationType) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddSubgraphPreparationType(
          static_cast<band::SubgraphPreparationType>(subgaphPreparationType));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addCPUMask(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint cpuMask) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddCPUMask(static_cast<band::CPUMaskFlag>(cpuMask));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addResourceMonitorDeviceFreqPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint deviceFlag,
    jstring devicePath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddResourceMonitorDeviceFreqPath(
          static_cast<band::DeviceFlag>(deviceFlag),
          ConvertJstringToString(env, devicePath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addResourceMonitorIntervalMs(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jint resourceMonitorIntervalMs) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddResourceMonitorIntervalMs(
          static_cast<int>(resourceMonitorIntervalMs));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addResourceMonitorLogPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jstring resourceMonitorLogPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddResourceMonitorLogPath(
          ConvertJstringToString(env, resourceMonitorLogPath));
}

JNIEXPORT jobject JNICALL Java_org_mrsnu_band_NativeConfigBuilderWrapper_build(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle) {
  jclass config_cls = env->FindClass("org/mrsnu/band/Config");
  jmethodID config_ctor = env->GetMethodID(config_cls, "<init>", "(J)V");
  RuntimeConfigBuilder* b =
      ConvertLongToConfigBuilder(env, configBuilderHandle);
  auto runtime_config = b->Build();
  if (!runtime_config.ok()) {
    return nullptr;
  } else {
    return env->NewObject(
        config_cls, config_ctor,
        reinterpret_cast<jlong>(new JNIRuntimeConfig(runtime_config.value())));
  }
}

}  // extern "C"