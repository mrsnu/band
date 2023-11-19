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
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addProfilePath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jstring profilePath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddProfilePath(ConvertJstringToString(env, profilePath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addLatencySmoothingFactor(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    float smoothingFactor) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddLatencySmoothingFactor(smoothingFactor);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addThermWindowSize(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint windowSize) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddThermalWindowSize(windowSize);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addCPUThermIndex(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, int cpuThermIndex) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddCPUThermIndex(cpuThermIndex);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addGPUThermIndex(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, int gpuThermIndex) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddGPUThermIndex(gpuThermIndex);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addDSPThermIndex(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, int dspThermIndex) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddDSPThermIndex(dspThermIndex);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addNPUThermIndex(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, int npuThermIndex) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddNPUThermIndex(npuThermIndex);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addTargetThermIndex(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    int targetThermIndex) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddTargetThermIndex(targetThermIndex);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addCPUFreqPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jstring cpuFreqPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddCPUFreqPath(ConvertJstringToString(env, cpuFreqPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addGPUFreqPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jstring gpuFreqPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddGPUFreqPath(ConvertJstringToString(env, gpuFreqPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addDSPFreqPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jstring dspFreqPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddDSPFreqPath(ConvertJstringToString(env, dspFreqPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addNPUFreqPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jstring npuFreqPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddNPUFreqPath(ConvertJstringToString(env, npuFreqPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addRuntimeFreqPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jstring runtimeFreqPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddRuntimeFreqPath(ConvertJstringToString(env, runtimeFreqPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addLatencyLogPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jstring latencyLogPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddLatencyLogPath(ConvertJstringToString(env, latencyLogPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addThermLogPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle,
    jstring thermLogPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddThermLogPath(ConvertJstringToString(env, thermLogPath));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addFreqLogPath(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jstring freqLogPath) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddFreqLogPath(ConvertJstringToString(env, freqLogPath));
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
    jint subgraphPreparationType) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddSubgraphPreparationType(
          static_cast<band::SubgraphPreparationType>(subgraphPreparationType));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_addCPUMask(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle, jint cpuMask) {
  ConvertLongToConfigBuilder(env, configBuilderHandle)
      ->AddCPUMask(static_cast<band::CPUMaskFlag>(cpuMask));
}

JNIEXPORT jboolean JNICALL
Java_org_mrsnu_band_NativeConfigBuilderWrapper_isValid(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle) {
  return static_cast<jboolean>(
      ConvertLongToConfigBuilder(env, configBuilderHandle)->IsValid());
}

JNIEXPORT jobject JNICALL Java_org_mrsnu_band_NativeConfigBuilderWrapper_build(
    JNIEnv* env, jclass clazz, jlong configBuilderHandle) {
  jclass config_cls = env->FindClass("org/mrsnu/band/Config");
  jmethodID config_ctor = env->GetMethodID(config_cls, "<init>", "(J)V");
  RuntimeConfigBuilder* b =
      ConvertLongToConfigBuilder(env, configBuilderHandle);
  return env->NewObject(
      config_cls, config_ctor,
      reinterpret_cast<jlong>(new JNIRuntimeConfig(b->Build())));
}

}  // extern "C"