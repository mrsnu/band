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

#include "band/java/src/main/native/jni_utils.h"
#include "band/model.h"

using band::Model;
using band::jni::ConvertLongToModel;

namespace {

int ConvertBackendTypeToInt(band::BackendType backend_type) {
  return static_cast<size_t>(backend_type);
}

band::BackendType ConvertJintToBackendType(jint backend_type) {
  return static_cast<band::BackendType>(backend_type);
}

}  // anonymous namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeModelWrapper_createModel(JNIEnv* env, jclass clazz) {
  return reinterpret_cast<jlong>(new Model());
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeModelWrapper_deleteModel(
    JNIEnv* env, jclass clazz, jlong modelHandle) {
  delete reinterpret_cast<Model*>(modelHandle);
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeModelWrapper_loadFromFile(
    JNIEnv* env, jclass clazz, jlong modelHandle, jint backendType,
    jstring filePath) {
  Model* model = ConvertLongToModel(env, modelHandle);
  const char* nativeFilePath = env->GetStringUTFChars(filePath, nullptr);
  auto status = model->FromPath(ConvertJintToBackendType(backendType), nativeFilePath);
  if (!status.ok()) {
    // TODO(widiba03304): refactor absl
    return;
  }
}

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeModelWrapper_loadFromBuffer(
    JNIEnv* env, jclass clazz, jlong modelHandle, jint backendType,
    jobject modelBuffer) {
  Model* model = ConvertLongToModel(env, modelHandle);
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(modelBuffer));
  size_t size = env->GetDirectBufferCapacity(modelBuffer);
  auto status = model->FromBuffer(ConvertJintToBackendType(backendType), buf, size);
  if (!status.ok()) {
    // TODO(widiba03304): refactor absl
    return;
  }
}

JNIEXPORT jintArray JNICALL
Java_org_mrsnu_band_NativeModelWrapper_getSupportedBackends(JNIEnv* env,
                                                            jclass clazz,
                                                            jlong modelHandle) {
  Model* model = ConvertLongToModel(env, modelHandle);
  auto supported_backend = model->GetSupportedBackends();
  jintArray ret = env->NewIntArray(supported_backend.size());
  int* backend_array = new int[supported_backend.size()];
  int i = 0;
  for (auto backend : supported_backend) {
    backend_array[i] = ConvertBackendTypeToInt(backend);
    i++;
  }
  env->SetIntArrayRegion(ret, 0, supported_backend.size(), backend_array);

  return ret;
}

}  // extern "C"