/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_
#define BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_

#include <jni.h>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/error_reporter.h"
#include "band/logger.h"
#include "band/model.h"
#include "band/tensor.h"

namespace band {
namespace jni {

#define JNI_DEFINE_CLS(tag, cls)                                            \
  static jclass tag##_cls = env->FindClass(cls);                            \
  if (tag##_cls == nullptr) {                                               \
    BAND_LOG_INTERNAL(band::BAND_LOG_ERROR, "Canont find class named `%s`", \
                      cls);                                                 \
  }

#define JNI_DEFINE_MTD(tag, cls_var, mtd, sig)                             \
  static jmethodID tag##_mtd = env->GetMethodID(cls_var, mtd, sig);        \
  if (tag##_mtd == nullptr) {                                              \
    BAND_LOG_INTERNAL(band::BAND_LOG_ERROR,                                \
                      "Cannot find method named `%s` with signature `%s`", \
                      mtd, sig);                                           \
  }

#define JNI_DEFINE_CLS_AND_MTD(tag, cls, mtd, sig) \
  JNI_DEFINE_CLS(tag, cls)                         \
  JNI_DEFINE_MTD(tag, tag##_cls, mtd, sig);

extern const char kIllegalArgumentException[];
extern const char kNullPointerException[];

struct JNIRuntimeConfig {
  JNIRuntimeConfig(RuntimeConfig config) : impl(config) {}

  RuntimeConfig impl;
};

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...);

bool CheckJniInitializedOrThrow(JNIEnv* env);

class BufferErrorReporter : public ErrorReporter {
 public:
  BufferErrorReporter(JNIEnv* env, int limit);
  ~BufferErrorReporter() override;
  int Report(const char* format, va_list args);
  const char* CachedErrorMessage();
  using ErrorReporter::Report;

 private:
  char* buffer_;
  int start_idx_ = 0;
  int end_idx_ = 0;
};

template <typename T>
T* CastLongToPointer(JNIEnv* env, jlong handle) {
  if (handle == 0 || handle == -1) {
    ThrowException(env, band::jni::kIllegalArgumentException,
                   "Internal error: Found invalid handle");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

template <typename T>
std::vector<T*> ConvertListToVectorOfPointer(JNIEnv* env, jobject object,
                                             jmethodID mtd) {
  JNI_DEFINE_CLS(list, "java/util/List");
  JNI_DEFINE_MTD(list_size, list_cls, "size", "()I");
  JNI_DEFINE_MTD(list_get, list_cls, "get", "(I)Ljava/lang/Object;");

  jint size = env->CallIntMethod(object, list_size_mtd);
  std::vector<T*> result;
  for (int i = 0; i < size; i++) {
    jobject element = env->CallObjectMethod(object, list_get_mtd, i);
    jlong native_handle = env->CallLongMethod(element, mtd);
    result.push_back(CastLongToPointer<T>(env, native_handle));
  }
  return result;
}

std::string ConvertJstringToString(JNIEnv* env, jstring jstr);

Engine* ConvertLongToEngine(JNIEnv* env, jlong handle);

RuntimeConfigBuilder* ConvertLongToConfigBuilder(JNIEnv* env, jlong handle);

RuntimeConfig* ConvertLongToConfig(JNIEnv* env, jlong handle);

Model* ConvertLongToModel(JNIEnv* env, jlong handle);

Tensor* ConvertLongToTensor(JNIEnv* env, jlong handle);

int ConvertLongToJobId(jint request_handle);

BufferErrorReporter* ConvertLongToErrorReporter(JNIEnv* env, jlong handle);

}  // namespace jni
}  // namespace band

#endif  // BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_