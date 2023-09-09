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

#include "band/java/src/main/native/jni_utils.h"

#include <stdarg.h>
#include <stdlib.h>

#include "jni_utils.h"

namespace band {
namespace jni {

const char kIllegalArgumentException[] = "java/lang/IllegalArgumentException";
const char kNullPointerException[] = "java/lang/NullPointerException";

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const size_t max_msg_len = 512;
  auto* message = static_cast<char*>(malloc(max_msg_len));
  if (message && (vsnprintf(message, max_msg_len, fmt, args) >= 0)) {
    env->ThrowNew(env->FindClass(clazz), message);
  } else {
    env->ThrowNew(env->FindClass(clazz), "");
  }
  if (message) {
    free(message);
  }
  va_end(args);
}

std::string ConvertJstringToString(JNIEnv* env, jstring jstr) {
  if (!jstr) {
    return "";
  }

  const jclass string_cls = env->GetObjectClass(jstr);
  const jmethodID get_bytes_method =
      env->GetMethodID(string_cls, "getBytes", "(Ljava/lang/String;)[B");
  const jbyteArray string_jbytes =
      static_cast<jbyteArray>(env->CallObjectMethod(
          jstr, get_bytes_method, env->NewStringUTF("UTF-8")));

  size_t length = static_cast<size_t>(env->GetArrayLength(string_jbytes));
  jbyte* bytes = env->GetByteArrayElements(string_jbytes, nullptr);

  std::string ret = std::string(reinterpret_cast<char*>(bytes), length);
  env->ReleaseByteArrayElements(string_jbytes, bytes, JNI_ABORT);
  env->DeleteLocalRef(string_jbytes);
  env->DeleteLocalRef(string_cls);
  return ret;
}

Engine* ConvertLongToEngine(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Engine>(env, handle);
}

RuntimeConfigBuilder* ConvertLongToConfigBuilder(JNIEnv* env, jlong handle) {
  return CastLongToPointer<RuntimeConfigBuilder>(env, handle);
}

RuntimeConfig* ConvertLongToConfig(JNIEnv* env, jlong handle) {
  return CastLongToPointer<RuntimeConfig>(env, handle);
}

Model* ConvertLongToModel(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Model>(env, handle);
}

Tensor* ConvertLongToTensor(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Tensor>(env, handle);
}

Buffer* ConvertLongToBuffer(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Buffer>(env, handle);
}

BufferProcessor* ConvertLongToBufferProcessor(JNIEnv* env, jlong handle) {
  return CastLongToPointer<BufferProcessor>(env, handle);
}

int ConvertLongToJobId(jint request_handle) {
  return static_cast<int>(request_handle);
}

}  // namespace jni
}  // namespace band