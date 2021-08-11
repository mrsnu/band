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

#include "tensorflow/lite/java/src/main/native/jni_utils.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

const char kIllegalArgumentException[] = "java/lang/IllegalArgumentException";
const char kIllegalStateException[] = "java/lang/IllegalStateException";
const char kNullPointerException[] = "java/lang/NullPointerException";
const char kIndexOutOfBoundsException[] = "java/lang/IndexOutOfBoundsException";
const char kUnsupportedOperationException[] =
    "java/lang/UnsupportedOperationException";

namespace tflite {
namespace jni {
TensorHandle::TensorHandle(TfLiteTensor* tensor)
  : tensor_(tensor) {}

TfLiteTensor* TensorHandle::tensor() const { return tensor_; }

TfLiteTensor* GetTensorFromHandle(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    ThrowException(env, kIllegalArgumentException,
                   "Internal error: Invalid handle to TfLiteTensor.");
    return nullptr;
  }
  return reinterpret_cast<TensorHandle*>(handle)->tensor();
}

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

BufferErrorReporter::BufferErrorReporter(JNIEnv* env, int limit) {
  buffer_ = new char[limit];
  if (!buffer_) {
    ThrowException(env, kNullPointerException,
                   "Internal error: Malloc of BufferErrorReporter to hold %d "
                   "char failed.",
                   limit);
    return;
  }
  buffer_[0] = '\0';
  limit_ = limit;
}

BufferErrorReporter::~BufferErrorReporter() { delete[] buffer_; }

int BufferErrorReporter::Report(const char* format, va_list args) {
  return vsnprintf(buffer_, limit_, format, args);
}

const char* BufferErrorReporter::CachedLastErrorMessage() { return buffer_; }

}  // namespace jni
}  // namespace tflite
