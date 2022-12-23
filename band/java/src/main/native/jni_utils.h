#ifndef BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_
#define BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_

#include <jni.h>

#include "band/config.h"
#include "band/config_builder.h"
#include "band/engine.h"
#include "band/model.h"
#include "band/tensor.h"
#include "band/error_reporter.h"

namespace Band {
namespace jni {

extern const char kIllegalArgumentException[];
extern const char kNullPointerException[];

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...);

bool CheckJniInitializedOrThrow(JNIEnv* env);

class BufferErrorReporter : public ErrorReporter {
 public:
  BufferErrorReporter(JNIEnv* env, int limit);
  ~BufferErrorReporter() override;
  int Report(const char* format, va_list args) override;
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
    ThrowException(env, Band::jni::kIllegalArgumentException,
                   "Internal error: Found invalid handle");
    return nullptr;
  }
  return reinterpret_cast<T*>(handle);
}

Engine* convertLongToEngine(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Engine>(env, handle);
}

RuntimeConfigBuilder* convertLongToConfigBuilder(JNIEnv* env, jlong handle) {
  return CastLongToPointer<RuntimeConfigBuilder>(env, handle);
}

RuntimeConfig* convertLongToConfig(JNIEnv* env, jlong handle) {
  return CastLongToPointer<RuntimeConfig>(env, handle);
}

Model* convertLongToModel(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Model>(env, handle);
}

Tensor* convertLongToTensor(JNIEnv* env, jlong handle) {
  return CastLongToPointer<Tensor>(env, handle);
}

int convertLongToJobId(jint request_handle) {
  return static_cast<int>(request_handle);
}

Tensors convertLongListToTensors(JNIEnv* env, jobject tensor_handles) {
  static jclass list_class = env->FindClass("java/util/List");
  if (list_class == nullptr) {
    if (!env->ExceptionCheck()) {
      // TODO(widiba03304): handle error
    }
  }
  static jmethodID list_size_method =
      env->GetMethodID(list_class, "size", "()I");
  if (list_size_method == nullptr) {
    if (!env->ExceptionCheck()) {
      // TODO(widiba03304): handle error
    }
  }
  static jmethodID list_get_method =
      env->GetMethodID(list_class, "get", "(I)Ljava/lang/Object;");
  if (list_get_method == nullptr) {
    if (!env->ExceptionCheck()) {
      // TODO(widiba03304): handle error
    }
  }
  static jclass long_class = env->FindClass("java/lang/Long");
  if (long_class == nullptr) {
    if (!env->ExceptionCheck()) {
      // TODO(widiba03304): handle error
    }
  }
  static jmethodID long_value_method =
      env->GetMethodID(long_class, "longValue", "()J");
  if (long_value_method == nullptr) {
    if (!env->ExceptionCheck()) {
      // TODO(widiba03304): handle error
    }
  }

  jint size = env->CallIntMethod(tensor_handles, list_size_method);
  Tensors tensors;
  for (jint i = 0; i < size; i++) {
    jobject jtensor_handle =
        env->CallObjectMethod(tensor_handles, list_get_method, i);
    if (jtensor_handle == nullptr) {
      if (!env->ExceptionCheck()) {
        // TODO(widiba03304): handle error
      }
      return tensors;
    }
    jlong tensor_handle =
        env->CallLongMethod(jtensor_handle, long_value_method);
    if (tensor_handle == 0) {
      if (!env->ExceptionCheck()) {
        // TODO(widiba03304): handle error
      }
      return tensors;
    }
    Tensor* tensor = reinterpret_cast<Tensor*>(tensor_handle);
    tensors.push_back(tensor);
  }
  return tensors;
}

BufferErrorReporter* convertLongToErrorReporter(JNIEnv* env, jlong handle) {
  return CastLongToPointer<BufferErrorReporter>(env, handle);
}

}  // namespace jni
}  // namespace Band

#endif  // BAND_JAVA_SRC_MAIN_NATIVE_JNI_UTILS_H_