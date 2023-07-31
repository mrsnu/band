#include "band/java/src/main/native/jni_utils.h"

#include <stdarg.h>
#include <stdlib.h>

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
  start_idx_ = 0;
  end_idx_ = limit - 1;
}

BufferErrorReporter::~BufferErrorReporter() { delete[] buffer_; }

int BufferErrorReporter::Report(const char* format, va_list args) {
  int size = 0;
  if (start_idx_ > 0 && start_idx_ < end_idx_) {
    buffer_[start_idx_++] = '\n';
    ++size;
  }
  if (start_idx_ < end_idx_) {
    size = vsnprintf(buffer_ + start_idx_, end_idx_ - start_idx_, format, args);
  }
  start_idx_ += size;
  return size;
}

const char* BufferErrorReporter::CachedErrorMessage() { return buffer_; }

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

int ConvertLongToJobId(jint request_handle) {
  return static_cast<int>(request_handle);
}

BufferErrorReporter* ConvertLongToErrorReporter(JNIEnv* env, jlong handle) {
  return CastLongToPointer<BufferErrorReporter>(env, handle);
}

}  // namespace jni
}  // namespace band