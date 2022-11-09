#include "band/java/src/main/native/jni_utils.h"

#include <stdarg.h>
#include <stdlib.h>

namespace Band {
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

}  // namespace jni
}  // namespace Band