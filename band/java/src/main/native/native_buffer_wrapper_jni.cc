#include <jni.h>

#include "band/buffer/buffer.h"
#include "band/java/src/main/native/jni_utils.h"

using band::Buffer;

extern "C" {

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeBufferWrapper_deleteBuffer(
    JNIEnv* env, jclass clazz, jlong bufferHandle) {
  delete reinterpret_cast<Buffer*>(bufferHandle);
}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeBufferWrapper_createFromByteBuffer(
    JNIEnv* env, jclass clazz, jbyteArray raw_buffer, jint width, jint height,
    jint bufferFormat) {
  jbyte* raw_buffer_bytes = env->GetByteArrayElements(raw_buffer, nullptr);
  Buffer* buffer = Buffer::CreateFromRaw(
      reinterpret_cast<const unsigned char*>(raw_buffer_bytes), width, height,
      band::BufferFormat(bufferFormat));
  env->ReleaseByteArrayElements(raw_buffer, raw_buffer_bytes, 0);
  return reinterpret_cast<jlong>(buffer);
}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeBufferWrapper_createFromYUVBuffer(
    JNIEnv* env, jclass clazz, jbyteArray y, jbyteArray u, jbyteArray v,
    jint width, jint height, jint yRowStride, jint uvRowStride,
    jint uvPixelStride, jint bufferFormat) {
  jbyte* y_bytes = env->GetByteArrayElements(y, nullptr);
  jbyte* u_bytes = env->GetByteArrayElements(u, nullptr);
  jbyte* v_bytes = env->GetByteArrayElements(v, nullptr);
  Buffer* buffer = Buffer::CreateFromYUVPlanes(
      reinterpret_cast<const unsigned char*>(y_bytes),
      reinterpret_cast<const unsigned char*>(u_bytes),
      reinterpret_cast<const unsigned char*>(v_bytes), width, height,
      yRowStride, uvRowStride, uvPixelStride, band::BufferFormat(bufferFormat));
  env->ReleaseByteArrayElements(y, y_bytes, 0);
  env->ReleaseByteArrayElements(u, u_bytes, 0);
  env->ReleaseByteArrayElements(v, v_bytes, 0);
  return reinterpret_cast<jlong>(buffer);
}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeBufferWrapper_createFromYUVPlanes(
    JNIEnv* env, jclass clazz, jobject y, jobject u, jobject v, jint width,
    jint height, jint yRowStride, jint uvRowStride, jint uvPixelStride,
    jint bufferFormat) {
  Buffer* buffer = Buffer::CreateFromYUVPlanes(
      reinterpret_cast<const unsigned char*>(env->GetDirectBufferAddress(y)),
      reinterpret_cast<const unsigned char*>(env->GetDirectBufferAddress(u)),
      reinterpret_cast<const unsigned char*>(env->GetDirectBufferAddress(v)),
      width, height, yRowStride, uvRowStride, uvPixelStride,
      band::BufferFormat(bufferFormat));
  return reinterpret_cast<jlong>(buffer);
}

}  // extern "C"
