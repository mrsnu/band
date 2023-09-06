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

#include <jni.h>

#include "band/buffer/buffer.h"
#include "band/java/src/main/native/jni_utils.h"

using band::Buffer;
using band::Tensor;
using band::jni::ConvertLongToTensor;

extern "C" {

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeBufferWrapper_deleteBuffer(
    JNIEnv* env, jclass clazz, jlong bufferHandle) {
  delete reinterpret_cast<Buffer*>(bufferHandle);
}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeBufferWrapper_createFromTensor(JNIEnv* env,
                                                         jclass clazz,
                                                         jlong tensorHandle) {
  Tensor* tensor = ConvertLongToTensor(env, tensorHandle);
  Buffer* buffer = Buffer::CreateFromTensor(tensor);
  return reinterpret_cast<jlong>(buffer);
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
