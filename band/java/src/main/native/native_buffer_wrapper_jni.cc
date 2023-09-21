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
using band::jni::ConvertJObjectToPointer;
using band::jni::ConvertLongToTensor;

extern "C" {

JNIEXPORT void JNICALL Java_org_mrsnu_band_NativeBufferWrapper_deleteBuffer(
    JNIEnv* env, jclass clazz, jlong bufferHandle) {
  delete reinterpret_cast<Buffer*>(bufferHandle);
}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeBufferWrapper_createFromTensor(JNIEnv* env,
                                                         jclass clazz,
                                                         jobject tensorObject) {
  Tensor* tensor = ConvertJObjectToPointer<Tensor>(env, "org/mrsnu/band/Tensor",
                                                   tensorObject);
  Buffer* buffer = Buffer::CreateFromTensor(tensor);
  return reinterpret_cast<jlong>(buffer);
}

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeBufferWrapper_createFromByteBuffer(
    JNIEnv* env, jclass clazz, jobject raw_buffer, jint width, jint height,
    jint bufferFormat) {
  Buffer* buffer =
      Buffer::CreateFromRaw(reinterpret_cast<const unsigned char*>(
                                env->GetDirectBufferAddress(raw_buffer)),
                            width, height, band::BufferFormat(bufferFormat));
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
