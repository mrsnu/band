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
#include "band/buffer/image_operator.h"
#include "band/buffer/image_processor.h"
#include "band/java/src/main/native/jni_utils.h"

using band::BufferFormat;
using namespace band::buffer;
using band::ImageProcessorBuilder;

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_createImageProcessorBuilder(
    JNIEnv* env, jclass clazz) {
  ImageProcessorBuilder* b = new ImageProcessorBuilder();
  return reinterpret_cast<jlong>(b);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_deleteImageProcessorBuilder(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle) {
  delete reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle);
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addCrop(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle, jint x0,
    jint y0, jint x1, jint y1) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<Crop>(x0, y0, x1, y1));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addResize(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle, jint width,
    jint height) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<Resize>(width, height));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addRotate(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle,
    jfloat angle_deg) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<Rotate>(angle_deg));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addFlip(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle,
    jboolean horizontal, jboolean vertical) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<Flip>(horizontal, vertical));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addColorSpaceConvert(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle, jint format) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<ColorSpaceConvert>(BufferFormat(format)));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addNormalize(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle, jfloat mean,
    jfloat std) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<Normalize>(mean, std, false));
}

JNIEXPORT void JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_addDataTypeConvert(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle) {
  reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
      ->AddOperation(std::make_unique<DataTypeConvert>());
}

JNIEXPORT jobject JNICALL
Java_org_mrsnu_band_NativeImageProcessorBuilderWrapper_build(
    JNIEnv* env, jclass clazz, jlong imageProcessorBuilderHandle) {
  jclass imageProcessor_cls = env->FindClass("org/mrsnu/band/ImageProcessor");
  jmethodID imageProcessor_constructor =
      env->GetMethodID(imageProcessor_cls, "<init>", "(J)V");
  auto status =
      reinterpret_cast<ImageProcessorBuilder*>(imageProcessorBuilderHandle)
          ->Build();
  if (!status.ok()) {
    BAND_LOG(band::LogSeverity::kError, "Failed to build ImageProcessor: %s",
                  status.status().message().data());
    return nullptr;
  } else {
    return env->NewObject(imageProcessor_cls, imageProcessor_constructor,
                          reinterpret_cast<jlong>(status.value().release()));
  }
}
}
