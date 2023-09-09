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

package org.mrsnu.band;

import java.nio.ByteBuffer;

class NativeImageProcessorBuilderWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeImageProcessorBuilderWrapper() {
    nativeHandle = createImageProcessorBuilder();
  }

  public void addCrop(int x, int y, int width, int height) {
    addCrop(nativeHandle, x, y, width, height);
  }

  public void addResize(int width, int height) {
    addResize(nativeHandle, width, height);
  }

  public void addRotate(int angle) {
    addRotate(nativeHandle, angle);
  }

  public void addFlip(boolean horizontal, boolean vertical) {
    addFlip(nativeHandle, horizontal, vertical);
  }

  public void addColorSpaceConvert(BufferFormat dstColorSpace) {
    addColorSpaceConvert(nativeHandle, dstColorSpace.getValue());
  }

  public void addNormalize(float mean, float std) {
    addNormalize(nativeHandle, mean, std);
  }

  public void addDataTypeConvert() {
    addDataTypeConvert(nativeHandle);
  }

  public ImageProcessor build() {
    return (ImageProcessor) build(nativeHandle);
  }

  @Override
  public void close() {
    deleteImageProcessorBuilder(nativeHandle);
    nativeHandle = 0;
  }

  private native long createImageProcessorBuilder();

  private native void deleteImageProcessorBuilder(long imageProcessorBuilderHandle);

  private native void addCrop(
      long imageProcessorBuilderHandle, int x, int y, int width, int height);

  private native void addResize(long imageProcessorBuilderHandle, int width, int height);

  private native void addRotate(long imageProcessorBuilderHandle, int angle);

  private native void addFlip(
      long imageProcessorBuilderHandle, boolean horizontal, boolean vertical);

  private native void addColorSpaceConvert(long imageProcessorBuilderHandle, int dstColorSpace);

  private native void addNormalize(long imageProcessorBuilderHandle, float mean, float std);

  private native void addDataTypeConvert(long imageProcessorBuilderHandle);

  private native Object build(long imageProcessorBuilderHandle);
}