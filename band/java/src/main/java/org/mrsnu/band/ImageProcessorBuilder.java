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

public class ImageProcessorBuilder implements AutoCloseable {
  private NativeImageProcessorBuilderWrapper wrapper;

  public ImageProcessorBuilder() {
    Band.init();
    wrapper = new NativeImageProcessorBuilderWrapper();
  }

  @Override
  public void close() {
    wrapper = null;
  }

  public ImageProcessorBuilder addCrop(int x0, int y0, int x1, int y1) {
    wrapper.addCrop(x0, y0, x1, y1);
    return this;
  }

  public ImageProcessorBuilder addResize(int width, int height) {
    wrapper.addResize(width, height);
    return this;
  }

  public ImageProcessorBuilder addRotate(int angle) {
    wrapper.addRotate(angle);
    return this;
  }

  public ImageProcessorBuilder addFlip(boolean horizontal, boolean vertical) {
    wrapper.addFlip(horizontal, vertical);
    return this;
  }

  public ImageProcessorBuilder addColorSpaceConvert(BufferFormat dstColorSpace) {
    wrapper.addColorSpaceConvert(dstColorSpace);
    return this;
  }

  public ImageProcessorBuilder addNormalize(float mean, float std) {
    wrapper.addNormalize(mean, std);
    return this;
  }

  public ImageProcessorBuilder addDataTypeConvert() {
    wrapper.addDataTypeConvert();
    return this;
  }

  public ImageProcessor build() {
    return wrapper.build();
  }
}