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

public class ImageProcessorBuilder {
  private NativeImageProcessorBuilderWrapper wrapper;

  public ImageProcessorBuilder() {
    Band.init();
    wrapper = new NativeImageProcessorBuilderWrapper();
  }

  public void addCrop(int x0, int y0, int x1, int y1) {
    wrapper.addCrop(x0, y0, x1, y1);
  }

  public void addResize(int width, int height) {
    wrapper.addResize(width, height);
  }

  public void addRotate(int angle) {
    wrapper.addRotate(angle);
  }

  public void addFlip(boolean horizontal, boolean vertical) {
    wrapper.addFlip(horizontal, vertical);
  }

  public void addColorSpaceConvert(BufferFormat dstColorSpace) {
    wrapper.addColorSpaceConvert(dstColorSpace);
  }

  public void addNormalize(float mean, float std) {
    wrapper.addNormalize(mean, std);
  }

  public void addDataTypeConvert() {
    wrapper.addDataTypeConvert();
  }

  public ImageProcessor build() {
    return wrapper.build();
  }
}