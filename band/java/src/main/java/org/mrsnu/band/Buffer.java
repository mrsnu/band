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

import android.media.Image.Plane;

public class Buffer {
  private NativeBufferWrapper wrapper;

  public Buffer(final Tensor tensor) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromTensor(tensor);
  }

  public Buffer(final byte[] buffer, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromByteBuffer(buffer, width, height, bufferFormat);
  }

  public Buffer(final byte[][] yuvBytes, int width, int height, int yRowStride, int uvRowStride,
      int uvPixelStride, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromYUVBuffer(
        yuvBytes, width, height, yRowStride, uvRowStride, uvPixelStride, bufferFormat);
  }

  public Buffer(final Plane[] planes, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromYUVPlane(planes, width, height, bufferFormat);
  }

  public long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}