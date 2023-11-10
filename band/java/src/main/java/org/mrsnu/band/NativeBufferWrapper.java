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
import android.util.Log;
import java.nio.ByteBuffer;

class NativeBufferWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeBufferWrapper() {
    Band.init();
  }

  public long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public void close() {
    deleteBuffer(nativeHandle);
  }

  public void setFromTensor(final Tensor tensor) {
    this.nativeHandle = createFromTensor(tensor);
  }

  public void setFromByteBuffer(
      final ByteBuffer buffer, int width, int height, BufferFormat bufferFormat) {
    this.nativeHandle = createFromByteBuffer(buffer, width, height, bufferFormat.getValue());
  }

  public void setFromYUVPlane(
      final Plane[] planes, int width, int height, BufferFormat bufferFormat) {
    this.nativeHandle = createFromYUVPlanes(planes[0].getBuffer(), planes[1].getBuffer(),
        planes[2].getBuffer(), width, height, planes[0].getRowStride(), planes[1].getRowStride(),
        planes[1].getPixelStride(), bufferFormat.getValue());
  }

  private static native void deleteBuffer(long bufferHandle);

  private static native long createFromTensor(Object tensorObject);

  private static native long createFromByteBuffer(
      ByteBuffer buffer, int width, int height, int bufferFormat);

  private static native long createFromYUVPlanes(ByteBuffer y, ByteBuffer u, ByteBuffer v,
      int width, int height, int yRowStride, int uvRowStride, int uvPixelStride, int bufferFormat);
}