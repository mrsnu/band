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
import java.nio.ByteBuffer;

// Caution: This class has a native resource.
// You may need to use `try-with-resources` statement to
// avoid potential memory leak.
public class Buffer implements AutoCloseable {
  private NativeBufferWrapper wrapper;

  public Buffer(final Tensor tensor) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromTensor(tensor);
  }

  @Override
  public void close() {
    wrapper = null;
  }

  public Buffer(final ByteBuffer buffer, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromByteBuffer(buffer, width, height, bufferFormat);
  }

  public Buffer(final Plane[] planes, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromYUVPlane(planes, width, height, bufferFormat);
  }

  private long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}