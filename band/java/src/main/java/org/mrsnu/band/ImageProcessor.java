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

public class ImageProcessor implements AutoCloseable {
  private long nativeHandle = 0;

  private ImageProcessor(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  public void process(Buffer srcBuffer, Tensor dstTensor) {
    process(nativeHandle, srcBuffer, dstTensor);
  }

  private long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public void close() {
    deleteImageProcessor(nativeHandle);
  }

  private native void process(
    long imageProcessorHandle, Object srcBufferHandle, Object dstTensorHandle);

  private native void deleteImageProcessor(long imageProcessorHandle);
}