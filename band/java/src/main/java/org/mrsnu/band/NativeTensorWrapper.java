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

import java.util.List;
import java.nio.ByteBuffer;

public class NativeTensorWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeTensorWrapper(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  @Override
  public void close() {
    deleteTensor(nativeHandle);
  }

  public DataType getType() {
    return DataType.fromValue(getType(nativeHandle));
  }

  public void setType(DataType dataType) {
    setType(nativeHandle, dataType.getValue());
  }

  public ByteBuffer getData() {
    return getData(nativeHandle);
  }

  public void setData(ByteBuffer data) {
    setData(nativeHandle, data);
  }

  public int[] getDims() {
    return getDims(nativeHandle);
  }

  public void setDims(int[] dims) {
    setDims(nativeHandle, dims);
  }

  public int getBytes() {
    return getBytes(nativeHandle);
  }

  public String getName() {
    return getName(nativeHandle);
  }

  public Quantization getQuantization() {
    return getQuantization(nativeHandle);
  }

  public void setQuantization(Quantization quantization) {
    setQuantization(nativeHandle, quantization);
  }

  public long getNativeHandle() {
    return nativeHandle;
  }

  private native void deleteTensor(long tensorHandle);

  private native int getType(long tensorHandle);

  private native void setType(long tensorHandle, int dataType);

  private native ByteBuffer getData(long tensorHandle);

  private native void setData(long tensorHandle, ByteBuffer data);

  private native int[] getDims(long tensorHandle);

  private native void setDims(long tensorHandle, int[] dims);

  private native int getBytes(long tensorHandle);

  private native String getName(long tensorHandle);

  private native Quantization getQuantization(long tensorHandle);

  private native void setQuantization(long tensorHandle, Quantization quantization);
}
