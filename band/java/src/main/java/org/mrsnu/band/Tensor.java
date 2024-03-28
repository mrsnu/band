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
import java.util.List;

// Caution: This class has a native resource.
// You may need to use `try-with-resources` statement to
// avoid potential memory leak.
public class Tensor implements AutoCloseable {
  private NativeTensorWrapper wrapper;

  Tensor(long nativeHandle) {
    Band.init();
    wrapper = new NativeTensorWrapper(nativeHandle);
  }

  @Override
  public void close() {
    wrapper = null;
  }

  public DataType getType() {
    return wrapper.getType();
  }

  public void setType(DataType dataType) {
    wrapper.setType(dataType);
  }

  public ByteBuffer getData() {
    return wrapper.getData();
  }

  public void setData(ByteBuffer data) {
    wrapper.setData(data);
  }

  public int[] getDims() {
    return wrapper.getDims();
  }

  public void setDims(int[] dims) {
    wrapper.setDims(dims);
  }

  public int getBytes() {
    return wrapper.getBytes();
  }

  public String getName() {
    return wrapper.getName();
  }

  public Quantization getQuantization() {
    return wrapper.getQuantization();
  }

  public void setQuantization(Quantization quantization) {
    wrapper.setQuantization(quantization);
  }

  private long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}
