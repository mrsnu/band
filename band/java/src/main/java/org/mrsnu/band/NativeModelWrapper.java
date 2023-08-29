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
import java.util.ArrayList;
import java.util.List;

public class NativeModelWrapper implements AutoCloseable {
  private long nativeHandle = 0;
  private ByteBuffer modelBuffer;

  NativeModelWrapper() {
    nativeHandle = createModel();
  }

  @Override
  public void close() {
    deleteModel(nativeHandle);
    nativeHandle = 0;
  }

  public void loadFromFile(BackendType backendType, String filePath) {
    loadFromFile(nativeHandle, backendType.getValue(), filePath);
  }

  public void loadFromBuffer(BackendType backendType, ByteBuffer modelBuffer) {
    this.modelBuffer = modelBuffer;
    loadFromBuffer(nativeHandle, backendType.getValue(), this.modelBuffer);
  }

  public List<BackendType> getSupportedBackends() throws IllegalStateException {
    List<BackendType> ret = new ArrayList<>();
    int[] nativeSupportedBackends = getSupportedBackends(nativeHandle);
    for (int i = 0; i < nativeSupportedBackends.length; i++) {
      ret.add(BackendType.fromValue(nativeSupportedBackends[i]));
    }
    return ret;
  }

  public long getNativeHandle() {
    return nativeHandle;
  }

  private static native long createModel();

  private static native void deleteModel(long modelHandle);

  private static native void loadFromFile(long modelHandle, int backendType, String filePath);

  private static native void loadFromBuffer(long modelHandle, int backendType, ByteBuffer modelBuffer);

  private static native int[] getSupportedBackends(long modelHandle);
}
