package org.mrsnu.band;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

class NativeModelWrapper implements AutoCloseable {
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
