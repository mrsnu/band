package org.mrsnu.band;

import java.util.Set;
import java.util.List;

public class NativeModelWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeModelWrapper() {
    nativeHandle = createModel();
  }

  @Override
  public void close() {
    deleteModel(nativeHandle);
    nativeHandle = 0;
  }

  public long getNativeHandle() {
    return nativeHandle;
  }

  public void loadFromFile(BackendType backendType, String filePath) {
    loadFromFile(nativeHandle, backendType.getValue(), filePath);
  }

  public List<BackendType> getSupportedBackends() throws IllegalStateException {
    if (nativeHandle == 0) {
      // TODO(widiba03304): define proper exceptions for java frontend.
      throw new IllegalStateException("This `Model` object has been closed.");
    }
    return getSupportedBackends(nativeHandle);
  }

  private static native long createModel();

  private static native void deleteModel(long modelHandle);

  private static native void loadFromFile(long modelHandle, int backendType, String filePath);

  private static native List<BackendType> getSupportedBackends(long modelHandle);
}
