package org.mrsnu.band;

public class NativeConfigWrapper extends NativeWrapper implements AutoCloseable {
  
  @Override
  public void close() {
    deleteConfig(nativeHandle);
    nativeHandle = 0;
  }

  private native void deleteConfig(long configHandle);
}
