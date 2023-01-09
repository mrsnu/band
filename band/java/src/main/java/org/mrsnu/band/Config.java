package org.mrsnu.band;

public class Config implements AutoCloseable {
  private long nativeHandle = 0;

  private Config(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  @Override
  public void close() {
    deleteConfig(nativeHandle);
  }

  private native void deleteConfig(long configHandle);
}
