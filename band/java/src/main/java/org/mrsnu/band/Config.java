package org.mrsnu.band;

public class Config implements BandInterface, AutoCloseable {
  private NativeConfigWrapper wrapper;

  Config() {
    wrapper = new NativeConfigWrapper();
  }

  @Override
  public void close() {
    
  }

  @Override
  public NativeConfigWrapper getNativeWrapper(NativeWrapper.NativeKey key) {
    return (NativeConfigWrapper) wrapper;
  }
}
