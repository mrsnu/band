package org.mrsnu.band;

import org.mrsnu.band.BandInterface;

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
