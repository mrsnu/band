package org.mrsnu.band;

import java.util.List;

public class Model implements BandInterface, AutoCloseable {
  private NativeModelWrapper wrapper;
  
  Model(BackendType backendType, String filePath) {
    wrapper.loadFromFile(backendType, filePath);
  }

  @Override
  public void close() {
    
  }

  public List<BackendType> getSupportedBackends() {
    return ((NativeModelWrapper) wrapper).getSupportedBackends();
  }

  public NativeModelWrapper getNativeWrapper(NativeWrapper.NativeKey key) {
    return (NativeModelWrapper) wrapper;
  }
}
