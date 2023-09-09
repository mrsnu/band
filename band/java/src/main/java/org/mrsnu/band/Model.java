package org.mrsnu.band;

import java.nio.ByteBuffer;
import java.util.List;

public class Model {
  private NativeModelWrapper wrapper;
  
  public Model(BackendType backendType, String filePath) {
    Band.init();
    wrapper = new NativeModelWrapper();
    wrapper.loadFromFile(backendType, filePath);
  }
  
  public Model(BackendType backendType, ByteBuffer modelBuffer) {
    Band.init();
    wrapper = new NativeModelWrapper();
    wrapper.loadFromBuffer(backendType, modelBuffer);
  }

  public List<BackendType> getSupportedBackends() {
    return wrapper.getSupportedBackends();
  }

  private long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}
