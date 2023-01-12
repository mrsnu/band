package org.mrsnu.band;

import java.util.List;

public class Tensor {
  private NativeTensorWrapper wrapper;

  Tensor(long nativeHandle) {
    Band.init();
    wrapper = new NativeTensorWrapper(nativeHandle);
  }
  
  public DataType getType() {
    return wrapper.getType();
  }

  public void setType(DataType dataType) {
    wrapper.setType(dataType);
  }

  public byte[] getData() {
    return wrapper.getData();
  }

  public List<Integer> getDims() {
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
}
