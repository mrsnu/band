package org.mrsnu.band;

import java.util.List;
import java.nio.ByteBuffer;

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

  public ByteBuffer getData() {
    return wrapper.getData();
  }

  public void setData(ByteBuffer data) {
    wrapper.setData(data);
  }

  public int[] getDims() {
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

  private long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}
