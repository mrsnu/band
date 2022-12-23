package org.mrsnu.band;

import org.mrsnu.band.BandInterface;
import java.util.List;

public class Tensor implements BandInterface, AutoCloseable {
  private NativeTensorWrapper wrapper;

  Tensor() {
    
  }

  @Override
  public void close() {
  }
  
  public DataType getType() {
    return null;
  }

  public void setType(DataType dataType) {

  }

  public byte[] getData() {
    return null;
  }

  public List<Integer> getDims() {
    return null;
  }

  public void setDims(int[] dims) {

  }

  public int getBytes() {
    return 0;
  }

  public String getName() {
    return null;
  }

  public Quantization getQuantization() {
    return null;
  }

  public void setQuantization(Quantization quantization) {

  }

  public NativeTensorWrapper getNativeWrapper(NativeWrapper.NativeKey key) {
    return wrapper;
  }
}
