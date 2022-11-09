package org.mrsnu.band;

public class QuantizationImpl implements Quantization {
  private long nativeHandle;
  
  public QuantizationImpl(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }
  
  public long getNativeHandle() {
    return nativeHandle;
  }
}
