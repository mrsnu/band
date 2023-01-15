package org.mrsnu.band;

import java.util.List;

public class NativeTensorWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeTensorWrapper(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  @Override
  public void close() {
  }

  public DataType getType() {
    return DataType.fromValue(getType(nativeHandle));
  }

  public void setType(DataType dataType) {
    setType(nativeHandle, dataType.getValue());
  }

  public byte[] getData() {
    return getData(nativeHandle);
  }

  public void setData(byte[] data) {
    setData(nativeHandle, data);
  }

  public List<Integer> getDims() {
    return getDims(nativeHandle);
  }

  public void setDims(int[] dims) {
    setDims(nativeHandle, dims);
  }

  public int getBytes() {
    return getBytes(nativeHandle);
  }

  public String getName() {
    return getName(nativeHandle);
  }

  public Quantization getQuantization() {
    return getQuantization(nativeHandle);
  }

  public void setQuantization(Quantization quantization) {
    setQuantization(nativeHandle, quantization);
  }

  public long getNativeHandle() {
    return nativeHandle;
  }

  private native int getType(long tensorHandle);

  private native void setType(long tensorHandle, int dataType);

  private native byte[] getData(long tensorHandle);

  private native void setData(long tensorHandle, byte[] data);

  private native List<Integer> getDims(long tensorHandle);

  private native void setDims(long tensorHandle, int[] dims);

  private native int getBytes(long tensorHandle);

  private native String getName(long tensorHandle);

  private native Quantization getQuantization(long tensorHandle);

  private native void setQuantization(long tensorHandle, Quantization quantization);
}
