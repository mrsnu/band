package org.mrsnu.band;

import java.util.List;

public class NativeTensorWrapper extends NativeWrapper implements AutoCloseable {
  private long nativeHandle;

  @Override
  public void close() {

  }

  public long getNativeHandle() {
    return nativeHandle;
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
    return null;
  }

  public void setQuantization(Quantization quantization) {
  }

  private native int getType(long tensorHandle);

  private native void setType(long tensorHandle, int dataType);

  private native byte[] getData(long tensorHandle);

  private native List<Integer> getDims(long tensorHandle);

  private native void setDims(long tensorHandle, int[] dims);

  private native int getBytes(long tensorHandle);

  private native String getName(long tensorHandle);

  private native int getQuantization(long tensorHandle);

  private native void setQuantization(long tensorHandle, long quantizationHandle);
}
