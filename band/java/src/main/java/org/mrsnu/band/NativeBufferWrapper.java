package org.mrsnu.band;

public class NativeBufferWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeBufferWrapper() {
    Band.init();
  }

  public long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public void close() {
    deleteBuffer(nativeHandle);
  }

  public void setFromByteBuffer(
      final byte[] buffer, int width, int height, BufferFormat bufferFormat) {
    this.nativeHandle = createFromByteBuffer(buffer, width, height, bufferFormat.getValue());
  }

  public void setFromYUVBuffer(final byte[][] yuvBytes, int width, int height, int yRowStride,
      int uvRowStride, int uvPixelStride, BufferFormat bufferFormat) {
    this.nativeHandle = createFromYUVBuffer(yuvBytes[0], yuvBytes[1], yuvBytes[2], width, height,
        yRowStride, uvRowStride, uvPixelStride, bufferFormat.getValue());
  }

  private static native void deleteBuffer(long bufferHandle);

  private static native long createFromByteBuffer(
      final byte[] buffer, int width, int height, int bufferFormat);

  private static native long createFromYUVBuffer(final byte[] y, final byte[] u, final byte[] v,
      int width, int height, int yRowStride, int uvRowStride, int uvPixelStride, int bufferFormat);
}