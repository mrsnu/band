package org.mrsnu.band;

import android.media.Image.Plane;
import java.nio.ByteBuffer;

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

  public void setFromYUVPlane(final Plane[] planes, int width, int height, BufferFormat bufferFormat) {
    this.nativeHandle = createFromYUVPlanes(planes[0].getBuffer(), planes[1].getBuffer(),
        planes[2].getBuffer(), width, height, planes[0].getRowStride(), planes[1].getRowStride(), planes[1].getPixelStride(), bufferFormat.getValue());
  }

  private static native void deleteBuffer(long bufferHandle);

  private static native long createFromByteBuffer(
      final byte[] buffer, int width, int height, int bufferFormat);

  private static native long createFromYUVBuffer(byte[] y, byte[] u, byte[] v,
      int width, int height, int yRowStride, int uvRowStride, int uvPixelStride, int bufferFormat);

  private static native long createFromYUVPlanes(ByteBuffer y, ByteBuffer u,
      ByteBuffer v, int width, int height, int yRowStride, int uvRowStride, int uvPixelStride, int bufferFormat);
}