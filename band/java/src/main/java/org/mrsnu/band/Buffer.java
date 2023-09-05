package org.mrsnu.band;

import android.media.Image.Plane;

public class Buffer {
  private NativeBufferWrapper wrapper;

  public Buffer(final byte[] buffer, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromByteBuffer(buffer, width, height, bufferFormat);
  }

  public Buffer(final byte[][] yuvBytes, int width, int height, int yRowStride, int uvRowStride,
      int uvPixelStride, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromYUVBuffer(
        yuvBytes, width, height, yRowStride, uvRowStride, uvPixelStride, bufferFormat);
  }

  public Buffer(final Plane[] planes, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromYUVPlane(planes, width, height, bufferFormat);
  }

  public long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}