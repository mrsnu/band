package org.mrsnu.band;

import android.media.Image.Plane;

public class Buffer {
  private NativeBufferWrapper wrapper;

  Buffer(final byte[] buffer, int width, int height, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromByteBuffer(buffer, width, height, bufferFormat);
  }

  Buffer(final byte[][] yuvBytes, int width, int height, int yRowStride, int uvRowStride,
      int uvPixelStride, BufferFormat bufferFormat) {
    wrapper = new NativeBufferWrapper();
    wrapper.setFromYUVBuffer(
        yuvBytes, width, height, yRowStride, uvRowStride, uvPixelStride, bufferFormat);
  }

  Buffer(final Plane[] planes) {
    wrapper = new NativeBufferWrapper();
    // TODO: implement this
  }

  private long getNativeHandle() {
    return wrapper.getNativeHandle();
  }
}