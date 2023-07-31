package org.mrsnu.band;

public class ImageProcessor implements AutoCloseable {
  private long nativeHandle = 0;

  private ImageProcessor(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  public void process(Buffer srcBuffer, Tensor dstTensor) {
    process(nativeHandle, srcBuffer, dstTensor);
  }

  private long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public void close() {
    deleteImageProcessor(nativeHandle);
  }

  private native void process(
      Object imageProcessorHandle, Object srcBufferHandle, Object dstTensorHandle);

  private native void deleteImageProcessor(long imageProcessorHandle);
}