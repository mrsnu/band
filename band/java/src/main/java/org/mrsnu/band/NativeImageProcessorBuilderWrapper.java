package org.mrsnu.band;

import java.nio.ByteBuffer;

public class NativeImageProcessorBuilderWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeImageProcessorBuilderWrapper() {
    nativeHandle = createImageProcessorBuilder();
  }

  public void addCrop(int x, int y, int width, int height) {
    addCrop(nativeHandle, x, y, width, height);
  }

  public void addResize(int width, int height) {
    addResize(nativeHandle, width, height);
  }

  public void addRotate(int angle) {
    addRotate(nativeHandle, angle);
  }

  public void addFlip(boolean horizontal, boolean vertical) {
    addFlip(nativeHandle, horizontal, vertical);
  }

  public void addConvertColor(BufferFormat dstColorSpace) {
    addConvertColor(nativeHandle, dstColorSpace.getValue());
  }

  public void addNormalize(float mean, float std) {
    addNormalize(nativeHandle, mean, std);
  }

  public void addDataTypeConvert() {
    addDataTypeConvert(nativeHandle);
  }

  public ImageProcessor build() {
    return (ImageProcessor) build(nativeHandle);
  }

  @Override
  public void close() {
    deleteImageProcessorBuilder(nativeHandle);
    nativeHandle = 0;
  }

  private native long createImageProcessorBuilder();

  private native void deleteImageProcessorBuilder(long imageProcessorBuilderHandle);

  private native void addCrop(
      long imageProcessorBuilderHandle, int x, int y, int width, int height);

  private native void addResize(long imageProcessorBuilderHandle, int width, int height);

  private native void addRotate(long imageProcessorBuilderHandle, int angle);

  private native void addFlip(
      long imageProcessorBuilderHandle, boolean horizontal, boolean vertical);

  private native void addConvertColor(long imageProcessorBuilderHandle, int dstColorSpace);

  private native void addNormalize(long imageProcessorBuilderHandle, float mean, float std);

  private native void addDataTypeConvert(long imageProcessorBuilderHandle);

  private native Object build(long imageProcessorBuilderHandle);
}