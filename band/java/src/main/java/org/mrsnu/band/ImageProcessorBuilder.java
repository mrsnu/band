package org.mrsnu.band;

public class ImageProcessorBuilder {
  private NativeImageProcessorBuilderWrapper wrapper;

  public ImageProcessorBuilder() {
    Band.init();
    wrapper = new NativeImageProcessorBuilderWrapper();
  }

  public void addCrop(int x, int y, int width, int height) {
    wrapper.addCrop(x, y, width, height);
  }

  public void addResize(int width, int height) {
    wrapper.addResize(width, height);
  }

  public void addRotate(int angle) {
    wrapper.addRotate(angle);
  }

  public void addFlip(boolean horizontal, boolean vertical) {
    wrapper.addFlip(horizontal, vertical);
  }

  public void addConvertColor(BufferFormat dstColorSpace) {
    wrapper.addConvertColor(dstColorSpace);
  }

  public void addNormalize(float mean, float std) {
    wrapper.addNormalize(mean, std);
  }

  public void addDataTypeConvert() {
    wrapper.addDataTypeConvert();
  }

  public ImageProcessor build() {
    return wrapper.build();
  }
}