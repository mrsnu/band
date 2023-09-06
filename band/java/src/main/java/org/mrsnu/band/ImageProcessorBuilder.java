package org.mrsnu.band;

public class ImageProcessorBuilder {
  private NativeImageProcessorBuilderWrapper wrapper;

  public ImageProcessorBuilder() {
    Band.init();
    wrapper = new NativeImageProcessorBuilderWrapper();
  }

  public void addCrop(int x0, int y0, int x1, int y1) {
    wrapper.addCrop(x0, y0, x1, y1);
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

  public void addColorSpaceConvert(BufferFormat dstColorSpace) {
    wrapper.addColorSpaceConvert(dstColorSpace);
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