package org.mrsnu.band;

public enum BufferFormat {
  GRAY_SCALE(0),
  RGB(1),
  RGBA(2),
  YV12(3),
  YV21(4),
  NV21(5),
  NV12(6),
  RAW(7);

  private final int value;
  private static final BufferFormat[] enumValues = BufferFormat.values();

  BufferFormat(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  public static BufferFormat fromValue(int value) {
    return enumValues[value];
  }
}