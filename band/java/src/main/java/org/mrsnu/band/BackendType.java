package org.mrsnu.band;

public enum BackendType {
  TFLITE(0);

  private final int value;

  BackendType(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  public static BackendType fromValue(int value) {
    if (value == 0) {
      return TFLITE;
    } else {
      return null;
    }
  }
}
