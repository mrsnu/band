package org.mrsnu.band;

public enum Device {
  CPU(0),
  GPU(1),
  DSP(2),
  NPU(3);

  private final int value;
  private static final Device[] enumValues = Device.values();

  Device(int value) {
    this.value = value;
  }

  int getValue() {
    return value;
  }

  public static Device fromValue(int value) {
    return enumValues[value];
  }
}
