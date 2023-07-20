package org.mrsnu.band;

public enum CpuMaskFlag {
  ALL(0),
  LITTLE(1),
  BIG(2),
  PRIMARY(3);

  private final int value;
  private static final CpuMaskFlag[] enumValues = CpuMaskFlag.values();

  CpuMaskFlag(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  public static CpuMaskFlag fromValue(int value) {
    return enumValues[value];
  }
}
