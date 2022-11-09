package org.mrsnu.band;

public enum CpuMaskFlags {
  ALL(0),
  LITTLE(1),
  BIG(2),
  PRIMARY(3);

  private final int value;
  private static final CpuMaskFlags[] enumValues = CpuMaskFlags.values();;
  
  CpuMaskFlags(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  public static CpuMaskFlags fromValue(int value) {
    return enumValues[value];
  }
}
