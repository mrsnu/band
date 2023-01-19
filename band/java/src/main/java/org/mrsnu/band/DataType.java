package org.mrsnu.band;

public enum DataType {
  NO_TYPE(0),
  FLOAT32(1),
  INT32(2),
  UINT8(3),
  INT64(4),
  STRING(5),
  BOOL(6),
  INT16(7),
  COMPLEX64(8),
  INT8(9),
  FLOAT16(10),
  FLOAT64(11);

  private final int value;
  private static final DataType[] enumValues = DataType.values();

  DataType(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }

  public static DataType fromValue(int value) {
    return enumValues[value];
  }
}
