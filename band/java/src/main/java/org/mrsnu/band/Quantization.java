package org.mrsnu.band;

public class Quantization {
  public enum QuantizationType {
    NO_QUANTIZATION(0),
    AFFINE_QUANTIZATION(1);

    private final int value;

    QuantizationType(int value) {
      this.value = value;
    }

    public int getValue() {
      return value;
    }
  }

  private QuantizationType quantizationType;
  private long paramHandle;

  Quantization(QuantizationType quantizatinoType, long paramHandle) {
    this.quantizationType = quantizatinoType;
    this.paramHandle = paramHandle;
  }

  public QuantizationType getQuantizationType() {
    return quantizationType;
  }

  public long getParamHandle() {
    return paramHandle;
  }
}