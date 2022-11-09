package org.mrsnu.band;

public interface Quantization {
  public enum QuantizationType {
    NO_QUANTIZATION(0),
    AFFINE_QUANTIZATION(1);

    private final int value;

    QuantizationType(int value) {
      this.value = value;
    }

    int getValue() {
      return value;
    }
  }
}