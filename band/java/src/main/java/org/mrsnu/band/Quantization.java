/*
 * Copyright 2023 Seoul National University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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