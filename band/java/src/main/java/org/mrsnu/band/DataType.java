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
