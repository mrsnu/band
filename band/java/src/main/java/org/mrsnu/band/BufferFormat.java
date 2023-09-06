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