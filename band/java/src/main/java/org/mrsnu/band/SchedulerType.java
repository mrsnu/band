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

public enum SchedulerType {
  FIXED_DEVICE(0),
  ROUND_ROBIN(1),
  ROUND_ROBIN_IDLE(2),
  SHORTEST_EXPECTED_LATENCY(3),
  FIXED_DEVICE_GLOBAL_QUEUE(4),
  HETEROGENEOUS_EARLIEST_FINISH_TIME(5),
  LEAST_SLACK_TIME_FRIST(6),
  HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED(7),
  THERMAL(8);
  
  private final int value;
  SchedulerType(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }
}
