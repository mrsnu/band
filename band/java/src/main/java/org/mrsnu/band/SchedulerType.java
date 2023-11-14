package org.mrsnu.band;

public enum SchedulerType {
  FIXED_DEVICE(0),
  ROUND_ROBIN(1),
  SHORTEST_EXPECTED_LATENCY(2),
  FIXED_DEVICE_GLOBAL_QUEUE(3),
  HETEROGENEOUS_EARLIEST_FINISH_TIME(4),
  LEAST_SLACK_TIME_FRIST(5),
  HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED(6),
  THERMAL(7);
  
  private final int value;
  SchedulerType(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }
}
