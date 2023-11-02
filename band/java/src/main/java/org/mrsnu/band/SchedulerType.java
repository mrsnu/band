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
