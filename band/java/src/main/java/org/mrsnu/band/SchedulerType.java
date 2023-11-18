package org.mrsnu.band;

public enum SchedulerType {
  FIXED_DEVICE(0),
  ROUND_ROBIN(1),
  FIXED_DEVICE_GLOBAL_QUEUE(2),
  HETEROGENEOUS_EARLIEST_FINISH_TIME(3),
  LEAST_SLACK_TIME_FRIST(4),
  HETEROGENEOUS_EARLIEST_FINISH_TIME_RESERVED(5),
  THERMAL(6);
  
  private final int value;
  SchedulerType(int value) {
    this.value = value;
  }

  public int getValue() {
    return value;
  }
}
