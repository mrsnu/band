package org.mrsnu.band;

public enum WorkerType {
  DEVICE_QUEUE(0),
  GLOBAL_QUEUE(1);
  
  private final int value;
  WorkerType(int value) {
    this.value = value;
  }
  
  int getValue() {
    return value;
  }
}
