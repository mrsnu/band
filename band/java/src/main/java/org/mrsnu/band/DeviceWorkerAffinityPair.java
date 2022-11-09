package org.mrsnu.band;

// TODO(widiba03304): This device-worker pair is not natural 
// as `worker` should be hidden from the users. Find a way to \
// hide such information.

public class DeviceWorkerAffinityPair {
  private final Device device;
  private final int worker;

  DeviceWorkerAffinityPair(Device device, int worker) {
    this.device = device;
    this.worker = worker;
  }

  Device getDevice() {
    return device;
  }

  int getWorker() {
    return worker;
  }
}
