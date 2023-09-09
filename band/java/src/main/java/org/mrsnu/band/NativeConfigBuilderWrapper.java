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

class NativeConfigBuilderWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeConfigBuilderWrapper() {
    nativeHandle = createConfigBuilder();
  }

  public void addOnline(boolean online) {
    addOnline(nativeHandle, online);
  }

  public void addNumWarmups(int numWarmups) {
    addNumWarmups(nativeHandle, numWarmups);
  }

  public void addNumRuns(int numRuns) {
    addNumRuns(nativeHandle, numRuns);
  }

  public void addProfileDataPath(String profileDataPath) {
    addProfileDataPath(nativeHandle, profileDataPath);
  }

  public void addSmoothingFactor(float smoothingFactor) {
    addSmoothingFactor(nativeHandle, smoothingFactor);
  }

  public void addPlannerLogPath(String plannerLogPath) {
    addPlannerLogPath(nativeHandle, plannerLogPath);
  }

  public void addScheduleWindowSize(int scheduleWindowSize) {
    addScheduleWindowSize(nativeHandle, scheduleWindowSize);
  }

  public void addSchedulers(SchedulerType[] schedulers) {
    int[] tmpArray = new int[schedulers.length];
    for (int i = 0; i < schedulers.length; i++) {
      tmpArray[i] = schedulers[i].getValue();
    }
    addSchedulers(nativeHandle, tmpArray);
  }

  public void addPlannerCPUMask(CpuMaskFlag cpuMask) {
    addPlannerCPUMask(nativeHandle, cpuMask.getValue());
  }

  public void addWorkers(Device[] workers) {
    int[] tmpArray = new int[workers.length];
    for (int i = 0; i < workers.length; i++) {
      tmpArray[i] = workers[i].getValue();
    }
    addWorkers(nativeHandle, tmpArray);
  }

  public void addWorkerCPUMasks(CpuMaskFlag[] cpuMasks) {
    int[] tmpArray = new int[cpuMasks.length];
    for (int i = 0; i < cpuMasks.length; i++) {
      tmpArray[i] = cpuMasks[i].getValue();
    }
    addWorkerCPUMasks(nativeHandle, tmpArray);
  }

  public void addWorkerNumThreads(int[] numThreads) {
    addWorkerNumThreads(nativeHandle, numThreads);
  }

  public void addAllowWorkSteal(boolean allowWorkSteal) {
    addAllowWorkSteal(nativeHandle, allowWorkSteal);
  }

  public void addAvailabilityCheckIntervalMs(int availabilityCheckIntervalMs) {
    addAvailabilityCheckIntervalMs(nativeHandle, availabilityCheckIntervalMs);
  }

  public void addMinimumSubgraphSize(int minimumSubgraphSize) {
    addMinimumSubgraphSize(nativeHandle, minimumSubgraphSize);
  }

  public void addSubgraphPreparationType(SubgraphPreparationType subgraphPreparationType) {
    addSubgraphPreparationType(nativeHandle, subgraphPreparationType.getValue());
  }

  public void addCPUMask(CpuMaskFlag cpuMask) {
    addCPUMask(nativeHandle, cpuMask.getValue());
  }

  public void addResourceMonitorDeviceFreqPath(Device flag, String devicePath) {
    addResourceMonitorDeviceFreqPath(nativeHandle, flag.getValue(), devicePath);
  }

  public void addResourceMonitorIntervalMs(int intervalMs) {
    addResourceMonitorIntervalMs(nativeHandle, intervalMs);
  }

  public void addResourceMonitorLogPath(String logPath) {
    addResourceMonitorLogPath(nativeHandle, logPath);
  }

  public Config build() {
    // TODO(widiba03304): Config should be built only by the ConfigBuilder.
    // By declaring the Config's private ctor and violating the access control in
    // JNI, it cannot be built with JAVA API.
    return (Config) build(nativeHandle);
  }

  private long getNativeHandle() {
    return nativeHandle;
  }

  @Override
  public void close() {
    deleteConfigBuilder(nativeHandle);
    nativeHandle = 0;
  }

  private native long createConfigBuilder();

  private native void deleteConfigBuilder(long configBuilderHandle);

  private native void addOnline(long configBuilderHandle, boolean online);

  private native void addNumWarmups(long configBuilderHandle, int numWarmups);

  private native void addNumRuns(long configBuilderHandle, int numRuns);

  private native void addProfileDataPath(long configBuilderHandle, String profileDataPath);

  private native void addSmoothingFactor(long configBuilderHandle, float smoothingFactor);

  private native void addPlannerLogPath(long configBuilderHandle, String plannerLogPath);

  private native void addScheduleWindowSize(long configBuilderHandle, int scheduleWindowSize);

  private native void addSchedulers(long configBuilderHandle, int[] scheduers);

  private native void addPlannerCPUMask(long configBuilderHandle, int cpuMasks);

  private native void addWorkers(long configBuilderHandle, int[] workers);

  private native void addWorkerCPUMasks(long configBuilderHandle, int[] cpuMasks);

  private native void addWorkerNumThreads(long configBuilderHandle, int[] numThreads);

  private native void addAllowWorkSteal(long configBuilderHandle, boolean allowWorkSteal);

  private native void addAvailabilityCheckIntervalMs(
      long configBuilderHandle, int availabilityCheckIntervalMs);

  private native void addMinimumSubgraphSize(long configBuilderHandle, int minimumSubgraphSize);

  private native void addSubgraphPreparationType(
      long configBuilderHandle, int subgraphPreparationType);

  private native void addCPUMask(long configBuilderHandle, int cpuMask);

  private native void addResourceMonitorDeviceFreqPath(
      long configBuilderHandle, int deviceFlag, String devicePath);

  private native void addResourceMonitorIntervalMs(long configBuilderHandle, int intervalMs);

  private native void addResourceMonitorLogPath(long configBuilderHandle, String logPath);

  private native Object build(long configBuilderHandle);
}
