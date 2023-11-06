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

public class ConfigBuilder {
  private NativeConfigBuilderWrapper wrapper;

  public ConfigBuilder() {
    Band.init();
    this.wrapper = new NativeConfigBuilderWrapper();
  }

  public ConfigBuilder addOnline(boolean online) {
    wrapper.addOnline(online);
    return this;
  }

  public ConfigBuilder addNumWarmups(int numWarmups) {
    wrapper.addNumWarmups(numWarmups);
    return this;
  }

  public ConfigBuilder addNumRuns(int numRuns) {
    wrapper.addNumRuns(numRuns);
    return this;
  }

  public ConfigBuilder addProfileDataPath(String profileDataPath) {
    wrapper.addProfileDataPath(profileDataPath);
    return this;
  }

  public ConfigBuilder addSmoothingFactor(float smoothingFactor) {
    wrapper.addSmoothingFactor(smoothingFactor);
    return this;
  }

  public ConfigBuilder addPlannerLogPath(String plannerLogPath) {
    wrapper.addPlannerLogPath(plannerLogPath);
    return this;
  }

  public ConfigBuilder addScheduleWindowSize(int scheduleWindowSize) {
    wrapper.addScheduleWindowSize(scheduleWindowSize);
    return this;
  }

  public ConfigBuilder addSchedulers(SchedulerType[] scheduers) {
    wrapper.addSchedulers(scheduers);
    return this;
  }

  public ConfigBuilder addPlannerCPUMask(CpuMaskFlag cpuMasks) {
    wrapper.addPlannerCPUMask(cpuMasks);
    return this;
  }

  public ConfigBuilder addWorkers(Device[] workers) {
    wrapper.addWorkers(workers);
    return this;
  }

  public ConfigBuilder addWorkerCPUMasks(CpuMaskFlag[] cpuMasks) {
    wrapper.addWorkerCPUMasks(cpuMasks);
    return this;
  }

  public ConfigBuilder addWorkerNumThreads(int[] numThreads) {
    wrapper.addWorkerNumThreads(numThreads);
    return this;
  }

  public ConfigBuilder addAllowWorkSteal(boolean allowWorkSteal) {
    wrapper.addAllowWorkSteal(allowWorkSteal);
    return this;
  }

  public ConfigBuilder addAvailabilityCheckIntervalMs(int availabilityCheckIntervalMs) {
    wrapper.addAvailabilityCheckIntervalMs(availabilityCheckIntervalMs);
    return this;
  }

  public ConfigBuilder addMinimumSubgraphSize(int minimumSubgraphSize) {
    wrapper.addMinimumSubgraphSize(minimumSubgraphSize);
    return this;
  }

  public ConfigBuilder addSubgraphPreparationType(SubgraphPreparationType subgraphPreparationType) {
    wrapper.addSubgraphPreparationType(subgraphPreparationType);
    return this;
  }

  public ConfigBuilder addCPUMask(CpuMaskFlag cpuMask) {
    wrapper.addCPUMask(cpuMask);
    return this;
  }

  public ConfigBuilder addResourceMonitorDeviceFreqPath(Device deviceFlag, String devicePath) {
    wrapper.addResourceMonitorDeviceFreqPath(deviceFlag, devicePath);
    return this;
  }

  public ConfigBuilder addResourceMonitorIntervalMs(int resourceMonitorIntervalMs) {
    wrapper.addResourceMonitorIntervalMs(resourceMonitorIntervalMs);
    return this;
  }

  public ConfigBuilder addResourceMonitorLogPath(String resourceMonitorLogPath) {
    wrapper.addResourceMonitorLogPath(resourceMonitorLogPath);
    return this;
  }

  public Config build() {
    return wrapper.build();
  }
}
