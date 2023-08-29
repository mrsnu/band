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

  public void addOnline(boolean online) {
    wrapper.addOnline(online);
  }

  public void addNumWarmups(int numWarmups) {
    wrapper.addNumWarmups(numWarmups);
    ;
  }

  public void addNumRuns(int numRuns) {
    wrapper.addNumRuns(numRuns);
  }

  public void addProfileDataPath(String profileDataPath) {
    wrapper.addProfileDataPath(profileDataPath);
  }

  public void addSmoothingFactor(float smoothingFactor) {
    wrapper.addSmoothingFactor(smoothingFactor);
  }

  public void addPlannerLogPath(String plannerLogPath) {
    wrapper.addPlannerLogPath(plannerLogPath);
  }

  public void addScheduleWindowSize(int scheduleWindowSize) {
    wrapper.addScheduleWindowSize(scheduleWindowSize);
  }

  public void addSchedulers(SchedulerType[] scheduers) {
    wrapper.addSchedulers(scheduers);
  }

  public void addPlannerCPUMask(CpuMaskFlag cpuMasks) {
    wrapper.addPlannerCPUMask(cpuMasks);
  }

  public void addWorkers(Device[] workers) {
    wrapper.addWorkers(workers);
  }

  public void addWorkerCPUMasks(CpuMaskFlag[] cpuMasks) {
    wrapper.addWorkerCPUMasks(cpuMasks);
  }

  public void addWorkerNumThreads(int[] numThreads) {
    wrapper.addWorkerNumThreads(numThreads);
  }

  public void addAllowWorkSteal(boolean allowWorkSteal) {
    wrapper.addAllowWorkSteal(allowWorkSteal);
  }

  public void addAvailabilityCheckIntervalMs(int availabilityCheckIntervalMs) {
    wrapper.addAvailabilityCheckIntervalMs(availabilityCheckIntervalMs);
  }

  public void addMinimumSubgraphSize(int minimumSubgraphSize) {
    wrapper.addMinimumSubgraphSize(minimumSubgraphSize);
  }

  public void addSubgraphPreparationType(SubgraphPreparationType subgraphPreparationType) {
    wrapper.addSubgraphPreparationType(subgraphPreparationType);
  }

  public void addCPUMask(CpuMaskFlag cpuMask) {
    wrapper.addCPUMask(cpuMask);
  }

  public void addResourceMonitorDeviceFreqPath(Device deviceFlag, String devicePath) {
    wrapper.addResourceMonitorDeviceFreqPath(deviceFlag, devicePath);
  }

  public void addResourceMonitorIntervalMs(int resourceMonitorIntervalMs) {
    wrapper.addResourceMonitorIntervalMs(resourceMonitorIntervalMs);
  }

  public void addResourceMonitorLogPath(String resourceMonitorLogPath) {
    wrapper.addResourceMonitorLogPath(resourceMonitorLogPath);
  }

  public Config build() {
    return wrapper.build();
  }
}
