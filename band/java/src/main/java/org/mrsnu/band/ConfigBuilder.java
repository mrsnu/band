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
    wrapper.addNumWarmups(numWarmups);;
  }

  public void addNumRuns(int numRuns) {
    wrapper.addNumRuns(numRuns);
  }

  public void addCopyComputationRatio(int[] copyComputationRatio) {
    wrapper.addCopyComputationRatio(copyComputationRatio);
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

  public void addPlannerCPUMask(CpuMaskFlags cpuMasks) {
    wrapper.addPlannerCPUMask(cpuMasks);
  }

  public void addWorkers(Device[] workers) {
    wrapper.addWorkers(workers);
  }

  public void addWorkerCPUMasks(CpuMaskFlags[] cpuMasks) {
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

  public void addCPUMask(CpuMaskFlags cpuMask) {
    wrapper.addCPUMask(cpuMask);
  }

  public boolean isValid() {
    return wrapper.isValid();
  }

  public Config build() {
    return wrapper.build();
  }
}
