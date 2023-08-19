package org.mrsnu.band;

public class ConfigBuilder {
  private NativeConfigBuilderWrapper wrapper;

  public ConfigBuilder() {
    Band.init();
    this.wrapper = new NativeConfigBuilderWrapper();
  }

  public void addNumWarmups(int numWarmups) {
    wrapper.addNumWarmups(numWarmups);
  }

  public void addNumRuns(int numRuns) {
    wrapper.addNumRuns(numRuns);
  }

  public void addProfilePath(String profilePath) {
    wrapper.addProfilePath(profilePath);
  }

  public void addFrequencyLatencySmoothingFactor(float smoothingFactor) {
    wrapper.addFrequencyLatencySmoothingFactor(smoothingFactor);
  }

  public void addLatencySmoothingFactor(float smoothingFactor) {
    wrapper.addLatencySmoothingFactor(smoothingFactor);
  }

  public void addThermWindowSize(int thermWindowSize) {
    wrapper.addThermWindowSize(thermWindowSize);
  }

  public void addCPUThermIndex(int cpuThermIndex) {
    wrapper.addCPUThermIndex(cpuThermIndex);
  }

  public void addGPUThermIndex(int gpuThermIndex) {
    wrapper.addGPUThermIndex(gpuThermIndex);
  }

  public void addDSPThermIndex(int dspThermIndex) {
    wrapper.addDSPThermIndex(dspThermIndex);
  }

  public void addNPUThermIndex(int npuThermIndex) {
    wrapper.addNPUThermIndex(npuThermIndex);
  }

  public void addCPUFreqPath(String cpuFreqPath) {
    wrapper.addCPUFreqPath(cpuFreqPath);
  }

  public void addGPUFreqPath(String gpuFreqPath) {
    wrapper.addGPUFreqPath(gpuFreqPath);
  }

  public void addDSPFreqPath(String dspFreqPath) {
    wrapper.addDSPFreqPath(dspFreqPath);
  }

  public void addNPUFreqPath(String npuFreqPath) {
    wrapper.addNPUFreqPath(npuFreqPath);
  }

  public void addLatencyLogPath(String latencyLogPath) {
    wrapper.addLatencyLogPath(latencyLogPath);
  }

  public void addThermLogPath(String thermLogPath) {
    wrapper.addThermLogPath(thermLogPath);
  }

  public void addFreqLogPath(String freqLogPath) {
    wrapper.addFreqLogPath(freqLogPath);
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

  public boolean isValid() {
    return wrapper.isValid();
  }

  public Config build() {
    return wrapper.build();
  }
}
