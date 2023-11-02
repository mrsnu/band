package org.mrsnu.band;

class NativeConfigBuilderWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  NativeConfigBuilderWrapper() {
    nativeHandle = createConfigBuilder();
  }

  public void addNumWarmups(int numWarmups) {
    addNumWarmups(nativeHandle, numWarmups);
  }

  public void addNumRuns(int numRuns) {
    addNumRuns(nativeHandle, numRuns);
  }

  public void addProfilePath(String profilePath) {
    addProfilePath(nativeHandle, profilePath);
  }

  public void addFrequencyLatencySmoothingFactor(float smoothingFactor) {
    addFrequencyLatencySmoothingFactor(nativeHandle, smoothingFactor);
  }

  public void addLatencySmoothingFactor(float smoothingFactor) {
    addLatencySmoothingFactor(nativeHandle, smoothingFactor);
  }

  public void addThermWindowSize(int thermWindowSize) {
    addThermWindowSize(nativeHandle, thermWindowSize);
  }

  public void addCPUThermIndex(int cpuThermIndex) {
    addCPUThermIndex(nativeHandle, cpuThermIndex);
  }
  
  public void addGPUThermIndex(int gpuThermIndex) {
    addGPUThermIndex(nativeHandle, gpuThermIndex);
  }

  public void addDSPThermIndex(int dspThermIndex) {
    addDSPThermIndex(nativeHandle, dspThermIndex);
  }

  public void addNPUThermIndex(int npuThermIndex) {
    addNPUThermIndex(nativeHandle, npuThermIndex);
  }

  public void addTargetThermIndex(int targetThermIndex) {
    addTargetThermIndex(nativeHandle, targetThermIndex);
  }

  public void addCPUFreqPath(String cpuFreqPath) {
    addCPUFreqPath(nativeHandle, cpuFreqPath);
  }

  public void addGPUFreqPath(String gpuFreqPath) {
    addGPUFreqPath(nativeHandle, gpuFreqPath);
  }

  public void addDSPFreqPath(String dspFreqPath) {
    addDSPFreqPath(nativeHandle, dspFreqPath);
  }

  public void addNPUFreqPath(String npuFreqPath) {
    addNPUFreqPath(nativeHandle, npuFreqPath);
  }

  public void addLatencyLogPath(String latencyLogPath) {
    addLatencyLogPath(nativeHandle, latencyLogPath);
  }

  public void addThermLogPath(String thermLogPath) {
    addThermLogPath(nativeHandle, thermLogPath);
  }
  
  public void addFreqLogPath(String freqLogPath) {
    addFreqLogPath(nativeHandle, freqLogPath);
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

  public Config build() {
    // TODO(widiba03304): Config should be built only by the ConfigBuilder.
    // By declaring the Config's private ctor and violating the access control in
    // JNI, it cannot be built with JAVA API.
    return (Config) build(nativeHandle);
  }

  public boolean isValid() {
    return isValid(nativeHandle);
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

  private native void addNumWarmups(long configBuilderHandle, int numWarmups);

  private native void addNumRuns(long configBuilderHandle, int numRuns);

  private native void addProfilePath(long configBuilderHandle, String profilePath);

  private native void addFrequencyLatencySmoothingFactor(long configBuilderHandle, float smoothingFactor);

  private native void addLatencySmoothingFactor(long configBuilderHandle, float smoothingFactor);

  private native void addThermWindowSize(long configBuilderHandle, int thermWindowSize);

  private native void addCPUThermIndex(long configBuilderHandle, int cpuThermIndex);

  private native void addGPUThermIndex(long configBuilderHandle, int gpuThermIndex);

  private native void addDSPThermIndex(long configBuilderHandle, int dspThermIndex);

  private native void addNPUThermIndex(long configBuilderHandle, int npuThermIndex);

  private native void addTargetThermIndex(long configBuilderHandle, int targetThermIndex);

  private native void addCPUFreqPath(long configBuilderHandle, String cpuFreqPath);

  private native void addGPUFreqPath(long configBuilderHandle, String gpuFreqPath);

  private native void addDSPFreqPath(long configBuilderHandle, String dspFreqPath);

  private native void addNPUFreqPath(long configBuilderHandle, String npuFreqPath);

  private native void addLatencyLogPath(long configBuilderHandle, String latencyLogPath);

  private native void addThermLogPath(long configBuilderHandle, String thermLogPath);

  private native void addFreqLogPath(long configBuilderHandle, String freqLogPath);

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

  private native boolean isValid(long configBuilderHandle);

  private native Object build(long configBuilderHandle);
}
