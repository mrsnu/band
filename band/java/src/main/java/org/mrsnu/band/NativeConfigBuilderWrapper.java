package org.mrsnu.band;

public class NativeConfigBuilderWrapper implements AutoCloseable {
  private long nativeHandle = 0;

  public NativeConfigBuilderWrapper() {
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

  public void addCopyComputationRatio(int[] copyComputationRatio) {
    addCopyComputationRatio(nativeHandle, copyComputationRatio);
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

  public void addPlannerCPUMask(CpuMaskFlags cpuMask) {
    addPlannerCPUMask(nativeHandle, cpuMask.getValue());
  }

  public void addWorkers(Device[] workers) {
    int[] tmpArray = new int[workers.length];
    for (int i = 0; i < workers.length; i++) {
      tmpArray[i] = workers[i].getValue();
    }
    addWorkers(nativeHandle, tmpArray);
  }

  public void addWorkerCPUMasks(CpuMaskFlags[] cpuMasks) {
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

  public void addCPUMask(CpuMaskFlags cpuMask) {
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

  private native void addCopyComputationRatio(long configBuilderHandle, int[] copyComputationRatio);

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

  private native void addAvailabilityCheckIntervalMs(long configBuilderHandle, int availabilityCheckIntervalMs);

  private native void addMinimumSubgraphSize(long configBuilderHandle, int minimumSubgraphSize);

  private native void addSubgraphPreparationType(long configBuilderHandle,
      int subgraphPreparationType);

  private native void addCPUMask(long configBuilderHandle, int cpuMask);

  private native boolean isValid(long configBuilderHandle);

  private native Object build(long configBuilderHandle);
}
