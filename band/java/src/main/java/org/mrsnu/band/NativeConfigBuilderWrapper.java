package org.mrsnu.band;

public class NativeConfigBuilderWrapper extends NativeWrapper implements AutoCloseable {
  public enum ConfigField {
    BAND_PROFILE_ONLINE,
    BAND_PROFILE_NUM_WARMUPS,
    BAND_PROFILE_NUM_RUNS,
    BAND_PROFILE_COPY_COMPUTATION_RATIO,
    BAND_PROFILE_SMOOTHING_FACTOR,
    BAND_PROFILE_DATA_PATH,
    BAND_PLANNER_SCHEDULE_WINDOW_SIZE,
    BAND_PLANNER_SCHEDULERS,
    BAND_PLANNER_CPU_MASK,
    BAND_PLANNER_LOG_PATH,
    BAND_WORKER_WORKERS,
    BAND_WORKER_CPU_MASKS,
    BAND_WORKER_NUM_THREADS,
    BAND_WORKER_ALLOW_WORKSTEAL,
    BAND_WORKER_AVAILABILITY_CHECK_INTERVAL_MS,
    BAND_MODEL_MODELS,
    BAND_MODEL_PERIODS,
    BAND_MODEL_BATCH_SIZES,
    BAND_MODEL_ASSIGNED_WORKERS,
    BAND_MODEL_SLOS_US,
    BAND_MODEL_SLOS_SCALE,
    BAND_MINIMUM_SUBGRAPH_SIZE,
    BAND_SUBGRAPH_PREPARATION_TYPE,
    BAND_CPU_MASK,
  }

  public NativeConfigBuilderWrapper() {

  }

  public void addOnline(boolean online) {}

  public void addNumWarmups(int numWarmups) {}

  public void addNumRuns(int numRuns) {}

  public void addCopyComputationRatio(int[] copyComputationRatio) {}

  public void addProfileDataPath(String profileDataPath) {}

  public void addSmoothingFactor(float smoothingFactor) {}

  public void addPlannerLogPath(String plannerLogPath) {}

  public void addScheduleWindowSize(int scheduleWindowSize) {}

  public void addSchedulers(SchedulerType[] scheduers) {}

  public void addPlannerCPUMask(CpuMaskFlags cpuMasks) {}

  public void addWorkers(Device[] workers) {}

  public void addWorkerCPUMasks(CpuMaskFlags[] cpuMasks) {}

  public void addWorkerNumThreads(int[] numThreads) {}

  public void addAllowWorkSteal(boolean allowWorkSteal) {}

  public void addAvailabilityCheckIntervalMs(int availabilityCheckIntervalMs) {}

  public void addMinimumSubgraphSize(int minimumSubgraphSize) {}

  public void addSubgraphPreparationType(SubgraphPreparationType subgraphPreparationType) {}

  public void addCPUMask(CpuMaskFlags cpuMask) {}

  // TODO(widiba03304): Need to move functions below.
  public void addModels(Model[] model) {}

  public void addPeriodsMs(int[] modelsPeriodMs) {}

  public void addBatchSizes(int[] modelsBatchSize) {}

  public void addAssignedWorkers(DeviceWorkerAffinityPair modelsAssignedWorker) {}

  public void addSlosUs(long[] modelsSloUs) {}

  public void addSlosScale(float modelsSloScale) {}

  public Config build() {
    return null;
  }

  public boolean isValid() {
    return true;
  }

  @Override
  public void close() {
    deleteConfigBuilder(nativeHandle);
    nativeHandle = 0;
  }

  public native void deleteConfigBuilder(long configBuilderHandle);
}
