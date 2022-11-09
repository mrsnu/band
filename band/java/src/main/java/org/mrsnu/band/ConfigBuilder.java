package org.mrsnu.band;

public class ConfigBuilder implements BandInterface, AutoCloseable {
  private NativeConfigBuilderWrapper wrapper;

  ConfigBuilder() {
    wrapper = new NativeConfigBuilderWrapper();
  }

  public void addOnline(boolean online) {

  }

  public void addNumWarmups(int numWarmups) {

  }

  public void addNumRuns(int numRuns) {

  }

  public void addCopyComputationRatio(int[] copyComputationRatio) {

  }

  public void addProfileDataPath(String profileDataPath) {

  }

  public void addSmoothingFactor(float smoothingFactor) {

  }

  public void addPlannerLogPath(String plannerLogPath) {

  }

  public void addScheduleWindowSize(int scheduleWindowSize) {

  }

  public void addSchedulers(SchedulerType[] scheduers) {

  }

  public void addPlannerCPUMask(CpuMaskFlags cpuMasks) {

  }

  public void addWorkers(Device[] workers) {

  }

  public void addWorkerCPUMasks(CpuMaskFlags[] cpuMasks) {

  }

  public void addWorkerNumThreads(int[] numThreads) {

  }

  public void addAllowWorkSteal(boolean allowWorkSteal) {

  }

  public void addAvailabilityCheckIntervalMs(int availabilityCheckIntervalMs) {

  }

  public void addMinimumSubgraphSize(int minimumSubgraphSize) {

  }

  public void addSubgraphPreparationType(SubgraphPreparationType subgraphPreparationType) {

  }

  public void addCPUMask(CpuMaskFlags cpuMask) {

  }

  // TODO(widiba03304): Need to move functions below.
  public void addModels(Model[] model) {

  }

  public void addPeriodsMs(int[] modelsPeriodMs) {

  }

  public void addBatchSizes(int[] modelsBatchSize) {

  }

  public void addAssignedWorkers(DeviceWorkerAffinityPair modelsAssignedWorker) {

  }

  public void addSlosUs(long[] modelsSloUs) {

  }

  public void addSlosScale(float modelsSloScale) {

  }

  public Config build() {
    return null;
  }

  public boolean isValid() {
    return true;
  }

  @Override
  public void close() {
    
  }

  public NativeConfigBuilderWrapper getNativeWrapper(NativeWrapper.NativeKey key) {
    return wrapper;
  }
}
