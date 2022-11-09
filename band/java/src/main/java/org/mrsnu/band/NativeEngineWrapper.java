package org.mrsnu.band;

import java.util.List;

public class NativeEngineWrapper extends NativeWrapper implements AutoCloseable {
  private long errorHandle;
  private static final int ERROR_BUFFER_SIZE = 512;

  NativeEngineWrapper(Config config) {
    Band.init();
    long errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
    init(errorHandle, config);
  }

  private void init(long errorHandle, Config config) {
    long configHandle = config.getNativeWrapper(nativeKey).getNativeHandle();
    this.errorHandle = errorHandle;
    this.nativeHandle = createEngine(configHandle);
  }

  @Override
  public void close() {
    deleteErrorReporter(errorHandle);
    deleteEngine(nativeHandle);
    errorHandle = 0;
    nativeHandle = 0;
  }

  public void registerModel(Model model) {
    long modelHandle = model.getNativeWrapper(nativeKey).getNativeHandle();
    registerModel(nativeHandle, modelHandle);
  }

  public int getNumInputTensors(Model model) {
    long modelHandle = model.getNativeWrapper(nativeKey).getNativeHandle();
    return getNumInputTensors(nativeHandle, modelHandle);
  }

  public int getNumOutputTensors(Model model) {
    long modelHandle = model.getNativeWrapper(nativeKey).getNativeHandle();
    return getNumOutputTensors(nativeHandle, modelHandle);
  }

  public long createInputTensor(Model model, int index) {
    long modelHandle = model.getNativeWrapper(nativeKey).getNativeHandle();
    return createInputTensor(nativeHandle, modelHandle, index);
  }

  public long createOutputTensor(Model model, int index) {
    long modelHandle = model.getNativeWrapper(nativeKey).getNativeHandle();
    return createOutputTensor(nativeHandle, modelHandle, index);
  }

  public void requestSync(Model model, List<Tensor> inputTensors, List<Tensor> outputTensors) {
    for (Tensor tensor : inputTensors) {
      
    }
    long modelHandle = model.getNativeWrapper(nativeKey).getNativeHandle();
    requestSync(nativeHandle, modelHandle, null, null);
  }

  public void requestAsync(Model model, List<Tensor> inputTensors) {

  }

  public void wait(Request request, List<Tensor> outputTensors) {

  }

  private static native long createErrorReporter(int size);

  private static native void deleteErrorReporter(long errorHandle);

  private static native long createEngine(long configHandle);

  private static native void deleteEngine(long engineHandle);

  private static native void registerModel(long engineHandle, long modelHandle);

  private static native int getNumInputTensors(long engineHandle, long modelHandle);

  private static native int getNumOutputTensors(long engineHandle, long modelHandle);

  private static native long createInputTensor(long engineHandle, long modelHandle, int index);

  private static native long createOutputTensor(long engineHandle, long modelHandle, int index);

  private static native void requestSync(long engineHandle, long modelHandle, List<Long> inputTensorHandles,
      List<Long> outputTensorHandles);

  private static native int requestAsync(long engineHandle, long modelHandle, List<Long> inputTensorHandles);

  private static native void wait(long engineHandle, long requestHandle, List<Long> outputTensorHandles);
}
