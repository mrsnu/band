package org.mrsnu.band;

import java.util.List;

public class NativeEngineWrapper implements AutoCloseable {
  private long nativeHandle = 0;
  private static final int ERROR_BUFFER_SIZE = 512;

  NativeEngineWrapper(Config config) {
    Band.init();
    nativeHandle = createEngine(config);
  }

  @Override
  public void close() {
    deleteEngine(nativeHandle);
    nativeHandle = 0;
  }

  public void registerModel(Model model) {
    registerModel(nativeHandle, model);
  }

  public int getNumInputTensors(Model model) {
    return getNumInputTensors(nativeHandle, model);
  }

  public int getNumOutputTensors(Model model) {
    return getNumOutputTensors(nativeHandle, model);
  }

  public Tensor createInputTensor(Model model, int index) {
    return new Tensor(createInputTensor(nativeHandle, model, index));
  }

  public Tensor createOutputTensor(Model model, int index) {
    return new Tensor(createOutputTensor(nativeHandle, model, index));
  }

  public void requestSync(Model model, List<Tensor> inputTensors, List<Tensor> outputTensors) {
    requestSync(nativeHandle, model, inputTensors, outputTensors);
  }

  public Request requestAsync(Model model, List<Tensor> inputTensors) {
    return new Request(requestAsync(nativeHandle, model, inputTensors));
  }

  public void wait(Request request, List<Tensor> outputTensors) {
    wait(nativeHandle, request.getJobId(), outputTensors);
  }

  private static native long createEngine(Config config);

  private static native void deleteEngine(long engineHandle);

  private static native void registerModel(long engineHandle, Model model);

  private static native int getNumInputTensors(long engineHandle, Model model);

  private static native int getNumOutputTensors(long engineHandle, Model model);

  private static native long createInputTensor(long engineHandle, Model model, int index);

  private static native long createOutputTensor(long engineHandle, Model model, int index);

  private static native void requestSync(long engineHandle, Model model, List<Tensor> inputTensors,
      List<Tensor> outputTensors);

  private static native int requestAsync(long engineHandle, Model model, List<Tensor> inputTensors);

  private static native void wait(long engineHandle, int jobId, List<Tensor> outputTensors);
}
