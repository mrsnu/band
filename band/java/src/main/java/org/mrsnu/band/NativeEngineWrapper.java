package org.mrsnu.band;

import java.util.ArrayList;
import java.util.List;

class NativeEngineWrapper implements AutoCloseable {
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

  public void requestSync(
      Model model, List<Tensor> inputTensors, List<Tensor> outputTensors, RequestOption option) {
    requestSync(nativeHandle, model, inputTensors, outputTensors, option.getTargetWorker(),
        option.getRequireCallback(), option.getSloUs(), option.getSloScale());
  }

  public Request requestAsync(Model model, List<Tensor> inputTensors, RequestOption option) {
    return new Request(requestAsync(nativeHandle, model, inputTensors, option.getTargetWorker(),
        option.getRequireCallback(), option.getSloUs(), option.getSloScale()));
  }

  public List<Request> requestAsyncBatch(
      List<Model> models, List<List<Tensor>> inputTensors, List<RequestOption> options) {
    List<Request> ret = new ArrayList<>();
    int[] targetWorkerList = new int[options.size()];
    boolean[] requireCallbackList = new boolean[options.size()];
    int[] sloUsList = new int[options.size()];
    float[] sloScaleList = new float[options.size()];

    for (int i = 0; i < options.size(); i++) {
      targetWorkerList[i] = options.get(i).getTargetWorker();
      requireCallbackList[i] = options.get(i).getRequireCallback();
      sloUsList[i] = options.get(i).getSloUs();
      sloScaleList[i] = options.get(i).getSloScale();
    }

    int[] results = requestAsyncBatch(nativeHandle, models, inputTensors, targetWorkerList,
        requireCallbackList, sloUsList, sloScaleList);
    for (int jobId : results) {
      ret.add(new Request(jobId));
    }
    return ret;
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
      List<Tensor> outputTensors, int target_worker, boolean require_callback, int slo_us,
      float slo_scale);

  private static native int requestAsync(long engineHandle, Model model, List<Tensor> inputTensors,
      int target_worker, boolean require_callback, int slo_us, float slo_scale);

  private static native int[] requestAsyncBatch(long engineHandle, List<Model> models,
      List<List<Tensor>> inputTensorsList, int[] targetWorkerList, boolean[] requireCallbackList,
      int[] sloUsList, float[] sloScaleList);

  private static native void wait(long engineHandle, int jobId, List<Tensor> outputTensors);
}
