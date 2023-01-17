package org.mrsnu.band;

import java.util.List;

public class Engine {
  private NativeEngineWrapper wrapper;

  public Engine(Config config) {
    Band.init();
    wrapper = new NativeEngineWrapper(config);
  }

  public void registerModel(Model model) {
    wrapper.registerModel(model);
  }

  public int getNumInputTensors(Model model) {
    return wrapper.getNumInputTensors(model);
  }

  public int getNumOutputTensors(Model model) {
    return wrapper.getNumOutputTensors(model);
  }

  public void requestSync(Model model, List<Tensor> inputTensors, List<Tensor> outputTensors) {
    wrapper.requestSync(model, inputTensors, outputTensors);
  }

  public Request requestAsync(Model model, List<Tensor> inputTensors) {
    return wrapper.requestAsync(model, inputTensors);
  }

  public List<Request> requestAsyncBatch(List<Model> models, List<List<Tensor>> inputTensorLists) {
    return wrapper.requestAsyncBatch(models, inputTensorLists);
  }

  public void wait(Request request, List<Tensor> outputTensors) {
    wrapper.wait(request, outputTensors);
  }

  public Tensor createInputTensor(Model model, int index) {
    return wrapper.createInputTensor(model, index);
  }

  public Tensor createOutputTensor(Model model, int index) {
    return wrapper.createOutputTensor(model, index);
  }
}
