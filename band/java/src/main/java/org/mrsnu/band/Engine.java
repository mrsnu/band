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

  public void requestAsync(Model model, List<Tensor> inputTensors) {
    wrapper.requestAsync(model, inputTensors);
  }

  public void wait(Request request, List<Tensor> outputTensors) {
    wrapper.wait(request, outputTensors);
  }

  public Tensor createInputTensor(Model model, int index) {
    wrapper.createInputTensor(model, index);
    return null;
  }

  public Tensor createOutputTensor(Model model, int index) {
    wrapper.createOutputTensor(model, index);
    return null;
  }
}
